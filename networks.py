# -*- coding:utf-8 _*-
"""

@file: networks.py

@Modify Time       @Author   @Version    @Description
------------       -------   --------    ------------
 21:05    PneuC     1.0         None
"""
from itertools import product

import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SingleConvExtractor(BaseFeaturesExtractor):
    # onehot_size = {
    #     'golddigger': (31, 55),
    #     'waterpuzzle': (23, 61),
    #     'treasurekeeper': (19, 27),
    #     'bravekeeper':(31, 55),
    #     'greedymouse': (31, 55),
    #     'trappedhero': (31, 55),
    # }

    def __init__(self, observation_space: gym.spaces.Dict, one_hot):
        # print(observation_space)
        # print(type(observation_space))
        super(SingleConvExtractor, self).__init__(observation_space, features_dim=64)

        self.one_hot = one_hot
        C, W, H = observation_space['GO'].shape
        self.in_channels = C
        if self.one_hot:
            linear_in = 64 * (((W-1) // 3 - 1) // 2) * (((H-1) // 3 - 1) // 2)
            self.Conv_G = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=3), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(linear_in, 256), nn.ReLU()
            )
        else:
            self.Conv_G = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 256), nn.ReLU()
            )
        self.FC = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

    def forward(self, state):
        # print(torch.max(state['GO']), state['GO'].dtype)
        # for k, channel in enumerate(state['GO'][0]):
        #     h, w = channel.shape
        #     print(f'------------------------type{k}------------------------')
        #     for i in range(h):
        #         for j in range(w):
        #             print(round(channel[i][j].item()), end='')
        #         print()
        # print(torch.sum(state['GO'], 1))
        # print()
        conv_out = self.Conv_G(state['GO'])
        # conv_out = self.Conv_G(torch.tensor(state['GO'], device=self.device).float())
        return self.FC(conv_out)


class DoubleInputConvExatractor(SingleConvExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, one_hot):
        super(DoubleInputConvExatractor, self).__init__(observation_space, one_hot)
        # # self.feature_dim = 64
        if self.one_hot:
            self.Conv_L = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 3, 1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 3 * 3, 256), nn.ReLU()
            )
        else:
            self.Conv_L = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 6, 4), nn.ReLU(),
                nn.Conv2d(32, 32, 4, 2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 5 * 5, 256), nn.ReLU()
            )
        self.FC = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )

    def forward(self, state):
        go_out = self.Conv_G(state['GO'])
        lo_out = self.Conv_L(state['LO'])
        # go_out = self.Conv_G(torch.tensor(state['GO'], self.device).float())
        # lo_out = self.Conv_L(torch.tensor(state['LO'], self.device).float())
        return self.FC(torch.cat([go_out, lo_out], dim=1))
