import sys
import os
import gym
import gym_gvgai
import numpy as np
from environment.recognition.identifier import Identifier
# edit by sty 5.29 
# The LoopEnv change the training levels continuely.
# The para alternative_epochs defines how often the available levels alternate during training
class  LoopEnv(gym.Env):
    # stable baseline will transpose the observation in
    def __init__(self, game):
        """
        :param game: tuple (name, levelid)
        :param image: Use image input or not
        """
        super(LoopEnv,self).__init__()

        
        self.alternative_epochs = 5
        self.cnt = 1
        self.available_lv = [0, 1]
        self.lv_cnt = 0
        self.game = game
        env_name = "gvgai-%s-lvl%d-v0" % (self.game, self.available_lv[self.lv_cnt])
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self,action):
        return self.env.step(action)

    def reset(self):
        if self.cnt % self.alternative_epochs == 0:
            self.lv_cnt = (self.lv_cnt + 1) % len(self.available_lv)
            self.env.unwrapped._setLevel( self.available_lv[self.lv_cnt])
        self.cnt += 1
        return self.env.reset()

    def render(self,mode="human"):
        self.env.render()

    def close(self):
        self.env.close()


class LoopRandomEnv(gym.Env):
    # randomly choosing level
    # stable baseline will transpose the observation in
    def __init__(self, game):
        """
        :param game: tuple (name, levelid)
        :param image: Use image input or not
        """
        super(LoopRandomEnv, self).__init__()

        self.alternative_epochs = 5
        self.cnt = 1
        self.available_lv = [0, 1]
        self.lv_cnt = 0
        self.game = game
        env_name = "gvgai-%s-lvl%d-v0" % (self.game, self.available_lv[self.lv_cnt])
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        choice = np.random.choice(self.available_lv)
        print("level : ",choice)
        self.env.unwrapped._setLevel(int(choice))
        return self.env.reset()

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()