import sys
sys.path.append('./')
import os
import gym
import gym_gvgai
import numpy as np
from environment.recognition.identifier import Identifier
from environment.LoopEnv import LoopEnv,LoopRandomEnv
# from environment.wrappers import ProcessFrame84
# import pygame
import time
import random
from root import PRJROOT
import json

# edit by sty 5.28
# The GOLOEnv transform the observation into global observation and local observation
# todo: your absolute path
da_environment_path = ""
gvgai_path = ""
class  GOLOEnv(gym.Env):
    # stable baseline will transpose the observation in
    __shapes = {
        'golddigger': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'treasurekeeper': ((16, 19, 27), (16, 5, 5)), # 9 13
        'waterpuzzle': ((16, 23, 61), (16,5, 5)),# 11 30
        'bravekeeper': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'greedymouse': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'trappedhero': (( 16, 31, 55), (16, 5, 5)), # 15, 27
    }

    def __init__(self, game, use_one_hot=False, use_LO=False,algorithm="DQN"):
        """
        :param game: tuple (name, levelid)
        :param use_one_hot: Use use_one_hot input or not
        """
        super(GOLOEnv,self).__init__()
        self.game = game
        self.env = LoopRandomEnv(game)
        self.action_space = self.env.action_space
        self.use_one_hot = use_one_hot
        self.use_LO = use_LO
        self.localObservationSize = 5
        # self.algorithm = algorithm
        self.path=''
        self.log_file = ''
        self.epsoide_list = []
        self.rewards_list = []
        # if self.algorithm=="DQN":
        #     self.path = os.getcwd()+"/log"
        #     self.log_file = self.game+"_"+self.algorithm+"_"+"LO_%s" % str(self.use_LO)+"_"+"Onehot_%s" % str(self.use_one_hot)+".json"
        self.idfy = Identifier(np.zeros(self.env.observation_space.shape),
                               PRJROOT + 'environment/recognition/identifier/%s.json' % game)
        if use_one_hot:
            if use_LO:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=1, shape=GOLOEnv.__shapes[game][0], dtype=np.uint8),
                    'LO': gym.spaces.Box(low=0, high=1, shape=GOLOEnv.__shapes[game][1], dtype=np.uint8)
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=1, shape=GOLOEnv.__shapes[game][0], dtype=np.uint8)
                })
        else:
            if use_LO:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
                    'LO': gym.spaces.Box(low=0, high=255, shape=(4, 50, 50), dtype=np.uint8)
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
                })

        self.img = None
        self.pgscreen = None
        self.go_sfc = None
        self.lo_sfc = None
        # print('ob=', self.observation_space)

    def step(self,action):
        """

        :param action: action for current state
        :return: global observation, local observation,reward, game over?, debug information
        """
        raw_observation, reward, done, info = self.env.step(action)
        self.rewards_list.append((reward,info["winner"]))


        self.img = raw_observation
        mat_go, mat_lo, img_go, img_lo = self.idfy.identify(raw_observation)
        ob = {}
        if self.use_one_hot:
            ob['GO'] = mat_go
            if self.use_LO:
                ob['LO'] = mat_lo
        else:
            ob['GO'] = img_go
            if self.use_LO:
                ob['LO'] = img_lo
        return ob, reward, done, info

    def reset(self):
        self.img = raw_observation = self.env.reset()
        mat_go, mat_lo, img_go, img_lo = self.idfy.identify(raw_observation)
        ob = {}
        if self.use_one_hot:
            ob['GO'] = mat_go
            if self.use_LO:
                ob['LO'] = mat_lo
        else:
            ob['GO'] = img_go
            if self.use_LO:
                ob['LO'] = img_lo
        return ob

    def render(self,mode="human"):
        if self.use_one_hot:
            self.env.render()
        if not self.use_LO:
            self.env.render()
        else:
           self.env.render()


class  testGOLOEnv(gym.Env):
    # stable baseline will transpose the observation in
    __shapes = {
        'golddigger': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'treasurekeeper': ((16, 19, 27), (16, 5, 5)), # 9 13
        'waterpuzzle': ((16, 23, 61), (16,5, 5)),# 11 30
        'bravekeeper': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'greedymouse': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'trappedhero': (( 16, 31, 55), (16, 5, 5)), # 15, 27
    }


    def __init__(self, game, use_one_hot=False, use_LO=False,algorithm="DQN",level=0):
        """
        :param game: tuple (name, levelid)
        :param use_one_hot: Use use_one_hot input or not
        """
        super(testGOLOEnv,self).__init__()
        self.game = game
        # print(self.game,"level: ",level)
        env_name = "gvgai-%s-lvl%d-v0" % (self.game, level)
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.use_one_hot = use_one_hot
        self.use_LO = use_LO
        self.localObservationSize = 5
        self.algorithm = algorithm
        self.path=''
        self.log_file = ''
        self.epsoide_list = []
        self.rewards_list = []
        # if self.algorithm=="DQN":
        #     self.path = os.getcwd()+"/log"
        #     self.log_file = self.game+"_"+self.algorithm+"_"+"LO_%s" % str(self.use_LO)+"_"+"Onehot_%s" % str(self.use_one_hot)+".json"
        self.idfy = Identifier(np.zeros(self.env.observation_space.shape),
                               PRJROOT + 'environment/recognition/identifier/%s.json' % game)
        if use_one_hot:
            if use_LO:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=1, shape=testGOLOEnv.__shapes[game][0], dtype=np.uint8),
                    'LO': gym.spaces.Box(low=0, high=1, shape=testGOLOEnv.__shapes[game][1], dtype=np.uint8)
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=1, shape=testGOLOEnv.__shapes[game][0], dtype=np.uint8)
                })
        else:
            if use_LO:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
                    'LO': gym.spaces.Box(low=0, high=255, shape=(4, 50, 50), dtype=np.uint8)
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
                })

        self.img = None
        self.pgscreen = None
        self.go_sfc = None
        self.lo_sfc = None
        # print('ob=', self.observation_space)

    def step(self,action):
        """

        :param action: action for current state
        :return: global observation, local observation,reward, game over?, debug information
        """
        raw_observation, reward, done, info = self.env.step(action)
        self.rewards_list.append((reward,info["winner"]))

        # if done:
        #     self.epsoide_list.append(self.rewards_list)
        #     if not os.path.exists(self.path+"/"+self.log_file):
        #         os.mknod(self.path+"/"+self.log_file)
        #     with open(self.path+"/"+self.log_file,"w") as f:
        #         json.dump(self.epsoide_list,f)
        #     self.rewards_list = []

        self.img = raw_observation
        mat_go, mat_lo, img_go, img_lo = self.idfy.identify(raw_observation)
        ob = {}
        if self.use_one_hot:
            ob['GO'] = mat_go
            if self.use_LO:
                ob['LO'] = mat_lo
        else:
            ob['GO'] = img_go
            if self.use_LO:
                ob['LO'] = img_lo
        return ob, reward, done, info

    def reset(self):
        self.img = raw_observation = self.env.reset()
        mat_go, mat_lo, img_go, img_lo = self.idfy.identify(raw_observation)
        ob = {}
        if self.use_one_hot:
            ob['GO'] = mat_go
            if self.use_LO:
                ob['LO'] = mat_lo
        else:
            ob['GO'] = img_go
            if self.use_LO:
                ob['LO'] = img_lo
        return ob
    def choose_level(self,level):
        print(self.game,"level: ",level)
        self.env.unwrapped._setLevel(level)

    def render(self,mode="human"):
        if self.use_one_hot:
            self.env.render()
        if not self.use_LO:
            self.env.render()
        else:
           self.env.render()
    def close(self):
        self.env.close()


class  daGOLOEnv(gym.Env):
    # stable baseline will transpose the observation in
    __shapes = {
        'golddigger': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'treasurekeeper': ((16, 19, 27), (16, 5, 5)), # 9 13
        'waterpuzzle': ((16, 23, 61), (16,5, 5)),# 11 30
        'bravekeeper': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'greedymouse': (( 16, 31, 55), (16, 5, 5)), # 15, 27
        'trappedhero': (( 16, 31, 55), (16, 5, 5)), # 15, 27
    }

    def __init__(self, game, use_one_hot=False, use_LO=False,level=0):
        """
        :param game: tuple (name, levelid)
        :param use_one_hot: Use use_one_hot input or not
        """
        super(daGOLOEnv,self).__init__()
        self.game = game
        self.level=level
        # print(self.game,"level: ",level)
        env_name = "gvgai-%s-lvl%d-v0" % (self.game, self.level)
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.use_one_hot = use_one_hot
        self.use_LO = use_LO
        self.localObservationSize = 5
        self.path=''
        self.log_file = ''
        self.epsoide_list = []
        self.rewards_list = []
        self.da_list = []
        self.change_level()

        
        self.idfy = Identifier(np.zeros(self.env.observation_space.shape),
                               PRJROOT + 'environment/recognition/identifier/%s.json' % game)
        if use_one_hot:
            if use_LO:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=1, shape=daGOLOEnv.__shapes[game][0], dtype=np.uint8),
                    'LO': gym.spaces.Box(low=0, high=1, shape=daGOLOEnv.__shapes[game][1], dtype=np.uint8)
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=1, shape=daGOLOEnv.__shapes[game][0], dtype=np.uint8)
                })
        else:
            if use_LO:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
                    'LO': gym.spaces.Box(low=0, high=255, shape=(4, 50, 50), dtype=np.uint8)
                })
            else:
                self.observation_space = gym.spaces.Dict({
                    'GO': gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
                })

        self.img = None
        self.pgscreen = None
        self.go_sfc = None
        self.lo_sfc = None
        # print('ob=', self.observation_space)

    def step(self,action):
        """

        :param action: action for current state
        :return: global observation, local observation,reward, game over?, debug information
        """
        raw_observation, reward, done, info = self.env.step(action)
        self.rewards_list.append((reward,info["winner"]))
        self.img = raw_observation
        mat_go, mat_lo, img_go, img_lo = self.idfy.identify(raw_observation)
        ob = {}
        if self.use_one_hot:
            ob['GO'] = mat_go
            if self.use_LO:
                ob['LO'] = mat_lo
        else:
            ob['GO'] = img_go
            if self.use_LO:
                ob['LO'] = img_lo
        return ob, reward, done, info

    def reset(self):
        # self.choose_level(1)
        self.choose_level()

        self.img = raw_observation = self.env.reset()
        mat_go, mat_lo, img_go, img_lo = self.idfy.identify(raw_observation)
        ob = {}
        if self.use_one_hot:
            ob['GO'] = mat_go
            if self.use_LO:
                ob['LO'] = mat_lo
        else:
            ob['GO'] = img_go
            if self.use_LO:
                ob['LO'] = img_lo
        return ob

    def choose_level(self):
        # print(self.game,"level: ",level)
        choice= np.random.choice(self.da_list)
        print(choice)
        self.env.unwrapped._setLevel(choice)
    
    def change_level(self):
        __available_level = {
            'golddigger': [267 , 124],
            'treasurekeeper': [60 , 52],
            'waterpuzzle': [151 , 119],
            'greedymouse' : [213 , 126],
            'trappedhero':[179 , 157],
            'bravekeeper':[241 , 236]

        }
        # level = np.random.choice([0,1])
        # game = self.game
        # num_available_level = __available_level[game][level]
        # available_list = np.arange(0,num_available_level,1)
        # choose_da = np.random.choice(available_list)
        
        da_path = da_environment_path+"%s_v0/da/"%self.game


        for level in [0,1]:
            for da_level in range(__available_level[self.game][level]):
                    txt_name = self.game+"_lvl%d_%d.txt"%(level,da_level)
                    path =da_path+txt_name
                    self.da_list.append(path)


    def recover(self):
        with open(da_environment_path+"golddigger_v0/golddigger_lvl%d.txt"%self.level,"r") as f:
            with open(gvgai_path+"GVGAI_GYM/gym_gvgai/envs/games/golddigger_v0/golddigger_lvl%d.txt"%self.level,"w") as f2:
                origin_txt = f.read()
                f2.write(origin_txt)

    def render(self,mode="human"):
        if self.use_one_hot:
            self.env.render()
        if not self.use_LO:
            self.env.render()
        else:
           self.env.render()
    def close(self):
        self.env.close()

if __name__ == '__main__':

    env = daGOLOEnv('golddigger',level=1)
    
    for i in range(10):
   
        env.reset()
        while True:
            action = random.randrange(5)
            ob, reward, done, info = env.step(action)
            env.render()
            # print(reward)
            if done:
                break