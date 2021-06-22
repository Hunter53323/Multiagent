"能源系统多智能体强化学习环境--单电池"

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Agent import Battery


class Multiagent_energy(gym.Env):

    def __init__(self,mode = "test"):
        self.battery_electricity = 0
        self.electricity_price_all = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,1,1,1,1,0.6,0.6,0.6,0.6,0.6,1,1,1,1,0.6,0.6,0.6,0.3]
        self.gas_price_all = 0.3

        self.agent = []  #暂定用list实现
        if mode == "test":
            Agent = Battery()
            self.agent.append(Agent)
        
    def step(self,action):
        pass

    def reset(self):
        pass
    
    def render(self):
        pass

    def close(self):
        pass