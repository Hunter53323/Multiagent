"能源系统多智能体强化学习环境--单电池"

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Agent import Battery, User

class Multiagent_energy(gym.Env):

    def __init__(self,mode = "test"):
        self.run_mode = mode
        #24个时间段的电价
        self.electricity_price_all = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,1,1,1,1,0.6,0.6,0.6,0.6,0.6,1,1,1,1,0.6,0.6,0.6,0.3]
        self.gas_price_all = 0.3

        self.current_time_period = 0
        self.current_electricity_price = None
        self.current_gas_price = None

        reward = 0

        self.agents = []  
        self.users = []
        self.users.append(User(self.run_mode))
        if self.run_mode == "test":
            Agent = Battery()
            self.agents.append(Agent)

        self.observation = {}
        #observation返回的参数数目
        self.observation_dim_all = 6 
        self.action_space_battery = self.agents[0].action_space

        self.not_done = True
        
        
        
    def step(self,actions):
        assert self.not_done, "24个时刻已经运行结束，请重置环境！"
        """
        self.current_electricity_price = self.electricity_price_all[self.current_time_period]
        self.current_gas_price = self.gas_price_all
        current_price = self._get_current_price()
        """
        if self.run_mode == "test":
            current_action = actions
        battery_electricity, battery_charge_number, battery_reward = self.agents[0].step(current_action)
        if battery_charge_number > 0:
            #买电的消耗
            buy_electricity_cost = -1 * self.observation['current_electricity_price'] * battery_charge_number
        else:
            buy_electricity_cost = 0
        #成本，测试模式下
        cost_all = buy_electricity_cost
        
        #用户满意度，测试模式下
        satisfaction = self.users[0].judge_satisfaction(battery_charge_number, 0)

        #惩罚
        punish = battery_reward
        earnings = 0
        reward = self.calculate_reward(cost_all, satisfaction, earnings, punish)
        """
        for obs in self.observation:
            if obs == ''
        """

        #当前时刻前进到下一个时刻
        self.current_time_period += 1
        done = bool(
            self.current_time_period >= 23
        )
        next_price = self._get_current_price()
        next_demand = self._getdemand()
        self.observation = merge(next_price, next_demand, battery_electricity)
        # if not done:
        #     next_price = self._get_current_price()
        #     next_demand = self._getdemand()
        #     self.observation = merge(next_price, next_demand, battery_electricity)
        # else:
        #     self.observation = {}
        #     self.not_done = False
        if done:
            self.not_done = False

        return self.observation, reward, done, {}
    #测试模式下的该函数
    def _getdemand(self):
        return self.users[0].generate_demand()

    def calculate_reward(self, cost, satisfactory, earnings, punish):
        return cost + satisfactory + earnings + punish

    def _get_current_price(self):
        self.current_electricity_price = self.electricity_price_all[self.current_time_period]
        self.current_gas_price = self.gas_price_all
        return {'current_electricity_price':self.current_electricity_price, 'current_gas_price':self.current_gas_price}

    def reset(self):
        #reset之后返回当前的状态
        self.current_time_period = 0
        self.not_done = True
        
        #需求：3
        demand = self.users[0].reset()
        #电池：1
        battery_electricity = self.agents[0].reset()
        #当前价格：2
        current_price = self._get_current_price()
        self.observation = merge(current_price, demand, battery_electricity)

        return self.observation
    
    def render(self):
        for obs in self.observation:
            print(obs+":", self.observation[obs], end = ' ')
        print("Time:", self.current_time_period, end = ' ')
        print()

    def close(self):
        pass

def merge(*dicts):
    res = {}
    for dic in dicts:
        res = {**res, **dic}
    #res = (**dict1, **dict2)
    return res

if __name__ == "__main__":
    import random
    bat = Multiagent_energy()
    bat.reset()
    bat.render()
    for i in range(15):
        bat.step(random.randrange(21))
        bat.render()
    bat.reset()
    print()
    bat.render()
    
    while True:
        bat.step(random.randrange(21))
        bat.render()