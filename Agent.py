"""
Agent包含Battery, WaterTank, CHP, Boiler, User
"""

from gym import spaces
import random
import numpy as np


class BaseAgent():
    def __init__(self, name = None):
        self.electricity = None
        self.heat = None
        self.gas = None

        self.action_space = None
        self.observation_space = None

        self.name = name

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

class Battery(BaseAgent):
    def __init__(self):
        super().__init__(name = "battery")
        self.electricity = 3
        #动作空间有无动作、充放电0-1、卖电0-1
        self.action_space = spaces.Discrete(31)
        #当前电量状态为满电的百分比,一维
        #该智能体能观测到的观测空间为当前电价、自己的电量、用户需求
        self.observation_space = spaces.Discrete(3)

        #常数定义
        self.max_electricity = 10
        self.min_electricity = 0.0
        self.charge_discharge_max = 10
        self.eta = 0.98

    def _judge_constraint(self):
        if (self.electricity > self.max_electricity) \
            or (self.electricity < self.min_electricity):
            self.electricity = max(self.min_electricity, min(self.electricity, self.max_electricity))
            reward = -10000
        else:
            reward = 0
        return reward

    def step(self, action):
        action = np.argmax(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        #将动作映射成具体的充放电数据
        charge_number, sell_number = self._get_charge_number(action)
        old_elec = self.electricity
        self.electricity = round(self.electricity + charge_number - sell_number,2)
        #电量约束,超出约束则给予惩罚
        reward = self._judge_constraint()
        if reward < 0 and sell_number == 0 and self.electricity == 0:
            charge_number = round(-old_elec,2)
        elif reward < 0 and charge_number == 0:
            sell_number = round(old_elec,2)

        battery_electricity = {'battery_electricity':self.electricity}
        return battery_electricity, charge_number, reward, sell_number


    def reset(self):
        self.electricity = 3.0

        battery_electricity = {'battery_electricity':self.electricity}
        return battery_electricity

    def _get_charge_number(self, action):
        charge_number = 0
        sell_number = 0
        if action > 20:
            sell_number = (action - 20)/10.0
        else:
            charge_number = (action - 10)/10.0
        return charge_number, sell_number

    def get_other_electricity(self, elec):
        self.electricity = round(self.electricity + elec - 0.1, 2)
        reward = self._judge_constraint()
        battery_electricity = {'battery_electricity':self.electricity}
        return battery_electricity, reward

    def render(self):
        #环境的render模块里面调用输出相应的数据
        pass

class WaterTank(BaseAgent):
    def __init__(self):
        super().__init__(name = "watertank")
        self.heat = 3

        #动作空间为放热从0-1
        self.action_space = spaces.Discrete(11)
        #该智能体能观测到的观测空间为自己的热量、用户需求
        self.observation_space = spaces.Discrete(2)

        #常数定义
        self.max_heat = 10
        self.min_heat = 0.0
        self.heat_release_max = 10

    def get_other_heat(self, heat_output):
        #输入其他智能体的热量
        self.heat = round(self.heat + heat_output - 0.1, 2)
        reward = self._judge_constraint()
        watertank_heat = {'watertank_heat': self.heat}
        return watertank_heat, reward

    def _judge_constraint(self):
        if (self.heat > self.max_heat) \
            or (self.heat < self.min_heat):
            self.heat = max(self.min_heat, min(self.heat, self.max_heat))
            reward = -10000
        else:
            reward = 0
        return reward

    def step(self, action):
        action = np.argmax(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        release_number = round(action / 10.0, 2)
        self.heat = self.heat - release_number
        #热量约束，超出约束则给予惩罚
        reward = self._judge_constraint()
        if reward < 0:
            release_number = self.heat + release_number

        watertank_heat = {'watertank_heat': self.heat}
        return watertank_heat, release_number, reward

    def reset(self):
        self.heat = 3.0
        watertank_heat = {'watertank_heat': self.heat}
        return watertank_heat

class CHP(BaseAgent):
    def __init__(self):
        super().__init__(name = "chp")
        
        #动作空间为买气量从0-1
        self.action_space = spaces.Discrete(11)
        #该智能体能观测到的观测空间为当前电、气价、用户热和电需求
        self.observation_space = spaces.Discrete(4)

        #常数定义
        self.max_generate_speed = 2

    def step(self, action):
        action = np.argmax(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        gas_consumption = round(action / 10.0,2)
        electricity_generate, heat_generate = self._generate_heat_and_electricity(gas_consumption)

        CHP_observation = {"CHP_electricity_generate": electricity_generate, "CHP_heat_generate": heat_generate}
        reward = 0#TODO:reward考虑如何计算
        return CHP_observation, gas_consumption, reward 
 

    def reset(self):
        CHP_observation = {"CHP_electricity_generate": 0, "CHP_heat_generate": 0}
        return CHP_observation
        

    def _generate_heat_and_electricity(self, gas):
        return round(0.35*gas,2), round(0.35*gas,2)

class Boiler(BaseAgent):
    def __init__(self):
        super().__init__(name="boiler")
        
        #产热从0-1
        self.action_space = spaces.Discrete(11)
        #该智能体能观测到的观测空间为当前气价、用户需求
        self.observation_space = spaces.Discrete(2)

        #常数定义
        self.max_generate_speed = 2

    def step(self, action):
        action = np.argmax(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        gas_consumption = round(action / 10.0,2)
        heat_generate = round(self._generate_heat(gas_consumption),2)

        Boiler_observation = {"boiler_heat_generate": heat_generate}
        reward = 0
        return Boiler_observation, gas_consumption, reward 

    def reset(self):
        Boiler_observation = {"boiler_heat_generate": 0}
        return Boiler_observation

    def _generate_heat(self, gas):
        #一单位气产生多少单位的热
        # return 0.8*gas
        return gas
class User(BaseAgent):
    def __init__(self, mode = "test"):
        super().__init__(name = "user")
        self.satisfaction = None
        self.electricity_demand = None
        self.gas_demand = None
        self.heat_demand = None
        
        self.electricity_demand_max = 1
        self.gas_demand_max = 1
        self.heat_demand_max = 1

        #确保需求和评判按照顺序执行
        self.process = 0

        #运行模式是否为测试模式
        self.run_mode = mode

    def random_demand(self,sup):
        return random.randrange(int(10*sup))/10.0

    def generate_demand_fixed(self, ctime):
        assert self.process == 0, "请先将上一步的生成需求进行满意度评判"
        self.process = 1

        elec_demand_fixed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.7, 1.0, 0.6, 0.5, 0.4, 0.4, 1.0, 0.8, 0.8, 0.7, 0.6, 0.2, 0.4, 0.3, 0.5, 0.3, 0.1, 0.4]
        heat_demand_fixed = [0.3, 0.5, 0.4, 0.6, 0.2, 0.3, 0.1, 0.7, 0.3, 1.0, 0.4, 0.2, 1.0, 0.2, 0.7 , 0.6, 0.7, 0.5, 0.5, 0.5, 0.3, 0.2, 0.7, 0.5]
        demand = {}
        self.gas_demand = 0
        self.heat_demand = heat_demand_fixed[ctime]
        self.electricity_demand = elec_demand_fixed[ctime]
        demand['electricity_demand'] = self.electricity_demand
        demand['gas_demand'] = self.gas_demand
        demand['heat_demand'] = self.heat_demand
        return demand

    def generate_demand(self):
        assert self.process == 0, "请先将上一步的生成需求进行满意度评判"
        self.process = 1

        self.electricity_demand = self.random_demand(self.electricity_demand_max)
        # self.gas_demand = random_demand(self.gas_demand_max)
        self.gas_demand = 0
        self.heat_demand = self.random_demand(self.heat_demand_max)
        if self.run_mode == "test":
            self.gas_demand = 0
            self.heat_demand = 0
            self.electricity_demand = random.choice([self.electricity_demand,0])
        demand = {}
        demand['electricity_demand'] = self.electricity_demand
        demand['gas_demand'] = self.gas_demand
        demand['heat_demand'] = self.heat_demand

        return demand
        #return self.electricity_demand, self.gas_demand, self.heat_demand

    def judge_satisfaction(self, electricity_provide, heat_provide, factor):
        assert self.process == 1, "判断前需先生成需求"
        self.process = 0
        extra_heat = max(0, heat_provide - self.heat_demand)
        extra_elec = max(0, electricity_provide - self.electricity_demand)
        elec_satisfaction = max(2*(electricity_provide - self.electricity_demand), 1 * (self.electricity_demand - electricity_provide))
        heat_satisfaction = max(2*(heat_provide - self.heat_demand), 1 * (self.heat_demand - heat_provide))
        #满意度的初始值和test模式中智能体的个数有关
        satisfaction = factor * (4 - heat_satisfaction - elec_satisfaction)
        return satisfaction, extra_heat, extra_elec

    def reset(self):
        self.process = 0
        return self.generate_demand_fixed(0)

class SolarPanel(BaseAgent):
    def __init__(self):
        super().__init__(name = "solarpanel")

    def generate(self, time):
        if time < 5 or time > 19:
            return 0.0
        generate_elec = round(random.randint(7 - abs(time - 12), 10 - abs(time - 12))/10.0, 2)
        return generate_elec

    def generate_norandom(self, time):
        if time < 5 or time > 19:
            return 0.0
        generate_elec = round(7 - abs(time - 12), 2)
        generate_elec_obs = {"solarpanel_generate_electricity":generate_elec}
        return generate_elec

    def reset(self):
        generate_elec_obs = {"solarpanel_generate_electricity":0}
        return generate_elec_obs



if __name__ == "__main__":
    #Battery, WaterTank, CHP, Boiler, User
    agentlist = ["Battery", "WaterTank", "CHP", "Boiler"]
    users = ["user"]
    for i in agentlist:
        agent = None
        exec("agent = "+i+"()")
        action_list = np.zeros(agent.action_space.n)
        action = random.randrange(agent.action_space.n)
        action_list[action] = 1
        value1, value2, reward = agent.step(action_list)
        print("动作：", action, end = " ")
        print(value1, end = " ")
        print("change:", value2, end = " ")
        print("reward:", reward)
    agent = SolarPanel()
    print("产生的电力：",agent.generate(random.randint(0,24)))
    
