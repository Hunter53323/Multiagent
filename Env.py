"能源系统多智能体强化学习环境--单电池"

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Agent import Battery, WaterTank, CHP, Boiler, User, SolarPanel
import defination

class Multiagent_energy(gym.Env):

    def __init__(self,mode = "test"):
        self.run_mode = mode
        #24个时间段的电价
        #self.electricity_price_all = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,1,1,1,1,0.6,0.6,0.6,0.6,0.6,1,1,1,1,0.6,0.6,0.6,0.3]
        self.electricity_price_all = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,1.0,1.0,1.0,1.0,0.6,0.6,0.6,0.6,0.6,1.0,1.0,1.0,1.0,0.6,0.6,0.6,0.3]

        self.gas_price_all = 0.3

        self.current_time_period = 0
        self.current_electricity_price = None
        self.current_gas_price = None

        self.agents = [Battery(), WaterTank(), CHP(), Boiler()] 
        self.agents_name = defination.AGENT_NAME 
        # self.users = [User(self.run_mode)]
        self.users = [User("normal")]
        self.panels = [SolarPanel()]

        self.observation = {}
        #observation返回的参数数目
        
        # self.action_space_battery = self.agents[0].action_space
        self.action_space = {}
        # self.action_space['battery'] = self.agents[0].action_space
        for ag in self.agents:
            self.action_space[ag.name] = ag.action_space.n
        #取得维数使用action_sapce.n
        self.observation_space = {}
        # self.observation_space['battery'] = self.agents[0].observation_space.n
        for ag in self.agents:
            self.observation_space[ag.name] = ag.observation_space.n
        self.not_done = True

        #常量定义
        self.earning_factor = 5
        self.cost_factor = 5   #TODO:加大
        # self.satisfactory_factor = 30
        self.satisfactory_factor = 10


        self.save_dict = {}
        
        
    def get_agent_names(self):
        return [ag.name for ag in self.agents]
    
    def _cal_costs(self, elec, gas):
        # cost一定是负的
        buy_electricity_cost = self.current_electricity_price * elec
        buy_gas_cost = self.current_gas_price * gas
        return round(-self.cost_factor * (buy_electricity_cost + buy_gas_cost),2)

    def _cal_earning(self, sell_number):
        return round(sell_number * self.current_electricity_price * self.earning_factor * 0.5,2)

    def _cal_punish(self, *punish):
        return sum(punish)

    def step_boiler(self, actions):
        assert self.not_done, "24个时刻已经运行结束，请重置环境！"
        solar_generate_elec = self.panels[0].generate(self.current_time_period)
        for key, value in actions.items():
            if key == "battery":
                battery_electricity, battery_charge_number, battery_punish, battery_sell_number = self.agents[0].step(value)
            elif key == "watertank":
                watertank_heat, watertank_release, watertank_punish = self.agents[1].step(value)
            elif key == "chp":
                chp_observation, chp_gas_consumption, chp_punish = self.agents[2].step(value)
            elif key == "boiler":
                boiler_heat_generate, boiler_gas_consumption, boiler_punish = self.agents[3].step(value)
            else:
                raise Exception("请检查智能体名称！")

        satisfaction, extra_heat, extra_elec = self.users[0].judge_satisfaction(0, boiler_heat_generate["boiler_heat_generate"], self.satisfactory_factor)
        # print(self.observation["heat_demand"], boiler_heat_generate["boiler_heat_generate"], satisfaction)
        

        #当前时刻前进到下一个时刻
        self.current_time_period += 1
        done = bool(
            self.current_time_period >= 23
        )
        next_price = self._get_current_price()
        next_demand = self._getdemand()
        self.observation = merge(next_price, next_demand, battery_electricity, watertank_heat, chp_observation, boiler_heat_generate)

        if done:
            self.not_done = False
        render_list = {"watertank_release":watertank_release, "boiler_gas_consumption":boiler_gas_consumption}
        
        cost_and_earning_dict = {"heat_generate":boiler_heat_generate["boiler_heat_generate"]}
        self.save(merge(render_list, self.observation, cost_and_earning_dict))

        return self.observation, satisfaction, done, {}
        # return self.observation, reward, done, render_list

    def step(self, actions):
        assert self.not_done, "24个时刻已经运行结束，请重置环境！"
        solar_generate_elec = self.panels[0].generate(self.current_time_period)
        for key, value in actions.items():
            if key == "battery":
                battery_electricity, battery_charge_number, battery_punish, battery_sell_number = self.agents[0].step(value)
            elif key == "watertank":
                watertank_heat, watertank_release, watertank_punish = self.agents[1].step(value)
            elif key == "chp":
                chp_observation, chp_gas_consumption, chp_punish = self.agents[2].step(value)
            elif key == "boiler":
                boiler_heat_generate, boiler_gas_consumption, boiler_punish = self.agents[3].step(value)
            else:
                raise Exception("请检查智能体名称！")
        total_electricity_generate = round(solar_generate_elec - min(battery_charge_number, 0) + chp_observation["CHP_electricity_generate"],2)
        total_heat_generate = round(boiler_heat_generate["boiler_heat_generate"] + watertank_release + chp_observation["CHP_heat_generate"],2)

        satisfaction, extra_heat, extra_elec = self.users[0].judge_satisfaction(total_electricity_generate, total_heat_generate, self.satisfactory_factor)
        # 生产的电和热供给用户之后余量生成新的电池和水箱容量
        # if battery_charge_number < 0:
        #     extra_elec = -0.1#TODO:放电的最终判断有些问题
        battery_electricity, battery_punish = self.agents[0].get_other_electricity(extra_elec)
        watertank_heat, watertank_punish = self.agents[1].get_other_heat(extra_heat)
        # 总成本
        cost_all = self._cal_costs(max(0,battery_charge_number), chp_gas_consumption + boiler_gas_consumption)
        # cost_all = 0  #TODO：特殊条件处理
        #惩罚
        punish = self._cal_punish(battery_punish, watertank_punish, chp_punish, boiler_punish)
        earnings = self._cal_earning(battery_sell_number)
        #计算总体的reward
        

        #当前时刻前进到下一个时刻
        self.current_time_period += 1
        done = bool(
            self.current_time_period >= 23 or
            punish < -5000 #done的新步骤
        )
        punish = max(punish, -100)  #惩罚不给加
        reward = self.calculate_reward(cost_all, satisfaction, earnings, punish)
        next_price = self._get_current_price()
        next_demand = self._getdemand()
        self.observation = merge(next_price, next_demand, battery_electricity, watertank_heat, chp_observation, boiler_heat_generate)

        if done:
            self.not_done = False
        render_list = {"battery_charge_number":battery_charge_number, "battery_sell_number":battery_sell_number, "watertank_release":watertank_release, "boiler_gas_consumption":boiler_gas_consumption, "solar_generate_elec":solar_generate_elec}
        
        cost_and_earning_dict = {"cost":cost_all, "earning":earnings, "elec_generate":total_electricity_generate, "heat_generate":total_heat_generate}
        self.save(merge(render_list, self.observation, cost_and_earning_dict))

        return self.observation, reward, done, {}
        # return self.observation, reward, done, render_list
    
    def _getdemand(self):
        # return self.users[0].generate_demand()
        return self.users[0].generate_demand_fixed(self.current_time_period)

    def calculate_reward(self, cost, satisfactory, earnings, punish):
        return cost + satisfactory + earnings + punish
        # return cost

    def _get_current_price(self):
        self.current_electricity_price = self.electricity_price_all[self.current_time_period]
        self.current_gas_price = self.gas_price_all
        return {'current_electricity_price':self.current_electricity_price, 'current_gas_price':self.current_gas_price}

    def reset(self):
        #reset之后返回当前的状态
        self.current_time_period = 0
        self.not_done = True
        #当前价格：2
        current_price = self._get_current_price()        
        #需求：3
        demand = self.users[0].reset()
        #电池：1
        battery_electricity = self.agents[0].reset()
        #水箱：1
        watertank_heat = self.agents[1].reset()
        #热电联产：2
        chp_observation = self.agents[2].reset()
        #锅炉：1
        boiler_heat_generate = self.agents[3].reset()

        self.observation = merge(current_price, demand, battery_electricity, watertank_heat, chp_observation, boiler_heat_generate)
        self.save_dict = {}
        self.save(self.observation)

        return self.observation
    
    def render(self):
        for key, value in self.observation.items():
            print(key[:]+":", value, end = ' ')            
            #print(key[:6]+":", value, end = ' ')
        print("Time:", self.current_time_period, end = ' ')
        print()

    def render_custom(self, render_th):
        for obs, value in render_th.items():
            print(obs+":", value, end = ' ')
        print("Time:", self.current_time_period, end = ' ')
        print()

    def save(self, obs_dict):
        for key, value in obs_dict.items():
            if key in self.save_dict.keys():
                self.save_dict[key].append(value)
            else:
                self.save_dict[key] = [value]

    def get_save(self):
        assert not self.not_done, "环境仍在进行，请等待环境运行完成再获取！"
        return self.save_dict


def merge(*dicts):
    """
    多个字典合成一个字典
    """
    res = {}
    for dic in dicts:
        res = {**res, **dic}
    return res

if __name__ == "__main__":
    import random
    bat = Multiagent_energy()
    bat.reset()
    bat.render()
    print(bat.action_space)
    print(bat.observation_space)

    def generate_action(agents):
        action_dict = {}
        for ag in agents:
            action = random.randint(0,ag.action_space.n-1)
            action = defination.action_to_list(action, ag.action_space.n)
            action_dict[ag.name] = action
        return action_dict
    
    for i in range(15):
        bat.step(generate_action(bat.agents))
        bat.render()
    bat.reset()
    print()
    bat.render()
    
    while True:
        bat.step(generate_action(bat.agents))
        bat.render()
    