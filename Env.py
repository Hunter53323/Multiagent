"能源系统多智能体强化学习环境"

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Agent import Battery, WaterTank, CHP, Boiler, User, SolarPanel
import defination

class Multiagent_energy(gym.Env):

    def __init__(self,mode = "test", id_num = 1):
        self.run_mode = mode
        #24个时间段的电价
        self.electricity_price_all = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,1.0,1.0,1.0,1.0,0.6,0.6,0.6,0.6,0.6,1.0,1.0,1.0,1.0,0.6,0.6,0.6,0.3,0]
        self.gas_price_all = 0.3

        self.current_time_period = 0
        self.current_electricity_price = None
        self.current_gas_price = None

        self.agents = []
        self.users = []
        self.panels = []
        for i in range(id_num):
            self.agents.extend([Battery(id=i+1), WaterTank(id=i+1), CHP(id=i+1), Boiler(id=i+1)])
            self.users.append(User("human"))
            self.panels.append(SolarPanel())
        self.agents_name = [ag.name for ag in self.agents] 

        #observation为观测到的所有状态量
        self.observation = {}
        #动作空间和状态空间的定义
        self.action_space = {}
        self.observation_space = {}
        for ag in self.agents:
            self.action_space[ag.name] = ag.action_space.n
        for ag in self.agents:
            self.observation_space[ag.name] = ag.observation_space.n

        self.not_done = True
        self.id_num = id_num

        #常量定义
        self.earning_factor = 5#原来是5 5 10
        self.cost_factor = 5
        self.satisfactory_factor = 10

        #用于保存和输出的参数
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

    def generate_elec_solar(self, ctime):
        elect = 0
        for so in self.panels:
            elect += so.generate(ctime)
        return elect

    def step(self, actions):
        assert self.not_done, "24个时刻已经运行结束，请重置环境！"
        solar_generate_elec = self.generate_elec_solar(self.current_time_period)
        ###参数初始化，以备后面使用#####
        #运行参数
        battery_electricity,battery_charge_number,battery_punish, battery_sell_number = [0]*self.id_num,[0]*self.id_num,[0]*self.id_num,[0]*self.id_num
        watertank_heat, watertank_release, watertank_punish = [0]*self.id_num,[0]*self.id_num,[0]*self.id_num
        chp_observation, chp_gas_consumption, chp_punish = [0]*self.id_num,[0]*self.id_num,[0]*self.id_num
        boiler_heat_generate, boiler_gas_consumption, boiler_punish = [0]*self.id_num,[0]*self.id_num,[0]*self.id_num
        #评价参数
        satisfaction, extra_heat, extra_elec, less_heat, less_elec = [0]*self.id_num,[0]*self.id_num,[0]*self.id_num,[0]*self.id_num,[0]*self.id_num
        battery_electricity, battery_punish = [0]*self.id_num,[0]*self.id_num
        watertank_heat, watertank_punish = [0]*self.id_num,[0]*self.id_num

        for key, value in actions.items():
            ag_id = int(key[-1])-1
            if key.find("battery") != -1:
                battery_electricity[ag_id], battery_charge_number[ag_id], battery_punish[ag_id], battery_sell_number[ag_id] = self.agents[ag_id*4].step(value)
            elif key.find("watertank") != -1:
                watertank_heat[ag_id], watertank_release[ag_id], watertank_punish[ag_id] = self.agents[ag_id*4+1].step(value)
            elif key.find("chp") != -1:
                chp_observation[ag_id], chp_gas_consumption[ag_id], chp_punish[ag_id] = self.agents[ag_id*4+2].step(value)
            elif key.find("boiler") != -1:
                boiler_heat_generate[ag_id], boiler_gas_consumption[ag_id], boiler_punish[ag_id] = self.agents[ag_id*4+3].step(value)
            else:
                raise Exception("请检查智能体名称！")
        total_electricity_generate = solar_generate_elec
        total_heat_generate = 0
        for i in range(self.id_num):
            total_electricity_generate += (chp_observation[i]["chp"+str(i+1)+"_electricity_generate"] - min(battery_charge_number[i], 0))
            total_heat_generate += (boiler_heat_generate[i]["boiler"+str(i+1)+"_heat_generate"] + watertank_release[i] + chp_observation[i]["chp"+str(i+1)+"_heat_generate"])
        for i in range(self.id_num):
            satisfaction[i], extra_heat[i], extra_elec[i], less_heat[i], less_elec[i] = self.users[i].judge_satisfaction(total_electricity_generate/self.id_num, total_heat_generate/self.id_num, self.satisfactory_factor)
        for i in range(self.id_num):
            battery_electricity[i], battery_punish[i] = self.agents[4*i].get_other_electricity(sum(extra_elec)/self.id_num)
            watertank_heat[i], watertank_punish[i] = self.agents[4*i+1].get_other_heat(sum(extra_heat)/self.id_num)
        # 总成本、惩罚与收益、满意度
        cost_all = 0
        punish = 0
        earnings = 0
        satisfaction_all = sum(satisfaction)
        for i in range(self.id_num):
            cost_all += self._cal_costs(max(0,battery_charge_number[i]) + less_elec[i], chp_gas_consumption[i] + boiler_gas_consumption[i] + less_heat[i])
            punish += self._cal_punish(battery_punish[i], watertank_punish[i], chp_punish[i], boiler_punish[i])
            earnings += self._cal_earning(battery_sell_number[i])
        
        #当前时刻前进到下一个时刻
        self.current_time_period += 1
        done = bool(
            self.current_time_period >= 24
            or punish < -5000 #done的新步骤
        )
        punish = max(punish, -200)  #惩罚限制
        # satisfaction = 20
        # #计算总体的reward
        reward = self.calculate_reward(cost_all, satisfaction_all, earnings, punish)
        next_price = self._get_current_price(self.current_time_period)
        next_demand = self._getdemand(self.current_time_period)
        self.observation = merge(next_price, next_demand)
        for i in range(self.id_num):
            self.observation = merge(self.observation, battery_electricity[i], watertank_heat[i], chp_observation[i], boiler_heat_generate[i])

        if done:
            self.not_done = False
        #数据的存储与显示
        render_list = {"solar_generate_elec":solar_generate_elec}     
        for i in range(self.id_num):
            render_list["battery"+str(i+1)+"_charge_number"] = battery_charge_number[i]
            render_list["battery"+str(i+1)+"_sell_number"]=battery_sell_number[i]
            render_list["watertank"+str(i+1)+"_release"]=watertank_release[i]
            render_list["boiler"+str(i+1)+"_gas_consumption"]=boiler_gas_consumption[i]
        render_list["buy_heat_additional"] = sum(less_heat)
        render_list["buy_elec_additional"] = sum(less_elec)
        cost_and_earning_dict = {"cost":cost_all, "earning":earnings, "elec_generate":total_electricity_generate, "heat_generate":total_heat_generate}
        self.save(merge(render_list, self.observation, cost_and_earning_dict))

        return self.observation, reward, done, {}
    
    # def _getdemand(self, ctime):
          #分离需求
    #     demand = {}
    #     for i in range(self.id_num):
    #         demand = merge(demand,self.users[i].generate_demand_fixed(ctime)) 
    #     return demand

    def _getdemand(self, ctime):
        #整合需求
        demand = {"heat_demand":0,"gas_demand":0,"electricity_demand":0}
        for i in range(self.id_num):
            cdemand = self.users[i].generate_demand_fixed(ctime)
            for key, value in cdemand.items():
                if key.find("heat") != -1:
                    demand["heat_demand"] += value
                elif key.find("gas") != -1:
                    demand["gas_demand"] += value
                elif key.find("electricity") != -1:
                    demand["electricity_demand"] += value
                else:
                    raise AssertionError("检查需求！")
        return demand

    def calculate_reward(self, cost, satisfactory, earnings, punish):
        return cost + satisfactory + earnings + punish
        # return cost

    def _get_current_price(self, ctime):
        self.current_electricity_price = self.electricity_price_all[ctime]
        self.current_gas_price = self.gas_price_all
        return {'current_electricity_price':self.current_electricity_price, 'current_gas_price':self.current_gas_price}

    def reset(self):
        #reset之后返回当前的状态
        self.current_time_period = 0
        self.not_done = True
        self.save_dict = {}
        for u in self.users:
            u.reset()

        #当前价格：2 需求：3
        self.observation = merge(self._get_current_price(self.current_time_period),self._getdemand(self.current_time_period))  
              
        for i in range(self.id_num):
            # 电池：1 热电联产：2 锅炉：1
            self.observation = merge(self.observation,self.agents[4*i].reset(),self.agents[4*i+1].reset(),self.agents[4*i+2].reset(),self.agents[4*i+3].reset())

        self.save(self.observation)
        return self.observation

    def render(self, render_th = None):
        if render_th == None:
            for key, value in self.observation.items():
                print(key[:]+":", value, end = ' ')
        elif isinstance(render_th,dict):
            for obs, value in render_th.items():
                print(obs+":", value, end = ' ')            
        print("Time:", self.current_time_period, end = ' ')
        print()


    def save(self, obs_dict):
        for key, value in obs_dict.items():
            if key in self.save_dict.keys():
                self.save_dict[key].append(round(value,2))
            else:
                self.save_dict[key] = [round(value,2)]

    def get_save(self, Force = False):
        if Force: return self.save_dict
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
    bat = Multiagent_energy(id_num=3)
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
        bat.render(render_th="time")
    bat.reset()
    print()
    bat.render()
    
    while True:
        bat.step(generate_action(bat.agents))
        bat.render(render_th = "time")
    