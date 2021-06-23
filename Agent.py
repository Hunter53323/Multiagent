from gym import spaces
import random
class BaseAgent():
    def __init__(self, mode = "Mix"):
        if mode == "Mix":
            self.electricity = 0
            self.gas = 0
        elif mode == "Elec":
            self.electricity = 0
        elif mode == "Gas":
            self.gas = 0

        
class Battery(BaseAgent):# TODO: 输出返回充放电多少，之后评价满意度
    def __init__(self):
        super().__init__(mode="Elec")
        self.electricity = 1
        #动作空间从放电百分之十到充电百分之十
        self.action_space = spaces.Discrete(21)
        #当前电量状态为满电的百分比
        self.observation_space = spaces.Discrete(101)

        #常数定义
        self.max_electricity = 10
        self.min_electricity = 0
        self.charge_discharge_max = 10
        self.eta = 0.98

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        charge_number = num_mapping(action)
        self.electricity += charge_number
        #电量约束,超出约束则给予惩罚
        if (self.electricity > self.max_electricity) \
            or (self.electricity < self.min_electricity):
            reward = -10
        else:
            reward = 0

        return self.electricity, self.charge_number, reward

    def reset(self):
        self.electricity = 1

        battery_electricity = {'battery_electricity':self.electricity}
        return battery_electricity
        #return self.electricity

    def render(self):
        #环境的render模块里面调用输出相应的数据
        pass

class User():
    def __init__(self, mode = "test"):
        self.satisfaction = None
        self.electricity_demand = None
        self.gas_demand = None
        self.heat_demand = None
        
        self.electricity_demand_max = 0.5
        self.gas_demand_max = 0.5
        self.heat_demand_max = 0.5

        #确保需求和评判按照顺序执行
        self.process = 0

        #运行模式是否为测试模式
        self.run_mode = mode
        
    def generate_demand(self):
        assert self.process == 0, "请先将上一步的生成需求进行满意度评判"
        self.process = 1

        self.electricity_demand = random_demand(self.electricity_demand_max)
        self.gas_demand = random_demand(self.gas_demand_max)
        self.heat_demand = random_demand(self.heat_demand_max)
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

    def judge_satisfaction(self, electricity_provide, gas_provide):
        assert self.process == 1, "判断前需先生成需求"
        self.process = 0

        gas_satisfaction = abs(gas_provide - self.gas_demand)
        elec_satisfaction = abs(electricity_provide - self.electricity_demand)
        #满意度的初始值和test模式中智能体的个数有关
        satisfaction = 0.5 - gas_satisfaction - elec_satisfaction
        return satisfaction

    def reset(self):
        self.process = 0
        return self.generate_demand()

def num_mapping(number):
    """
    将离散的actoin space转换成具体的充放数据
    """
    return (number - 10)/10.0

def random_demand(sup):
    return random.randrange(int(10*sup))/10.0

if __name__ == "__main__":
    battery = Battery()
    action = random.randrange(21)
    elec, reward = battery.step(action)
    print("动作：",action)
    print("电池电量：", elec)
    print("奖励：",reward)