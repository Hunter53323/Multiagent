from gym import spaces
class BaseAgent():
    def __init__(self, mode = "Mix"):
        if mode == "Mix":
            self.electricity = 0
            self.gas = 0
        elif mode == "Elec":
            self.electricity = 0
        elif mode == "Gas":
            self.gas = 0

        
class Battery(BaseAgent):
    def __init__(self):
        super().__init__(mode="Elec")
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

        change_number = num_mapping(action)
        self.electricity += change_number
        #电量约束
        if (self.electricity > self.max_electricity) \
            or (self.electricity < self.min_electricity):
            reward = -10
        else:
            reward = 0

        return self.electricity, reward

    def reset(self):
        self.electricity = 0
        return self.electricity

    def render(self):
        #环境的render模块里面调用输出相应的数据
        pass


def num_mapping(number):
    """
    将离散的actoin space转换成具体的充放数据
    """
    return (number - 10)/10.0

if __name__ == "__main__":
    import random
    battery = Battery()
    action = random.randrange(21)
    elec, reward = battery.step(action)
    print("动作：",action)
    print("电池电量：", elec)
    print("奖励：",reward)