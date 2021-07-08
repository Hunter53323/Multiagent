from Env import Multiagent_energy

#封装原本的函数类变成标准的gym格式
class MAAC_wrapper(Multiagent_energy):
    def __init__(self):
        super().__init__()

    def step(self, actions):
        return super().step(actions)

    def reset(self):
        return super().reset()
