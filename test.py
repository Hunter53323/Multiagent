import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT
    def choose_action(self, x):
        #两种思路，一种是一个动作一维取最大值，另一种是一类动作一维映射成动作空间内的动作
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        _ , idx = x[0].max(0) #21个动作里面取最大的一个
        x[:, :] = 0
        x[:, idx] = 1
        print(x)
        return idx #type:tensor([k])

aa = Actor(5,5)
s = [1,2,3,4,5]
s = torch.unsqueeze(torch.FloatTensor(s), 0)
aa.choose_action(s)