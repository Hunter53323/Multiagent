#电池的critic知道所有信息
#电池的actor知道电价、当前电池电量、传下来的Q值
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import defination

LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1) # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1) # initilizaiton of OUT
    def choose_action(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions = x * 2 # for the game "Pendulum-v0", action range is [-2, 2]
        return actions

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)
    def value_of_action(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x+y))
        return actions_value

    def shut_down_grad(self):
        self.requires_grad_ = False

class MADDPG(DDPG):
    def __init__(self, a_dims, s_dims, *agents):
        s_dim_all = len(defination.OBSERVATION)
        self.DDPGs = [DDPG(a_dims[agent], s_dims[agent], s_dim_all, agent) for agent in agents]

    def choose_action(self, obs):
        actions = []
        for agent in self.DDPGs:
            if agent.name == "battery":
                sub_obs = {key: value for key, value in obs.items() if key in defination.OBSERVATION_BATTERY}
            else:
                raise Exception("其他智能体的部分还没有完成！")
            actions.append(agent.choose_action(sub_obs))
        return actions

    def store_transition(self, s_critic, a, r, s__critic):
        return super().store_transition(s_critic, a, r, s__critic)

    def learn(self):
        return super().learn()

class DDPG(object):
    def __init__(self, a_dim, s_dim, s_dim_critic, name):
        self.a_dim, self.s_dim, self.s_dim_critic = a_dim, s_dim, s_dim_critic
        self.memory = np.zeros((MEMORY_CAPACITY, 2*s_dim + 2*s_dim_critic + a_dim + 1), dtype=np.float32)
        self.pointer = 0 # serves as updating the memory data 
        # Create the 4 network objects
        self.actor_eval = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic_eval = Critic(s_dim_critic, a_dim)
        self.critic_target = Critic(s_dim_critic, a_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()
        self.name = name

    def formatting(self, s_critic, s__critic):
        if self.name == "battery":
            s = {key: value for key, value in s_critic.items() if key in defination.OBSERVATION_BATTERY}
            s_ = {key: value for key, value in s__critic.items() if key in defination.OBSERVATION_BATTERY}
        else:
            raise Exception("请正确使用格式化函数！")

        return s, s_

    def store_transition(self, s_critic, a, r, s__critic): # how to store the episodic data to buffer
        """
        储存当前的observation
        """
        #提取对应智能体的数据
        s , s_ = self.formatting(s_critic, s__critic)
        #数据转化为列表
        s = defination.dict_to_list(s)
        s_ = defination.dict_to_list(s_)
        a = defination.dict_to_list(a)
        s_critic = defination.dict_to_list(s_critic)
        s__critic = defination.dict_to_list(s__critic)

        transition = np.hstack((s, s_critic, a, [r], s_, s__critic))
        index = self.pointer % MEMORY_CAPACITY # replace the old data with new data 
        self.memory[index, :] = transition
        self.pointer += 1
    
    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()
    
    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')           
        # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_s_critic = torch.FloatTensor(batch_trans[:, self.s_dim : self.s_dim + self.s_dim_critic])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim + self.s_dim_critic : self.s_dim + self.s_dim_critic + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, self.s_dim + self.s_dim_critic + self.a_dim : self.s_dim + self.s_dim_critic + self.a_dim + 1])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim_critic - self.s_dim:])
        batch_s__critic = torch.FloatTensor(batch_trans[:, -self.s_dim_critic:])
        
        # make action and evaluate its action values
        a = self.actor_eval.choose_action(batch_s)
        q = self.critic_eval.value_of_action(batch_s_critic, a)
        actor_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # compute the target Q value using the information of next state
        a_target = self.actor_target.choose_action(batch_s_)
        q_tmp = self.critic_target.value_of_action(batch_s__critic, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval.value_of_action(batch_s_critic, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()