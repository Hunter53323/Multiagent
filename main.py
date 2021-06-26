import Agent
import Env
import RLpolicy
from RLpolicy import MADDPG
import time
import numpy as np

EPISODES = 500
EP_STEPS = 23
RENDER = False

def normal_discrete(mean, var, action_space, low, high):
    """
    生成动作空间内离散的正态分布动作
    """
    mean = np.argmax(mean)
    continue_number = np.random.normal(mean, var)
    min_number_list = abs(action_space - continue_number)
    random_action = np.clip(np.argmin(min_number_list), low, high)
    action_list = np.zeros(len(action_space))
    action_list[random_action] = 1
    return action_list

def main():
    env = Env.Multiagent_energy(mode = "test")
    agent_names = env.get_agent_names()
    
    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    # a_bound = env.action_space.high
    # a_low_bound = env.action_space.low

    s_dims = env.observation_space
    a_dims = {key:value.n for key, value in env.action_space.items()}
    a_bounds = {key:value.n-1 for key, value in env.action_space.items()}
    a_low_bounds = {key:0  for key, value in env.action_space.items()}

    ddpg = MADDPG(a_dims, s_dims, agent_names)
    var = 10#3 # the controller of exploration which will decay during training process
    t1 = time.time()
    for i in range(EPISODES):
        s = env.reset()
        ep_r = 0
        for j in range(EP_STEPS):
            if RENDER and i>300 : env.render()
            # add explorative noise to action
            a = ddpg.choose_action(s)
            for key, value in a.items():
                action_space_list = np.array(range(env.action_space[key].n))
                a[key] = normal_discrete(value, var, action_space_list, a_low_bounds[key], a_bounds[key])
            s_, r, done, info = env.step(a)
            ddpg.store_transition(s, a, r , s_) # store the transition to memory
            if ddpg.pointer > RLpolicy.MEMORY_CAPACITY:
                var *= 0.9995 # decay the exploration controller factor
                ddpg.learn()
                
            s = s_
            ep_r += r
            if j == EP_STEPS - 1:
                print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
                #if ep_r > -300 : RENDER = True
                break
    print('Running time: ', time.time() - t1)
    #TODO: 每次24个时刻之后的总的reward作为总的reward，然后每个大epoch进行训练

if __name__=="__main__":
    main()