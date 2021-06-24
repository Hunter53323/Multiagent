import Agent
import Env
from RLpolicy import DDPG
import time
import numpy as np

EPISODES = 200
EP_STEPS = 200
MEMORY_CAPACITY = 10000
RENDER = False

def main():
    env = Env.Multiagent_energy(mode = "test")
    
    #下面的是复制的代码还没有进行修改
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    a_low_bound = env.action_space.low

    ddpg = DDPG(a_dim, s_dim, s_dim_critic)
    var = 3 # the controller of exploration which will decay during training process
    t1 = time.time()
    for i in range(EPISODES):
        s = env.reset()
        ep_r = 0
        for j in range(EP_STEPS):
            if RENDER: env.render()
            # add explorative noise to action
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
            s_, r, done, info = env.step(a)
            #store按照修改过的格式进行使用
            ddpg.store_transition(s, a, r / 10, s_) # store the transition to memory
            
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= 0.9995 # decay the exploration controller factor
                ddpg.learn()
                
            s = s_
            ep_r += r
            if j == EP_STEPS - 1:
                print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
                if ep_r > -300 : RENDER = True
                break
    print('Running time: ', time.time() - t1)

if __name__=="__main__":
    main()