import Agent
import Env
import DDPG_policy
from DDPG_policy import MADDPG
import time
import numpy as np
from logger import Mylogger
import defination

# EPISODES = 2000
EP_STEPS = 24
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
    ii = 400
    EPISODES = 3000
    max_reward = -10000
    env = Env.Multiagent_energy()
    agent_names = env.get_agent_names()
    Log = Mylogger("DDPG_data")
    best_actions = {}
    
    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    # a_bound = env.action_space.high
    # a_low_bound = env.action_space.low

    s_dims = env.observation_space
    a_dims = {key:value for key, value in env.action_space.items()}
    #观测空间维数的更改
    for key in s_dims.keys():
        s_dims[key] = len(defination.OBSERVATION)
    a_bounds = {key:value-1 for key, value in env.action_space.items()}
    a_low_bounds = {key:0  for key, value in env.action_space.items()}

    ddpg = MADDPG(a_dims, s_dims, agent_names)
    var = 3#3 # the controller of exploration which will decay during training process
    t1 = time.time()
    i = 0
    while i < EPISODES:
        i += 1
        s = env.reset()
        ep_r = 0
        if RENDER and i>ii:time.sleep(1)
        for j in range(EP_STEPS):
            # if RENDER and i>ii : env.render()
            # add explorative noise to action
            a = ddpg.choose_action(s)
            for key, value in a.items():
                action_space_list = np.array(range(env.action_space[key]))
                a[key] = normal_discrete(value, var, action_space_list, a_low_bounds[key], a_bounds[key])
            s_, r, done, info = env.step(a)
            ddpg.store_transition(s, a, r , s_) # store the transition to memory
            if ddpg.pointer > DDPG_policy.MEMORY_CAPACITY:
                var *= 0.9995 # decay the exploration controller factor
                ddpg.learn()
                
            s = s_
            ep_r += r
            if RENDER and i>ii : 
                env.render_custom(info)
            if j == EP_STEPS - 1:
                
                #if RENDER and i>350 : env.render()
                if EPISODES == 2000:EPISODES += i
                if ep_r > max_reward:
                    best_actions = env.get_save()
                    max_reward = ep_r
                    try:
                        total_cost = round(sum(best_actions["cost"]),2)
                        total_earning = round(sum(best_actions["earning"]),2)
                    except:
                        continue
                print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
                Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                #if ep_r > -300 : RENDER = True
                break
            if done:
                Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                print("errorEpisode: ", i, ' Reward: %i' % (ep_r))
                break
    print('Running time: ', time.time() - t1)
    print("best_reward:", max_reward)
    print("best_actions:")
    for items in best_actions.items():
        print(items)
    print("total_cost:",total_cost)
    print("total_earning:", total_earning)

if __name__=="__main__":
    main()