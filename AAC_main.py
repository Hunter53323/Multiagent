import Agent
import Env
import AAC.buffer as buffer
from AAC.buffer import myBuffer
from AAC_policy import Actor_Attention_Critic
import time
import numpy as np
from logger import Mylogger

# EPISODES = 1000

#buffer储存的时候优先存储reward更高的动作，进而进行学习
EP_STEPS = 24
RENDER = False
BATCH_SIZE = 512#TODO:修正batch size
useGPU = True

buffer_length = int(1e6)

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
    Log = Mylogger("MAAC_data_no_attention")
    # Log = Mylogger("MAAC_data")
    env = Env.Multiagent_energy()
    model = Actor_Attention_Critic.init_from_env(env, q_lr=0.0001, critic_hidden_dim = 256, pol_hidden_dim= 256)
    replay_buffer = myBuffer(buffer_length, model.nagents,
                                 [obsp for obsp in env.observation_space.values()],
                                 [acsp for acsp in env.action_space.values()])

    a_bounds = {key:value-1 for key, value in env.action_space.items()}
    a_low_bounds = {key:0  for key, value in env.action_space.items()}

    t1 = time.time()
    print("start simulation!")
    EPISODES = 3000 #初始找到之后的循环次数
    max_reward = -100000
    best_actions = {}
    total_cost = 0
    total_earning = 0
    rewards = []
    var = 5
    i = 0
    while i < EPISODES:
        s = env.reset()
        ep_r = 0
        model.prep_rollouts(device='gpu')
        for j in range(EP_STEPS):
            
            a = model.step(s, to_gpu=useGPU)
            for key in a.keys():
                a[key] = a[key][0]
            for key, value in a.items():
                action_space_list = np.array(range(env.action_space[key]))
                a[key] = normal_discrete(value, var, action_space_list, a_low_bounds[key], a_bounds[key])
            s_, r, done, info = env.step(a)
            ep_r += r
            replay_buffer.push(s, a, r , s_, done) # store the transition to memory

            if replay_buffer.pointer > buffer.MEMORY_CAPACITY:
                var *= 0.9995
                # model.prep_training(device='gpu')
                sample = replay_buffer.sample(BATCH_SIZE, to_gpu=useGPU)
                model.learn(sample, logger=Log.logger, device = 'gpu')
                model.prep_rollouts(device='gpu')
            # ep_rews = replay_buffer.get_average_rewards(EP_STEPS)
            s = s_
            
            if j == EP_STEPS - 1:
                if (EPISODES == 3000) and (replay_buffer.pointer > buffer.MEMORY_CAPACITY):
                    EPISODES += i
                print('Episode: ', i, ' Reward: %i' % (ep_r))
                # if ep_r > 0:
                #     Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                if ep_r > max_reward:
                    best_actions = env.get_save()
                    max_reward = ep_r
                    try:
                        total_cost = round(sum(best_actions["cost"]),2)
                        total_earning = round(sum(best_actions["earning"]),2)
                    except:
                        continue
                break
            if done:
                # if ep_r > 0:
                #     Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                print("errorEpisode: ", i, ' Reward: %i' % (ep_r))
                break
        i += 1
    print('Running time: ', time.time() - t1)
    print("best_reward:", max_reward)
    print("best_actions:")
    for items in best_actions.items():
        print(items)
    print("total_cost:",total_cost)
    print("total_earning:", total_earning)
 
if __name__=="__main__":
    main()