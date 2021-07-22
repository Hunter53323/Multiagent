import Agent
import Env
import MADDPG.buffer as buffer
from MADDPG.buffer import myBuffer
from MADDPG_policy import MADDPG
import time
import numpy as np
from logger import Mylogger
import matplotlib as plt

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
    Log = Mylogger("MADDPG_data")
    env = Env.Multiagent_energy()
    model = MADDPG.init_from_env(env, lr=0.001, hidden_dim=256)
    replay_buffer = myBuffer(buffer_length, model.nagents,
                                 [obsp for obsp in env.observation_space.values()],
                                 [acsp for acsp in env.action_space.values()])

    a_bounds = {key:value-1 for key, value in env.action_space.items()}
    a_low_bounds = {key:0  for key, value in env.action_space.items()}
    a_loss = []
    c_loss = []

    t1 = time.time()
    print("start simulation!")
    EPISODES = 10000 #初始找到之后的循环次数
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

        explr_pct_remaining = max(0, 25000 - i) / 25000
        model.scale_noise(0.0 + (0.3 - 0.0) * explr_pct_remaining)
        model.reset_noise()

        for j in range(EP_STEPS):
            a = model.step(s, to_gpu=useGPU, explore=True)
            for key in a.keys():
                a[key] = a[key][0]
            # for key, value in a.items():
            #     action_space_list = np.array(range(env.action_space[key]))
            #     a[key] = normal_discrete(value, var, action_space_list, a_low_bounds[key], a_bounds[key])
            s_, r, done, info = env.step(a)
            replay_buffer.push(s, a, r , s_, done) # store the transition to memory

            if replay_buffer.pointer > buffer.MEMORY_CAPACITY:
                var *= 0.9995
                model.prep_training(device='gpu')
                for a_i in range(model.nagents):
                    sample = replay_buffer.sample(BATCH_SIZE, to_gpu=useGPU)
                    model.update(sample, a_i, logger=Log.logger, actor_loss_list=a_loss, critic_loss_list=c_loss)
                model.update_all_targets()
                model.prep_rollouts(device='gpu')

            s = s_
            ep_r += r
            if j == EP_STEPS - 1:
                # if (EPISODES == 3000) and (replay_buffer.pointer > buffer.MEMORY_CAPACITY):EPISODES += i
                if (EPISODES == 10000) and (replay_buffer.pointer > buffer.MEMORY_CAPACITY):EPISODES += i
                print('Episode: ', i, ' Reward: %i' % (ep_r))
                if ep_r > 0:
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
                if ep_r > 0:
                    Log.logger.add_scalar("mean_episode_rewards", ep_r, i)
                print("errorEpisode: ", i, ' Reward: %i' % (ep_r))
                if ep_r > max_reward:
                    best_actions = env.get_save()
                    max_reward = ep_r
                    try:
                        total_cost = round(sum(best_actions["cost"]),2)
                        total_earning = round(sum(best_actions["earning"]),2)
                    except:
                        continue
                break
        i += 1
    print('Running time: ', time.time() - t1)
    print("best_reward:", max_reward)
    print("best_actions:")
    for items in best_actions.items():
        print(items)
    print("total_cost:",total_cost)
    print("total_earning:", total_earning)
    index_aloss = list(range(1, len(a_loss) + 1))

    plt.plot(index_aloss, a_loss)
    plt.ylabel('actor_loss')
    plt.show()

if __name__=="__main__":
    main()