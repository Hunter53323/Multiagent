import Agent
import Env
import AAC.buffer as buffer
from AAC.buffer import myBuffer
from RLpolicy import Actor_Attention_Critic
import time
import numpy as np

EPISODES = 1000
EP_STEPS = 23
RENDER = False
BATCH_SIZE = 32

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
    ii = 400
    env = Env.Multiagent_energy()

    # s_dims = env.observation_space
    # a_dims = {key:value for key, value in env.action_space.items()}
    # a_bounds = {key:value-1 for key, value in env.action_space.items()}
    # a_low_bounds = {key:0  for key, value in env.action_space.items()}

    model = Actor_Attention_Critic.init_from_env(env)
    replay_buffer = myBuffer(buffer_length, model.nagents,
                                 [obsp for obsp in env.observation_space.values()],
                                 [acsp for acsp in env.action_space.values()])
    var = 3#3 # the controller of exploration which will decay during training process
    t1 = time.time()
    for i in range(EPISODES):
        s = env.reset()
        ep_r = 0
        if RENDER and i>ii:time.sleep(1)
        for j in range(EP_STEPS):
            if RENDER and i>ii : env.render()
            
            a = model.step(s)
            # for key, value in a.items():
            #     action_space_list = np.array(range(env.action_space[key]))
            #     a[key] = normal_discrete(value, var, action_space_list, a_low_bounds[key], a_bounds[key])
            for key in a.keys():
                a[key] = a[key][0]
            s_, r, done, info = env.step(a)
            replay_buffer.push(s, a, r , s_, done) # store the transition to memory

            if replay_buffer.pointer > buffer.MEMORY_CAPACITY:
                # var *= 0.9995 # decay the exploration controller factor
                sample = replay_buffer.sample(BATCH_SIZE)
                # model.prep_training(device='gpu')
                model.learn(sample)
                # model.prep_rollouts(device='cpu')
            # ep_rews = replay_buffer.get_average_rewards(EP_STEPS)

            s = s_
            ep_r += r
            if RENDER and i>ii : 
                env.render_custom(info)
                
            if j == EP_STEPS - 1:
                #if RENDER and i>350 : env.render()
                print('Episode: ', i, ' Reward: %i' % (ep_r))# , 'Explore: %.2f' % var)
                #if ep_r > -300 : RENDER = True
                break
    print('Running time: ', time.time() - t1)

if __name__=="__main__":
    main()