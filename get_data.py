from tensorboard.backend.event_processing import event_accumulator

log_data = event_accumulator.EventAccumulator(
    r'C:\Users\95397\Desktop\Github代码\Multiagent\models\MAAC_data\run26\logs\events.out.tfevents.1626829979.DESKTOP-VN1HGTT')

log_data.Reload()
# print(log_data.scalars.Keys())

MAAC_reward = log_data.scalars.Items('mean_episode_rewards') 
print(len(MAAC_reward))