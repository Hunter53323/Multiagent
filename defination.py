import numpy as np
OBSERVATION = np.array(['battery_electricity', 
                'current_electricity_price', 
                'current_gas_price',
                'electricity_demand',
                'gas_demand',
                'heat_demand'])
AGENT = ['battery','user']
obs_battery = [0,1,3]
obs_other = []

OBSERVATION_BATTERY = OBSERVATION[obs_battery]
OBSERVATION_OTHER = []

def dict_to_list(dic):
    return np.array([value for value in dic.values()])
