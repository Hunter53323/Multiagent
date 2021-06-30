import numpy as np
OBSERVATION = np.array([ 
                'current_electricity_price', 
                'current_gas_price',
                'electricity_demand',
                'gas_demand',
                'heat_demand',
                'battery_electricity',
                'watertank_heat',
                "CHP_electricity_generate",
                "CHP_heat_generate",
                "boiler_heat_generate"
                ])
AGENT = ["Battery", "WaterTank", "CHP", "Boiler", "User"]
AGENT_NAME = ["battery", "watertank", "chp", "boiler"]

obs_battery = [0,2,5]
obs_watertank = [4,6]
obs_chp = [0,1,2,4]
obs_boiler = [1,4]

OBSERVATION_BATTERY = OBSERVATION[obs_battery]
OBSERVATION_WATERTANK = OBSERVATION[obs_watertank]
OBSERVATION_CHP = OBSERVATION[obs_chp]
OBSERVATION_BOILER = OBSERVATION[obs_boiler]

def dict_to_list(dic):
    return np.array([value for value in dic.values()])

def action_to_list(action, action_space_dim):
    action_list = np.zeros(action_space_dim)
    action_list[action] = 1
    return action_list

def list_to_action(action_list):
    return np.argmax(action_list)