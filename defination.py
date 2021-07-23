"""
DEFINATION用于定义一些全局都会使用的量和通用函数
"""
#########这里是需求和太阳能产生的原始数据########
# elected= [1.4809395,1.40578350000000,1.35973775000000,1.31863325000000,1.32567250000000,1.37803600000000,1.48022000000000,1.55149650000000,1.60550175000000,1.65035325000000,1.68689900000000,1.71790825000000,1.77225775000000,1.81427325000000,1.82744600000000,1.82345050000000,1.82934400000000,1.84328000000000,1.80750450000000,1.79104775000000,1.77386025000000,1.75332500000000,1.65654225000000,1.5194122500000]
# elected = [1.41792825000000,1.34702500000000,1.31376825000000,1.29036325000000,1.31315300000000,1.37028775000000,1.48184775000000,1.55617000000000,1.60467200000000,1.66998850000000,1.74866200000000,1.81629425000000,1.93536325000000,2.00868950000000,2.06325500000000,2.10604800000000,2.12270525000000,2.08174000000000,2.05837225000000,2.00233625000000,1.95302750000000,1.91342000000000,1.79306750000000,1.64606850000000]
# elected = [1.52943675000000,1.43300325000000,1.36004925000000,1.31362850000000,1.29685975000000,1.29985050000000,1.30856000000000,1.35249400000000,1.46927975000000,1.62967450000000,1.77425175000000,1.89722400000000,2.00294950000000,2.07976175000000,2.13364650000000,2.18338025000000,2.23142475000000,2.22922925000000,2.18698975000000,2.12457775000000,2.03875175000000,1.97277425000000,1.84567925000000,1.70479325000000]

# solar = [0,0,0,0,0,0.373426666666667,0.897366666666667,1.21317333333333,1.36592666666667,1.46178000000000,1.49558666666667,1.49840666666667,1.47934000000000,1.42158666666667,1.34944666666667,1.25656000000000,1.09164000000000,0.711500000000000,0.221080000000000,0,0,0,0,0]

# elec = [round(i,1) for i in solar]
# print(elec)
################################################
import numpy as np
OBSERVATION = np.array([ 
                'current_electricity_price', 
                'current_gas_price',
                'electricity_demand',
                'gas_demand',
                'heat_demand',
                'battery_electricity',
                'watertank_heat',
                "chp_electricity_generate",
                "chp_heat_generate",
                "boiler_heat_generate"
                ])

OBSERVATION = np.array([ 
                'current_electricity_price', 
                'current_gas_price',
                'electricity_demand',
                'gas_demand',
                'heat_demand',
                '_electricity',  #battery
                '_heat',   #watertank
                "_electricity_generate", #chp
                "_heat_generate", #chp
                "_heat_generate"  #boiler
                ])
AGENT = ["Battery", "WaterTank", "CHP", "Boiler", "User", "SolarPanel"]
# AGENT_NAME = ["battery1", "watertank1", "chp1", "boiler1","battery2", "watertank2", "chp2", "boiler2"]
AGENT_NAME = None
obs_battery = [0,2,5]
obs_watertank = [4,6]
obs_chp = [0,1,2,4]
obs_boiler = [1,4]

OBSERVATION_BATTERY = OBSERVATION[obs_battery]
OBSERVATION_WATERTANK = OBSERVATION[obs_watertank]
OBSERVATION_CHP = OBSERVATION[obs_chp]
OBSERVATION_BOILER = OBSERVATION[obs_boiler]

def OBSERVATION_BATTERY(name):
    result = OBSERVATION[obs_battery]
    result[-1] = name + result[-1]
    return result

def OBSERVATION_WATERTANK(name):
    result = OBSERVATION[obs_watertank]
    result[-1] = name + result[-1]
    return result

def OBSERVATION_CHP(name):
    return OBSERVATION[obs_chp]

def OBSERVATION_BOILER(name): 
    return OBSERVATION[obs_boiler]

def dict_to_list(dic):
    return np.array([value for value in dic.values()])

def action_to_list(action, action_space_dim):
    action_list = np.zeros(action_space_dim)
    action_list[action] = 1
    return action_list

def list_to_action(action_list):
    return np.argmax(action_list)