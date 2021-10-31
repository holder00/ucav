# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:27:31 2021

@author: Takumi
"""

import numpy as np

def get_state(env,uav,position,distances):
    uav_state = np.zeros(env.observation_space.shape[1])
    uav_state[0] = uav.hitpoint        
    uav_state[1] = uav.mrm_num
    if uav.inrange:
        uav_state[2] = 0
    else:
        uav_state[2] = 1
    if uav.detect_launch_ML:
        uav_state[3] = 0
    else:
        uav_state[3] = 1
    uav_state[4] = np.cos(uav.ops_az())
    uav_state[5] = np.sin(uav.ops_az())
    uav_state[6:9] = position/(np.array([env.WINDOW_SIZE_lat,env.WINDOW_SIZE_lon,env.WINDOW_SIZE_alt]))   
    
    if uav.faction == "blue" and not uav.hitpoint == 0:
        uav_state[9:9+env.action_space.shape[0]] = env.action_dict_c['blue_' + str(uav.id)]
    else:
        uav_state[9:9+env.action_space.shape[0]] = -np.ones(env.action_space.shape[0])
        
    uav_state[9+env.action_space.shape[0]:] = distances/(env.WINDOW_SIZE_lat*3)

    return uav_state

def get_obs(env):
    obs = {}
    observation = np.zeros(env.observation_space.shape)
    position = np.zeros([env.blue_num+env.red_num,3])
    for i in range(env.blue_num):
        position[i] = env.blue[i].pos
        
    for i in range(env.red_num):
        position[i+env.blue_num] = env.red[i].pos
    distances = env.distances_calc(position)
    
    for i in range(env.blue_num):
        # 状態の作成  
        if not env.blue[i].hitpoint == 0:            
            observation[i,:] = get_state(env,env.blue[i],position[i],distances[i])

  
    for i in range(env.red_num):
        # 状態の作成
        if not env.red[i].hitpoint == 0:   
            observation[i+env.blue_num,:] = get_state(env,env.red[i],position[i+env.blue_num],distances[i+env.blue_num])
    
    for i in range(env.blue_num):
        obs['blue_' + str(i)] = np.vstack([observation[i,:],np.delete(observation,i,0)]).astype(np.float32)
        
    return obs