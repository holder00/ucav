# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 20:36:30 2022

@author: Takumi
"""
import numpy as np

def distance_search(env,uav_id,faction):
    if faction == "blue":
        num = env.red_num
        uav = env.blue[uav_id]
        tgts = env.red
        # tgt_ids = [0]*num
    elif faction == "red":
        num = env.blue_num
        # tgt_ids = [0]*num
        uav = env.red[uav_id]
        tgts = env.blue
    temp_far_dist = np.zeros([num,2])
    for i in range(num):
        temp_far_dist[i,0] = np.linalg.norm(uav.pos - tgts[i].pos)
        temp_far_dist[i,1] = tgts[i].id
    far = temp_far_dist[np.argsort(temp_far_dist[:, 0])]
    
    return far