# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:16:16 2021

@author: Takumi
"""
import numpy as np
class reward_calc:
    # def __init__(self):
    #     self.reward = 0
        
    def calc(pos_p, pos_c):
        reward = pos_p - pos_c
        
        return reward
        
    
    def reward_ng(craft, ng_area_lat, ng_area_lon, ng_range):
        num = len(craft)
        reward_ng = np.zeros([num, 2])
        for i in range(num):
            if craft[i].pos[0] < (ng_area_lat[0] + ng_area_lat[1])/2:
                reward_ng[i,0] = (ng_area_lat[0]-craft[i].pos[0])/ng_range
    
            elif craft[i].pos[0] > ng_area_lat[1]:
                reward_ng[i,0] = (craft[i].pos[0]-ng_area_lat[1])/ng_range
            else:
                reward_ng[i,0] = -0.5
    
            if craft[i].pos[1] < ng_area_lon[0]:
                reward_ng[i,1] = (ng_area_lon[0]-craft[i].pos[1])/ng_range
    
            elif craft[i].pos[1] > ng_area_lon[1]:
                reward_ng[i,1] = (craft[i].pos[1]-ng_area_lon[1])/ng_range
    
            else:
                reward_ng[i,1] = -0.5
        # print(reward_ng)    
        return reward_ng