# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:04:00 2021

@author: UAS
"""
import numpy as np

class obs_calc:
    def distances_calc(position):
        tmp_index = np.arange(position.shape[0])
        xx, yy = np.meshgrid(tmp_index, tmp_index)
        distances = np.linalg.norm(position[xx]-position[yy], axis=2)
        
        return distances
    
    def calc(blue, red, obs_shape):
        blue_num = len(blue)
        red_num = len(red)
        observation = np.zeros(obs_shape)
        obs_index_distance = (obs_shape[1]-(red_num+blue_num))
        
        for i in range(blue_num):
            observation[i][0:2] = blue[i].pos
            observation[i][2] = blue[i].hitpoint
            observation[i][3] = blue[i].mrm_num
            if blue[i].detect_launch_ML:
                observation[i][4] = 0
            else:
                observation[i][4] = 1
        
        for i in range(red_num):      
            observation[i+blue_num][0:2] = red[i].pos
            observation[i+blue_num][2] = red[i].hitpoint
            observation[i+blue_num][3] = red[i].mrm_num
            if red[i].detect_launch:
                observation[i+blue_num][4] = 0
            else:
                observation[i+blue_num][4] = 1        
        distances = obs_calc.distances_calc(observation[0:(blue_num+red_num),0:2]) 
        observation[0:blue_num+red_num,obs_index_distance:obs_index_distance+(red_num+blue_num)] = distances[0:(red_num+blue_num),0:(red_num+blue_num)]
        
        return observation
    
    
