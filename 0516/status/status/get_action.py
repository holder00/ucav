# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:52:21 2021

@author: ookit1
"""
import copy

def get_action(env,action_dict):
    env_cp = copy.deepcoy()
    for i in range(env_cp.blue_num):
        if not env_cp.blue[i].hitpoint == 0:
            action_index = action_dict['blue_' + str(i)]
            env_cp.blue[i].tgt_update_ML(env_cp.red[action_index[0]])
            if action_index[1] == 0:
                env_cp.blue[i].detect_launch = True
            else:
                env_cp.blue[i].detect_launch = False
            if action_index[2] == 0:
                env_cp.blue[i].cool_down = 0
            else:
                env_cp.blue[i].cool_down = env_cp.blue[i].cool_down_limit   
                    
    return act