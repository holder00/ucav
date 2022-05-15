# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:24:26 2022

@author: OokiT1
"""

class policies_control:
    def create(faction,uav_id,policy_config):
        policy = {faction+str(uav_id):policy_config} 
        return policy
        
    # def create_policies(faction,agent_num):
    #     for i in range(agent_num):
            
        
    def agent_list(faction,agent_num):
        agents = [0]*agent_num
        for i in range(agent_num):
            agents[i] = [faction+str(i)]
        return agents
