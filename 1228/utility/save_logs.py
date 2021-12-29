# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 17:59:51 2021

@author: OokiT1
"""
import numpy as np
import pickle
import os

def save_logs(res_name,results,steps,CONTINUAL):
    # res_name = "lr1e-5"
    
    c = [list(map(lambda blue: results["policy_reward_max"][blue], ["blue_0","blue_1"])),
         list(map(lambda blue: results["policy_reward_mean"][blue], ["blue_0","blue_1"])),
         list(map(lambda blue: results["policy_reward_min"][blue], ["blue_0","blue_1"]))]
    b = list(map(lambda blue: np.array([c[0][blue],c[1][blue],c[2][blue]]), [0,1]))
    

    a = np.hstack((np.array(list(map(lambda blue: results["info"]["learner"][blue]["learner_stats"]["vf_loss"], ["blue_0","blue_1"]))),
                   np.array([results["episode_reward_max"], results["episode_reward_mean"], results["episode_reward_min"]])))
    if steps > 0 or CONTINUAL:
        f = open(res_name+"episode_vf_loss"+".pkl", mode="rb")
        prev_a = pickle.load(f)
        f.close()
        a = np.vstack((prev_a,a))
        f = open(res_name+"policy"+".pkl", mode="rb")
        prev_b = pickle.load(f)
        f.close()
        b = list(map(lambda blue: np.vstack((prev_b[blue],b[blue])), [0,1]))
        
    f = open(res_name+"episode_vf_loss"+".pkl",'wb')
    pickle.dump(a,f)
    f.close()
    f = open(res_name+"policy"+".pkl",'wb')
    pickle.dump(b,f)
    f.close()