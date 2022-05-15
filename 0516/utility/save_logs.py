# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 17:59:51 2021

@author: OokiT1
"""
import numpy as np
import pickle
import os
from weapon.missile_3d import missile_3d
def save_logs_IMPALA(res_name,results,steps,CONTINUAL):
    # res_name = "lr1e-5"
    
    c = [list(map(lambda blue: results["policy_reward_max"][blue], ["blue_0","blue_1"])),
         list(map(lambda blue: results["policy_reward_mean"][blue], ["blue_0","blue_1"])),
         list(map(lambda blue: results["policy_reward_min"][blue], ["blue_0","blue_1"]))]
    b = list(map(lambda blue: np.array([c[0][blue],c[1][blue],c[2][blue]]), graph_list))
    

    a = np.hstack((np.array(list(map(lambda blue: results["info"]["learner"][blue]["learner_stats"]["vf_loss"], ["default_policy","default_policy"]))),
                   np.array([results["episode_reward_max"], results["episode_reward_mean"], results["episode_reward_min"]])))
    if steps > 0 or CONTINUAL:
        f = open(res_name+"episode_vf_loss"+".pkl", mode="rb")
        prev_a = pickle.load(f)
        f.close()
        a = np.vstack((prev_a,a))
        f = open(res_name+"policy"+".pkl", mode="rb")
        prev_b = pickle.load(f)
        f.close()
        b = list(map(lambda blue: np.vstack((prev_b[blue],b[blue])), graph_list))
        
    f = open(res_name+"episode_vf_loss"+".pkl",'wb')
    pickle.dump(a,f)
    f.close()
    f = open(res_name+"policy"+".pkl",'wb')
    pickle.dump(b,f)
    f.close()

def save_logs(res_name,results,steps,CONTINUAL,agent_num):
    # res_name = "lr1e-5"
    agent_list = [0]*agent_num
    graph_list = [0]*agent_num
    for i in range(agent_num): 
        agent_list[i] = "blue_"+str(i)
        graph_list[i] = i
    
    c = [list(map(lambda blue: results["policy_reward_max"][blue], agent_list)),
         list(map(lambda blue: results["policy_reward_mean"][blue], agent_list)),
         list(map(lambda blue: results["policy_reward_min"][blue], agent_list))]
    b = list(map(lambda blue: np.array([c[0][blue],c[1][blue],c[2][blue]]), graph_list))
    

    a = np.hstack((np.array(list(map(lambda blue: results["info"]["learner"][blue]["learner_stats"]["vf_loss"], agent_list))),
                   np.array([results["episode_reward_max"], results["episode_reward_mean"], results["episode_reward_min"]])))
    if steps > 0 or CONTINUAL:
        f = open(res_name+"episode_vf_loss"+".pkl", mode="rb")
        prev_a = pickle.load(f)
        f.close()
        a = np.vstack((prev_a,a))
        f = open(res_name+"policy"+".pkl", mode="rb")
        prev_b = pickle.load(f)
        f.close()
        b = list(map(lambda blue: np.vstack((prev_b[blue],b[blue])), graph_list))
        
    f = open(res_name+"episode_vf_loss"+".pkl",'wb')
    pickle.dump(a,f)
    f.close()
    f = open(res_name+"policy"+".pkl",'wb')
    pickle.dump(b,f)
    f.close()
def save_env_info(env):
    f = open("info"+".pkl",'wb')
    a = {"blue_radar_range":env.blue[0].radar_range,"blue_sensor_az":env.blue[0].sensor_az,"blue_mrm_range":env.blue[0].mrm_range,
         "red_radar_range":env.red[0].radar_range,"red_sensor_az":env.red[0].sensor_az,"red_mrm_range":env.red[0].mrm_range,
         "mrm_radar_range":missile_3d(env.blue[0]).radar_range,"mrm_sensor_az":missile_3d(env.blue[0]).sensor_az,
         "WINDOW_SIZE_lat":env.WINDOW_SIZE_lat,"WINDOW_SIZE_lon":env.WINDOW_SIZE_lon,"WINDOW_SIZE_alt":env.WINDOW_SIZE_alt}
    pickle.dump(a,f)
    f.close()
def save_hists(res_name,steps,hist,hist_dir):
    # res_name = "lr1e-5"
    f = open(hist_dir+str(steps)+"_"+res_name+"_hist"+".pkl",'wb')
    pickle.dump(hist,f)
    f.close()

    # c = [list(map(lambda blue: results["policy_reward_max"][blue], agent_list)),
    #      list(map(lambda blue: results["policy_reward_mean"][blue], agent_list)),
    #      list(map(lambda blue: results["policy_reward_min"][blue], agent_list))]
    # b = list(map(lambda blue: np.array([c[0][blue],c[1][blue],c[2][blue]]), graph_list))
    

    # a = np.hstack((np.array(list(map(lambda blue: results["info"]["learner"][blue]["learner_stats"]["vf_loss"], agent_list))),
    #                np.array([results["episode_reward_max"], results["episode_reward_mean"], results["episode_reward_min"]])))
    # if steps > 0 or CONTINUAL:
    #     f = open(res_name+"episode_vf_loss"+".pkl", mode="rb")
    #     prev_a = pickle.load(f)
    #     f.close()
    #     a = np.vstack((prev_a,a))
    #     f = open(res_name+"policy"+".pkl", mode="rb")
    #     prev_b = pickle.load(f)
    #     f.close()
    #     b = list(map(lambda blue: np.vstack((prev_b[blue],b[blue])), graph_list))
        
    # f = open(res_name+"episode_vf_loss"+".pkl",'wb')
    # pickle.dump(a,f)
    # f.close()
    # f = open(res_name+"policy"+".pkl",'wb')
    # pickle.dump(b,f)
    # f.close()