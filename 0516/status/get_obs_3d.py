# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:27:31 2021

@author: Takumi
"""

import numpy as np
import copy
from utility.harf_angle import harf_angle
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_phi
from status.distance_search import distance_search


def get_obs_self_play(env):
    obs_b = get_obs(env,"blue")
    obs_r = get_obs(env,"red")
    
    return {**obs_b, **obs_r}

def get_owns_state(env,self_uav,uav,uav_state):
    harf = lambda val:2*val-1
    uav_state["mrm_num"] = uav.mrm_num
    if uav.inrange:
        uav_state["inrange"] = 1
    else:
        uav_state["inrange"] = 2
    if uav.detect_launch:
        uav_state["detect"] = 1
    else:
        uav_state["detect"] = 2
    
    if self_uav.id == uav.id:
        uav_state["vector_psi_x"] = np.array([np.cos(uav.psi)])
        uav_state["vector_psi_y"] = np.array([np.sin(uav.psi)])
        uav_state["vector_gam_x"] = np.array([np.cos(uav.gam)])
        uav_state["vector_gam_y"] = np.array([np.sin(uav.gam)])
    else:
        aa_psi = calc_each_aspect_psi(self_uav,uav)
        aa_gam = calc_each_aspect_gam(self_uav,uav)
        
        uav_state["vector_psi_x"] = np.array([np.cos(aa_psi)])
        uav_state["vector_psi_y"] = np.array([np.sin(aa_psi)])
        uav_state["vector_gam_x"] = np.array([np.cos(aa_gam)])
        uav_state["vector_gam_y"] = np.array([np.sin(aa_gam)])
        
    uav_state["velocity"] = np.array([harf(uav.V/340.0/1.5)])
    close_id = distance_search(env,self_uav.id,self_uav.faction)
    tgt_id = np.where(close_id == uav.tgt.id)
    
    uav_state["tgt_id"] = int(tgt_id[0])+1 #Close = 0
    # uav_state["tgt_id"] = uav.tgt.id + 1
    
    return uav_state

def get_owns_dead_state(uav,uav_state):
    harf = lambda val:2*val-1
    uav_state["mrm_num"] = 0

    uav_state["inrange"] = 0
    uav_state["detect"] = 0
        
    uav_state["vector_psi_x"] = np.array([harf(0)])
    uav_state["vector_psi_y"] = np.array([harf(0)])
    uav_state["vector_gam_x"] = np.array([harf(0)])
    uav_state["vector_gam_y"] = np.array([harf(0)])
    uav_state["velocity"] = np.array([harf(0)])
    uav_state["tgt_id"] = 0
    
    return uav_state

def get_state_blue(env,uav):
    harf = lambda val:2*val-1
    lat_norm = lambda val:harf((val+1)/3)
    uav_state = {}
    uav_state["hitpoint"] = np.array([harf(uav.hitpoint)])
    if not uav.hitpoint == 0:
        uav_state = get_owns_state(env,uav,uav,uav_state)
    else:
        uav_state = get_owns_dead_state(uav,uav_state)
        

    return uav_state

def get_state_f_blue(env,uav,self_uav):
    harf = lambda val:2*val-1
    dist_norm = lambda val:val/(3*env.WINDOW_SIZE_lat)
    uav_state = {}
    uav_state["hitpoint"] = np.array([harf(uav.hitpoint)])

    if not uav.hitpoint == 0:
        uav_state = get_owns_state(env,self_uav,uav,uav_state)
        
        uav_state["distances"] = np.array([harf(dist_norm(np.linalg.norm(uav.pos - self_uav.pos)))])
        op_psi = calc_each_ops_psi(self_uav,uav)
        op_gam = calc_each_ops_gam(self_uav,uav)
        uav_state["psi_x"] = np.array([np.cos(op_psi)])
        uav_state["psi_y"] = np.array([np.sin(op_psi)])
        uav_state["gam_x"] = np.array([np.cos(op_gam)])
        uav_state["gam_y"] = np.array([np.sin(op_gam)])
    else:
        uav_state = get_owns_dead_state(uav,uav_state)
        
        uav_state["distances"] = np.array([harf(0)])
        uav_state["psi_x"] = np.array([harf(0)])
        uav_state["psi_y"] = np.array([harf(0)])
        uav_state["gam_x"] = np.array([harf(0)])
        uav_state["gam_y"] = np.array([harf(0)])


    return uav_state

def get_state_red(env,uav,self_uav):
    harf = lambda val:2*val-1
    dist_norm = lambda val:val/(3*env.WINDOW_SIZE_lat)
    uav_state = {}
    uav_state["hitpoint"] = np.array([harf(uav.hitpoint)])
    if not uav.hitpoint == 0:
        uav_state["tgt_id"] = uav.id+1
        uav_state["distances"] = np.array([harf(dist_norm(np.linalg.norm(uav.pos - self_uav.pos)))])
        op_psi = calc_each_ops_psi(self_uav,uav)
        op_gam = calc_each_ops_gam(self_uav,uav)
        uav_state["psi_x"] = np.array([np.cos(op_psi)])
        uav_state["psi_y"] = np.array([np.sin(op_psi)])
        uav_state["gam_x"] = np.array([np.cos(op_gam)])
        uav_state["gam_y"] = np.array([np.sin(op_gam)])
        
        aa_psi = calc_each_aspect_psi(self_uav,uav)
        aa_gam = calc_each_aspect_gam(self_uav,uav)
        
        uav_state["vector_psi_x"] = np.array([np.cos(aa_psi)])
        uav_state["vector_psi_y"] = np.array([np.sin(aa_psi)])
        uav_state["vector_gam_x"] = np.array([np.cos(aa_gam)])
        uav_state["vector_gam_y"] = np.array([np.sin(aa_gam)])
        uav_state["velocity"] = np.array([harf(uav.V/340.0/1.5)])
    else:
        uav_state["tgt_id"] = 0
        uav_state["distances"] = np.array([harf(0)])

        uav_state["psi_x"] = np.array([harf(0)])
        uav_state["psi_y"] = np.array([harf(0)])
        uav_state["gam_x"] = np.array([harf(0)])
        uav_state["gam_y"] = np.array([harf(0)])
        
        uav_state["vector_psi_x"] = np.array([harf(0)])
        uav_state["vector_psi_y"] = np.array([harf(0)])
        uav_state["vector_gam_x"] = np.array([harf(0)])
        uav_state["vector_gam_y"] = np.array([harf(0)])
        uav_state["velocity"] = np.array([harf(0)])

    return uav_state

def get_state_mrm(env,mrm,self_uav,is_launched):
    harf = lambda val:2*val-1
    dist_norm = lambda val:val/(3*env.WINDOW_SIZE_lat)
    uav_state = {}
    if is_launched and not mrm.hitpoint == 0:
        uav_state["status"] = 1
        uav_state["hitpoint"] = np.array([harf(mrm.hitpoint/mrm.hitpoint_ini)])
        uav_state["parent_id"] = mrm.parent.id+1
        if mrm.inrange:
            uav_state["inrange"] = 1
        else:
            uav_state["inrange"] = 2
        uav_state["tgt_id"] = mrm.tgt.id+1
        temp_mrm = limit_calc(mrm.pos, [env.WINDOW_SIZE_lat, env.WINDOW_SIZE_lon, env.WINDOW_SIZE_alt])
        
        uav_state["distances"] = np.array([harf(dist_norm(np.linalg.norm(temp_mrm - self_uav.pos)))])
        op_psi = calc_each_ops_psi_pos(self_uav,temp_mrm)
        op_gam = calc_each_ops_gam_pos(self_uav,temp_mrm)
        uav_state["psi_x"] = np.array([np.cos(op_psi)])
        uav_state["psi_y"] = np.array([np.sin(op_psi)])
        uav_state["gam_x"] = np.array([np.cos(op_gam)])
        uav_state["gam_y"] = np.array([np.sin(op_gam)])
        
        aa_psi = calc_each_aspect_psi(self_uav,mrm)
        aa_gam = calc_each_aspect_gam(self_uav,mrm)
        
        uav_state["vector_psi_x"] = np.array([np.cos(aa_psi)])
        uav_state["vector_psi_y"] = np.array([np.sin(aa_psi)])
        uav_state["vector_gam_x"] = np.array([np.cos(aa_gam)])
        uav_state["vector_gam_y"] = np.array([np.sin(aa_gam)])
        uav_state["velocity"] = np.array([harf(mrm.V/340.0/3)])
        # uav_state["velocity"] = np.array([harf(mrm.V/340.0/4)])
    else:
        uav_state["status"] = 0
        uav_state["hitpoint"] = np.array([harf(0.0)])
        uav_state["parent_id"] = 0
        uav_state["inrange"] = 0
        uav_state["tgt_id"] = 0
        
        uav_state["distances"] = np.array([harf(0)])

        uav_state["psi_x"] = np.array([harf(0)])
        uav_state["psi_y"] = np.array([harf(0)])
        uav_state["gam_x"] = np.array([harf(0)])
        uav_state["gam_y"] = np.array([harf(0)])
        
        uav_state["vector_psi_x"] = np.array([harf(0)])
        uav_state["vector_psi_y"] = np.array([harf(0)])
        uav_state["vector_gam_x"] = np.array([harf(0)])
        uav_state["vector_gam_y"] = np.array([harf(0)])
        uav_state["velocity"] = np.array([harf(0)])
    
    # uav_state["distances"] = np.array([harf(mrm_distance/(env.WINDOW_SIZE_lat*3))])
    return uav_state


def get_obs(env,own_side):
    if own_side == "blue":
        owns_uav = env.blue
        owns_num = env.blue_num
        enem_uav = env.red
        enem_num = env.red_num
        key_label = "blue_"
    elif own_side == "red":
        owns_uav = env.red
        owns_num = env.red_num
        enem_uav = env.blue
        enem_num = env.blue_num
        key_label = "red_"
    obs = {}
    for i in range(owns_num):
        obs[key_label + str(i)] = {}
        obs[key_label + str(i)]["owns"] = {}
        obs[key_label + str(i)]["enems"] = {}
        obs[key_label + str(i)]["mrms"] = {}
    observation = {}
    observation["owns"] = {}
    observation["enems"] = {}
    # position = np.zeros([owns_num+enem_num,3])
    # for i in range(owns_num):
        # position[i] = owns_uav[i].pos

    # for i in range(enem_num):
        # position[i+owns_num] = enem_uav[i].pos
    # distances = distances_calc(position)
    
    # nearest_dist = np.zeros([owns_num,enem_num])
    # for i in range(owns_num):
        # nearest_dist[i,:] = np.sort(distances[i,owns_num:])
    
    # nearest_id = find_nearest_craft(env,owns_num,owns_uav,enem_uav)

    for i in range(owns_num):#どの機体の観測量を作るのかのループ
        uav_id = 0
        self_uav = owns_uav[i]
        close_id = distance_search(env,self_uav.id,self_uav.faction)
        
        for j in range(owns_num):#僚機のうちどの機体の観測量を取得するのかを決めるループ
            if i == j:
                obs[key_label + str(i)]["owns"]["self"] = get_state_blue(env,self_uav)
            else:
                obs[key_label + str(i)]["owns"]["friend_"+str(uav_id)] = get_state_f_blue(env,owns_uav[j],self_uav)
                uav_id = uav_id +1
        for j in range(enem_num):#相手機のうちどの機体の観測量を取得するのかを決めるループ
            ids = int(close_id[j,1])
            obs[key_label + str(i)]["enems"]["red_" + str(ids)] = get_state_red(env,enem_uav[ids],self_uav)
        # if nearest_id[i] == 0:
        #     ids = 0
        #     obs[key_label + str(i)]["enems"]["red_" + str(0)] = get_state_red(env,enem_uav[ids],self_uav)

        #     ids = 1
        #     obs[key_label + str(i)]["enems"]["red_" + str(1)] = get_state_red(env,enem_uav[ids],self_uav)
        # else:
        #     ids = 1
        #     obs[key_label + str(i)]["enems"]["red_" + str(0)] = get_state_red(env,enem_uav[ids],self_uav)

        #     ids = 0
        #     obs[key_label + str(i)]["enems"]["red_" + str(1)] = get_state_red(env,enem_uav[ids],self_uav)
            
        for j in range(owns_num*2):
            obs[key_label + str(i)]["mrms"]["own_mrm_" + str(j)] = get_state_mrm(env,env.mrm[j],self_uav,is_launched=False)
        mrm_id = 0
        for j in range(env.mrm_num):
            if env.mrm[j].parent.faction == "blue":
                is_launched = True
                obs[key_label + str(i)]["mrms"]["own_mrm_" + str(mrm_id)] = get_state_mrm(env,env.mrm[j],self_uav,is_launched)     
                mrm_id += 1
        # obs["blue_" + str(i)] = np.array([env.timer])
    return obs

def calc_each_ops_psi(blue,red):
    tgt_pos = red.pos - blue.pos 
    tgt_az = np.arctan2(tgt_pos[1],tgt_pos[0])-blue.psi
    tgt_az = harf_angle(tgt_az)
    return tgt_az
    
def calc_each_ops_gam(blue,red):
    tgt_pos = red.pos - blue.pos
    tgt_az = np.arctan2(tgt_pos[1],tgt_pos[0])
    rot_psi = get_rotation_matrix_3d_psi(-tgt_az)
    tgt_vec = np.dot(rot_psi,tgt_pos)
    tgt_gam = np.arctan2(tgt_vec[2],np.abs(tgt_vec[0]))-blue.gam
    tgt_gam = harf_angle(tgt_gam)
    return tgt_gam

def calc_each_ops_psi_pos(blue,pos):
    tgt_pos = pos - blue.pos 
    tgt_az = np.arctan2(tgt_pos[1],tgt_pos[0])-blue.psi
    tgt_az = harf_angle(tgt_az)
    return tgt_az
    
def calc_each_ops_gam_pos(blue,pos):
    tgt_pos = pos - blue.pos
    tgt_az = np.arctan2(tgt_pos[1],tgt_pos[0])
    rot_psi = get_rotation_matrix_3d_psi(-tgt_az)
    tgt_vec = np.dot(rot_psi,tgt_pos)
    tgt_gam = np.arctan2(tgt_vec[2],np.abs(tgt_vec[0]))-blue.gam
    tgt_gam = harf_angle(tgt_gam)
    return tgt_gam

def calc_each_aspect_psi(blue,red):
    temp_psi = red.psi - blue.psi
    return harf_angle(temp_psi)

def calc_each_aspect_gam(blue,red):
    temp_gam = red.gam - blue.gam
    return harf_angle(temp_gam)
    

# def find_nearest_craft(env,owns_num,owns_uav,enem_uav):
#     #2機まで対応
#     nearest_id = [0]*owns_num
#     nearest_dist = [0]*owns_num
#     for i in range(owns_num):
#         for j in range(owns_num):
#             temp_dist = np.linalg.norm(owns_uav[i].pos - enem_uav[j].pos)
#             if j == 0:
#                 nearest_dist[i] = temp_dist
#             elif nearest_dist[i] > temp_dist:
#                 nearest_id[i] = 1
#                 nearest_dist[i] = temp_dist

#     return nearest_id

# def distances_calc(position):
#     tmp_index = np.arange(position.shape[0])
#     xx, yy = np.meshgrid(tmp_index, tmp_index)
#     distances = np.linalg.norm(position[xx]-position[yy], axis=2)


#     return distances

def limit_calc(pos_c, limit):
    pos = pos_c
    for i in range(len(limit)):
        if pos_c[i] <= 0:
            pos[i] = 0

        if pos_c[i] >= limit[i]:
            pos[i] = limit[i]

    return pos