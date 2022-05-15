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
# def get_state(env,uav,position,distances):
#     # uav_state = np.zeros(env.observation_space.shape[1])
#     uav_state = {}
#     # if uav.faction == "blue":
#     #     uav_state = env.obs_dict.sample()
#     # elif uav.faction == "red":
#     #     uav_state = env.obs_dict_red.sample()
#     uav_state["hitpoint"] = np.array([uav.hitpoint])

#     uav_state["mrm_num"] = uav.mrm_num
#     if uav.inrange:
#         uav_state["inrange"] = np.array([0])
#     else:
#         uav_state["inrange"] = np.array([1])
#     if uav.detect_launch_ML:
#         uav_state["detect"] = np.array([0])
#     else:
#         uav_state["detect"] = np.array([1])
#     aa = -(uav.psi + uav.ops_az())
#     if aa > np.pi:
#         aa = aa - 2*np.pi
#     if aa < -np.pi:
#         aa = aa + 2*np.pi
#     uav_state["tgt_psi_x"] = np.array([np.cos(aa)])
#     uav_state["tgt_psi_y"] = np.array([np.sin(aa)])
#     aa = -(uav.gam + uav.ops_gam())
#     if aa > np.pi:
#         aa = aa - 2*np.pi
#     if aa < -np.pi:
#         aa = aa + 2*np.pi
#     aa = -aa
#     uav_state["tgt_gam_x"] = np.array([np.cos(aa)])
#     uav_state["tgt_gam_y"] = np.array([np.sin(aa)])    

#     # uav_state["tgt_psi_x"] = np.array([np.cos(uav.ops_az())])
#     # uav_state["tgt_psi_y"] = np.array([np.sin(uav.ops_az())])

#     # uav_state["tgt_gam_x"] = np.array([np.cos(uav.ops_gam())])
#     # uav_state["tgt_gam_y"] = np.array([np.sin(uav.ops_gam())])

#     uav_state["self_pos_x"] = np.array([position[0]])/env.WINDOW_SIZE_lat
#     uav_state["self_pos_y"] = np.array([position[1]])/env.WINDOW_SIZE_lon
#     uav_state["self_pos_z"] = np.array([position[2]])/env.WINDOW_SIZE_alt

#     uav_state["distances"] = distances/(env.WINDOW_SIZE_lat*3)
#     # act_num = len(env.action_dict_c['blue_' + str(uav.id)])
#     if uav.faction == "blue":
#         # act = env.action_dict_c['blue_' + str(uav.id)]
#         # uav_state.update(act)
#         # if uav.role == "shooter":
#         #     uav_state["role"] = 0
#         # else:
#         #     uav_state["role"] = 1
#         uav_state["vector_psi_x"] = np.array([uav.psi/(np.pi)])
#         # uav_state["vector_psi_y"] = np.array([np.sin(uav.psi)])
#         uav_state["vector_gam_x"] = np.array([uav.gam/(np.pi/2)])
#         # uav_state["vector_gam_y"] = np.array([np.sin(uav.gam)])
#         uav_state["velocity"] = np.array([uav.V/340.0])
#         uav_state["tgt_id"] = uav.tgt.id
        
#         # uav_state["phi_x"] = np.array([np.cos(uav.phi)])
#         # uav_state["phi_y"] = np.array([np.sin(uav.phi)])

#     return uav_state

def get_state_blue(env,uav,position,distances):
    harf = lambda val:2*val-1
    lat_norm = lambda val:harf((val+1)/3)
    uav_state = {}
    uav_state["hitpoint"] = np.array([harf(uav.hitpoint)])

    # uav_state["mrm_num"] = np.array([harf(uav.mrm_num/2)])
    uav_state["mrm_num"] = uav.mrm_num
    if uav.inrange:
        # uav_state["inrange"] = np.array([harf(0)])
        uav_state["inrange"] = 0
    else:
        # uav_state["inrange"] = np.array([harf(1)])
        uav_state["inrange"] = 1
    if uav.detect_launch_ML:
        # uav_state["detect"] = np.array([harf(0)])
        uav_state["detect"] = 0
    else:
        # uav_state["detect"] = np.array([harf(1)])
        uav_state["detect"] = 1
    aa = -(uav.psi + uav.ops_az())
    aa = harf_angle(aa)

    uav_state["tgt_psi_x"] = np.array([np.cos(aa)])
    uav_state["tgt_psi_y"] = np.array([np.sin(aa)])

    aa = -(uav.gam + uav.ops_gam())
    aa = harf_angle(aa)
    aa = -aa
    uav_state["tgt_gam_x"] = np.array([np.cos(aa)])
    uav_state["tgt_gam_y"] = np.array([np.sin(aa)])    
    uav_state["self_pos_x"] = np.array([lat_norm(position[0]/env.WINDOW_SIZE_lat)])
    uav_state["self_pos_y"] = np.array([harf(position[1]/env.WINDOW_SIZE_lon)])
    uav_state["self_pos_z"] = np.array([harf(position[2]/env.WINDOW_SIZE_alt)])
    uav_state["distances"] = harf(distances/(env.WINDOW_SIZE_lat*3))
    uav_state["vector_psi_x"] = np.array([uav.psi/np.pi])
    # uav_state["vector_psi_y"] = np.array([np.sin(uav.psi)])
    uav_state["vector_gam_x"] = np.array([uav.gam/np.pi/2])
    # uav_state["vector_gam_y"] = np.array([np.sin(uav.gam)])
    uav_state["velocity"] = np.array([harf(uav.V/340.0/1.5)])
    uav_state["tgt_id"] = uav.tgt.id
    # uav_state["fire"] = np.array([harf(uav.cool_down/uav.cool_down_limit)])
    # uav_state["tgt_id"] = np.array([harf(uav.tgt.id)])
    # uav_state["fire"] = np.array([harf(uav.cool_down/uav.cool_down_limit)])
        

    return uav_state

def get_state_red(env,red_id,red_distance,op_psi,op_gam):
    harf = lambda val:2*val-1
    lat_norm = lambda val:harf((val+1)/3)
    uav_state = {}
    uav_state["id"] = red_id
    uav_state["hitpoint"] = np.array([harf(env.red[red_id].hitpoint)])
    uav_state["self_pos_x"] = np.array([lat_norm(env.red[red_id].pos[0]/env.WINDOW_SIZE_lat)])
    uav_state["self_pos_y"] = np.array([harf(env.red[red_id].pos[1]/env.WINDOW_SIZE_lon)])
    uav_state["self_pos_z"] = np.array([harf(env.red[red_id].pos[2]/env.WINDOW_SIZE_alt)])
    uav_state["psi_x"] = np.array([np.cos(op_psi)])
    uav_state["psi_y"] = np.array([np.sin(op_psi)])
    uav_state["gam_x"] = np.array([np.cos(op_gam)])
    uav_state["gam_y"] = np.array([np.sin(op_gam)])
    uav_state["velocity"] = np.array([harf(env.red[red_id].V/340.0/1.5)])
    uav_state["distances"] = np.array([harf(red_distance/(env.WINDOW_SIZE_lat*3))])
    return uav_state

def get_state_mrm(env,mrm,is_launched):
    harf = lambda val:2*val-1
    lat_norm = lambda val:harf((val+1)/3)
    uav_state = {}
    if is_launched:
        uav_state["status"] = 1
        uav_state["hitpoint"] = np.array([harf(mrm.hitpoint/mrm.hitpoint_ini)])
        uav_state["parent_id"] = mrm.parent.id
        if mrm.inrange:
            uav_state["inrange"] = 0
        else:
            uav_state["inrange"] = 1
        uav_state["tgt_id"] = mrm.tgt.id
        temp_mrm = limit_calc(mrm.pos, [env.WINDOW_SIZE_lat, env.WINDOW_SIZE_lon, env.WINDOW_SIZE_alt])
        
        uav_state["self_pos_x"] = np.array([lat_norm(temp_mrm/env.WINDOW_SIZE_lat)])
        uav_state["self_pos_y"] = np.array([harf(temp_mrm/env.WINDOW_SIZE_lon)])
        uav_state["self_pos_z"] = np.array([harf(temp_mrm/env.WINDOW_SIZE_alt)])
        uav_state["velocity"] = np.array([harf(mrm.V/340.0/4)])
    else:
        uav_state["status"] = 0
        uav_state["hitpoint"] = np.array([harf(0.0)])
        uav_state["parent_id"] = 2
        uav_state["inrange"] = 1
        uav_state["tgt_id"] = 2
        uav_state["self_pos_x"] = np.array([lat_norm(0.0/env.WINDOW_SIZE_lat)])
        uav_state["self_pos_y"] = np.array([harf(0.0/env.WINDOW_SIZE_lon)])
        uav_state["self_pos_z"] = np.array([harf(0.0/env.WINDOW_SIZE_alt)])
        uav_state["velocity"] = np.array([harf(0.0/340.0/4)])
    
    # uav_state["distances"] = np.array([harf(mrm_distance/(env.WINDOW_SIZE_lat*3))])
    return uav_state


def get_obs(env):
    obs = {}
    for i in range(env.blue_num):
        obs['blue_' + str(i)] = {}
        obs['blue_' + str(i)]["blues"] = {}
        obs['blue_' + str(i)]["reds"] = {}
        obs['blue_' + str(i)]["mrms"] = {}
    observation = {}
    observation["blues"] = {}
    observation["reds"] = {}
    position = np.zeros([env.blue_num+env.red_num,3])
    for i in range(env.blue_num):
        position[i] = env.blue[i].pos

    for i in range(env.red_num):
        position[i+env.blue_num] = env.red[i].pos
    distances = distances_calc(position)
    
    nearest_dist = np.zeros([env.blue_num,env.red_num])
    for i in range(env.blue_num):
        nearest_dist[i,:] = np.sort(distances[i,env.blue_num:])
    
    nearest_id = find_nearest_craft(env)



    for i in range(env.blue_num):
        uav_id = 0
        for j in range(env.blue_num):
            if i == j:
                obs['blue_' + str(i)]["blues"]["self"] = get_state_blue(env,env.blue[j],position[j],distances[j])
            else:
                obs['blue_' + str(i)]["blues"]["blue_0"] = get_state_blue(env,env.blue[j],position[j],distances[j])
                uav_id = uav_id +1
        if nearest_id[i] == 0:
            ids = 0
            nearest_distances = nearest_dist[i,0]
            op_psi = calc_each_ops_psi(env.blue[i],env.red[ids])
            op_gam = calc_each_ops_gam(env.blue[i],env.red[ids])
            obs['blue_' + str(i)]["reds"]["red_" + str(0)] = get_state_red(env,ids,nearest_distances,op_psi,op_gam)

            ids = 1
            nearest_distances = nearest_dist[i,1]
            op_psi = calc_each_ops_psi(env.blue[i],env.red[ids])
            op_gam = calc_each_ops_gam(env.blue[i],env.red[ids])
            obs['blue_' + str(i)]["reds"]["red_" + str(1)] = get_state_red(env,ids,nearest_distances,op_psi,op_gam)
        else:
            ids = 1
            nearest_distances = nearest_dist[i,0]
            op_psi = calc_each_ops_psi(env.blue[i],env.red[ids])
            op_gam = calc_each_ops_gam(env.blue[i],env.red[ids])
            obs['blue_' + str(i)]["reds"]["red_" + str(0)] = get_state_red(env,ids,nearest_distances,op_psi,op_gam)

            ids = 0
            nearest_distances = nearest_dist[i,1]
            op_psi = calc_each_ops_psi(env.blue[i],env.red[ids])
            op_gam = calc_each_ops_gam(env.blue[i],env.red[ids])
            obs['blue_' + str(i)]["reds"]["red_" + str(1)] = get_state_red(env,ids,nearest_distances,op_psi,op_gam)
            
        for j in range(env.blue_num*2):
            obs['blue_' + str(i)]["mrms"]["own_mrm_" + str(j)] = get_state_mrm(env,env.mrm[j],is_launched=False)
        
        for j in range(env.mrm_num):
            if env.mrm[j].parent.faction == "blue":
                is_launched = True
                obs['blue_' + str(i)]["mrms"]["own_mrm_" + str(j)] = get_state_mrm(env,env.mrm[j],is_launched)        
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

def find_nearest_craft(env):
    #2機まで対応
    nearest_id = [0]*env.blue_num
    nearest_dist = [0]*env.blue_num
    for i in range(env.blue_num):
        for j in range(env.red_num):
            temp_dist = np.linalg.norm(env.blue[i].pos - env.red[j].pos)
            if j == 0:
                nearest_dist[i] = temp_dist
            elif nearest_dist[i] > temp_dist:
                nearest_id[i] = 1
                nearest_dist[i] = temp_dist

    return nearest_id

def distances_calc(position):
    tmp_index = np.arange(position.shape[0])
    xx, yy = np.meshgrid(tmp_index, tmp_index)
    distances = np.linalg.norm(position[xx]-position[yy], axis=2)


    return distances

def limit_calc(pos_c, limit):
    pos = pos_c
    for i in range(len(limit)):
        if pos_c[i] <= 0:
            pos[i] = 0

        if pos_c[i] >= limit[i]:
            pos[i] = limit[i]

    return pos