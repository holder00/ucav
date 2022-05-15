# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:53:25 2022

@author: OokiT1
"""
import numpy as np
from utility.harf_angle import harf_angle
from status.distance_search import distance_search


def decision_make(env,action_index,uav_id,faction):
    close_id = distance_search(env,uav_id,faction)
    #0:Close target, 1:Far target
    tgt_id = int(close_id[action_index["tgt_id"],1])
 
    if faction == "blue":
        uav = env.blue[uav_id]
        tgt = env.red[tgt_id]
    elif faction == "red":
        uav = env.red[uav_id]
        tgt = env.blue[tgt_id]


    uav.tgt_update_ML(tgt)
    
    #射撃抑制している可能性があるためuav3d確認のこと
    if action_index["fire"] == 0:
        uav.fire = False
    else:
        uav.fire = True

    # psi = np.arctan2(action_index["vector_psi_y"],action_index["vector_psi_x"])/18
    psi = (action_index["vector_psi_x"])*(np.pi/18)
    gam = (action_index["vector_gam_x"])*(np.pi/3)
    if uav.pos[2] < 1000 and gam < 0:
        gam = 0

    vel = 270+(action_index["velocity"])*40

    uav.V_ref = vel
    uav.psi_ref = uav.psi + psi
    uav.gam_ref = gam

    uav.psi_ref = harf_angle(uav.psi_ref)
    uav.gam_ref = harf_angle(uav.gam_ref)
    
def decision_make_rule(env,action_index,uav_id,faction):
    if faction == "blue":
        uav = env.blue[uav_id]
        for j in range(env.red_num):
            uav.tgt_update(env.red[j])
    elif faction == "red":
        uav = env.red[uav_id]
        for j in range(env.blue_num):
            uav.tgt_update(env.blue[j])
    
    # for i in range(self.red_num):

    uav.fire = True
    if uav.detect_launch:
        ref_psi_sgn = np.pi
        ref_gam_sgn = -1
    else:
        ref_psi_sgn = 0
        ref_gam_sgn = 1
    rand_psi = np.random.uniform(0.9,1.1)
    rand_gam = np.random.uniform(0.9,1.1)
    rand_vel = np.random.uniform(0.9,1.1)
    uav.psi_ref = rand_psi*(uav.ops_az() + ref_psi_sgn)
    uav.gam_ref = rand_gam*ref_gam_sgn*uav.ops_gam()
    uav.V_ref = rand_vel*240
    
    uav.psi_ref = (uav.ops_az() + ref_psi_sgn)
    uav.gam_ref = ref_gam_sgn*uav.ops_gam()
    uav.V_ref = 250
    
    uav.psi_ref = harf_angle(uav.psi_ref)
    uav.gam_ref = harf_angle(uav.gam_ref)