# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:11:44 2021

@author: Takumi
"""
import mmap
import time
# from datetime import datetime
from struct import *
import ctypes
from multiprocessing import Process
import subprocess
import copy

# Kernel32 = ctypes.windll.Kernel32
# mutex = Kernel32.CreateMutexA(0,0,"Global/UAV_MUTEX")


# import gym
# from gym import spaces
import numpy as np
# import math
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from settings.initial_settings import *  # Import settings  # For training
# from settings.test_settings import *  # Import settings  # For testing
# from settings.reset_conditions import reset_conditions
# from modules.resets import reset_red, reset_blue, reset_block
# from modules.observations import get_observation
# from modules.rewards import get_reward
# import cv2
from status.get_obs_3d import get_obs
# from status.reward_calc import reward_calc
from status.init_space import init_space
from UAV.uav_3d import uav_3d
from weapon.missile_3d import missile_3d
from utility.terminate_uavsimproc import teminate_proc
from utility.harf_angle import harf_angle
# from SENSOR.SENSOR_MAIN import sensor
# from numba import jit


class MyEnv(MultiAgentEnv):
    def __init__(self, config={}):
        super(MyEnv, self).__init__()
        np.set_printoptions(precision=2,suppress=True)
        self.WINDOW_SIZE_lat = 200*1000 #画面サイズの決定
        self.WINDOW_SIZE_lon = 150*1000 #画面サイズの決定
        self.WINDOW_SIZE_alt = 12*1000

        self.GOAL_RANGE = 1500 #ゴールの範囲設定
        self.timer = 0
        self.sim_dt = 1
        self.sim_freq = 100
        self.time_limit = 600*2

        self.blue_num = 2

        self.blue_side = 10*1000 #int(self.WINDOW_SIZE_lat/4)
        area_max = self.WINDOW_SIZE_lat - self.WINDOW_SIZE_lat/4*0
        self.blue_safe_area = [area_max-self.blue_side,area_max]

        self.red_num = 2
        self.red_side = 10*1000 #int(self.WINDOW_SIZE_lat/4)
        area_min = self.WINDOW_SIZE_lat/4*0
        self.red_safe_area = [area_min,area_min+self.red_side]
        self.ng_range = 250
        self.ng_area_lat = [self.ng_range, self.WINDOW_SIZE_lat-self.ng_range]
        self.ng_area_lon = [self.ng_range, self.WINDOW_SIZE_lon-self.ng_range]
        self.ng_area_alt = [self.ng_range, self.WINDOW_SIZE_alt-self.ng_range]

        self.action_space = init_space.action(self)
        self.observation_space = init_space.observation(self)

        self.action_dict_c = {}
        self.init_flag = False
        # obs = self.reset()
        # a_proc_id = self.a.pid
        # print(self.a.pid)
        # self.a.kill()
    def reset(self):
        obs = {}
        self.sim_cnt = 0
        self.blue = [0]*self.blue_num
        self.red = [0]*self.red_num
        self.before_distance = {}
        self.timer = 0
        self.reward_k = 0
        self.reward_d = 0

        self.reward_fire = 0
        self.reward_win = 0
        self.rewards_total = {}
        self.reward_total = 0
        self.sim_comp = False
        self.sim_time = 0
        mrm_num_ini = 0

        for i in range(self.blue_num):
            self.blue[i] = uav_3d(self.WINDOW_SIZE_lat-self.blue_side, self.WINDOW_SIZE_lon, self.blue_safe_area,"blue",i,0)
            mrm_num_ini = mrm_num_ini + self.blue[i].mrm_num
        for i in range(self.red_num):
            self.red[i] = uav_3d(self.red_side, self.WINDOW_SIZE_lon, self.red_safe_area,"red",i,0)
            mrm_num_ini = mrm_num_ini + self.red[i].mrm_num
        self.mrm = [0]*(mrm_num_ini)
        self.mrm_num = 0

        for i in range(self.blue_num):
            for j in range(self.red_num):
                self.blue[i].tgt_update(self.red[j])
                self.red[j].tgt_update(self.blue[i])
        for i in range(self.blue_num):
            self.action_dict_c["blue_" + str(i)] = self.action_space.sample()

        for i in range(self.blue_num):
            self.rewards_total["blue_" + str(i)] = 0
            self.before_distance["blue_" + str(i)] = np.linalg.norm(self.blue[i].pos - self.red[i].pos)
        # self.sensor = sensor(self.blue,self.red)
        #UCAV.exeが起動している場合、プロセスキルする。
        teminate_proc.UAVsimprockill(proc_name="UCAV.exe")
        #exe起動待ち
        time.sleep(0.5)
        #exeのパス
        if not self.init_flag:
            # subprocess.Popen(r"./UCAV.exe")
            subprocess.Popen(r"./UCAV.exe /BLUE 1 /RED 0 /MEM 1")
            #exe起動待ち
            time.sleep(0.5)
            subprocess.Popen(r"./UCAV.exe /BLUE 1 /RED 0 /MEM 2")
            #exe起動待ち
            time.sleep(0.5)
            subprocess.Popen(r"./UCAV.exe /BLUE 1 /RED 0 /MEM 3")
            #exe起動待ち
            time.sleep(0.5)
            subprocess.Popen(r"./UCAV.exe /BLUE 1 /RED 0 /MEM 4")
            #exe起動待ち
            time.sleep(0.5)

            #共有メモリに初期値設定
            for i in range(self.blue_num):
                self.SetPyToSimData(self.blue[i],self.red,1,0,i+1)
            for i in range(self.red_num):
                self.SetPyToSimData(self.red[i],self.red,1,0,i+self.blue_num+1)

        obs = get_obs(self)
        return obs

    def step(self, action_dict):
        
        obs, rewards, dones, infos = {}, {}, {}, {}
        rewards_fire = {}
        rewards_inrage = {}
        rewards_inraged = {}
        reward_distance = {}
        self.timer = self.timer + self.sim_dt

        reward_temp = [0]*self.blue_num

        reward_k_temp = {}
        reward_d_temp = {}
        self.reward_fire = 0
        # reward_ng = [0]*self.blue_num

        # position = np.zeros([self.blue_num+self.red_num,2])

        is_touch = False
        is_fin = False
        dones["__all__"] = False
        reward = 0
        # reward_detect = 0
        # reward_inrange = 0
        # for i in range(self.red_num):
        #     team_pos = (self.blue[0].pos - self.blue[1].pos)/2
        #     if np.linalg.norm(team_pos - self.blue[0].tgt.pos) > np.linalg.norm(team_pos - self.red[i].pos):
        #         # print(self.faction, "TGT CHANGE")
        #         self.blue[0].tgt = self.red[i]
        #         self.blue[1].tgt = self.red[i]
        if np.linalg.norm(self.blue[0].pos - self.blue[0].tgt.pos) < np.linalg.norm(self.blue[1].pos - self.blue[1].tgt.pos):
            
            self.blue[0].role = "decoy"
            self.blue[1].role = "shooter"
        else:
            self.blue[0].role = "shooter"  
            self.blue[1].role = "decoy"   
        
        penalty_trigger = {}
        # print(self.timer," tgt:",self.blue[0].tgt.id,"Blue_0 Role:",self.blue[0].role,"Blue_1 Role:",self.blue[1].role)
        for i in range(self.blue_num):
            if not self.blue[i].hitpoint == 0:
                action_index = action_dict["blue_" + str(i)]
                if action_index["tgt_id"] == 0:
                    tgt_id = 0
                else:
                    tgt_id = 1
                self.blue[i].tgt_update_ML(self.red[tgt_id])
                # self.blue[i].tgt_update_ML(self.red[i])
                # self.blue[i].tgt_update_ML(self.red[tgt_id])

                if action_index["fire"] == 0:
                    self.blue[i].cool_down = 0

                else:
                    self.blue[i].cool_down = self.blue[i].cool_down_limit

                # psi = np.arctan2(action_index["vector_psi_y"],action_index["vector_psi_x"])
                # gam = np.arctan(action_index["vector_gam_y"]/action_index["vector_gam_x"])
                psi = (action_index["vector_psi_x"])*np.pi
                gam = (action_index["vector_gam_x"])*(np.pi/2)
                vel = ((action_index["velocity"]+1)/2)*340
                # gam = np.deg2rad(0)
                # psi = (np.pi/180)*(self.timer) - np.pi
                # psi = np.pi
                # psi = np.deg2rad(135)
                self.blue[i].V_ref = vel[0]
                self.blue[i].psi_ref = self.blue[i].ops_az()+psi[0]
                self.blue[i].gam_ref = self.blue[i].ops_gam()*0+gam[0]

            # if self.blue[i].detect_launch_ML:
            # if action_index["role"] == 0:
            #     self.blue[i].role = "shooter"
            # else:
            #     self.blue[i].role = "decoy"
                
            # if self.blue[i].role == "decoy":
                
            #     if self.blue[i].ops_az() - self.blue[i].psi > np.pi:
            #         ref_psi_sgn = np.pi/2
            #     else:
            #         ref_psi_sgn = -np.pi/2
                
            #     ref_gam_sgn = 1
                
            #     self.blue[i].psi_ref = self.blue[i].ops_az() + ref_psi_sgn
            #     self.blue[i].gam_ref = ref_gam_sgn*self.blue[i].ops_gam()
                
            #     # ref_psi_sgn = 
            # if self.blue[i].role == "shooter":
            #     if self.blue[i].detect_launch_ML:
            #         ref_psi_sgn = np.pi
            #         ref_gam_sgn = -1
            #     else:
            #         ref_psi_sgn = 0
            #         ref_gam_sgn = 1
            #     self.blue[i].psi_ref = self.blue[i].ops_az() + ref_psi_sgn
            #     self.blue[i].gam_ref = ref_gam_sgn*self.blue[i].ops_gam()
            self.blue[i].psi_ref = harf_angle(self.blue[i].psi_ref)
            self.blue[i].gam_ref = harf_angle(self.blue[i].gam_ref)

            
        for i in range(self.red_num):
            if self.red[i].detect_launch:
                ref_psi_sgn = np.pi
                ref_gam_sgn = -1
            else:
                ref_psi_sgn = 0
                ref_gam_sgn = 1
            self.red[i].psi_ref = self.red[i].ops_az() + ref_psi_sgn
            self.red[i].gam_ref = ref_gam_sgn*self.red[i].ops_gam()
            
            self.red[i].psi_ref = harf_angle(self.red[i].psi_ref)
            self.red[i].gam_ref = harf_angle(self.red[i].gam_ref)


#python ⇒ uavsim
        # if self.timer > 0:
        self.sim_cnt = self.sim_cnt + 1
        for i in range(self.blue_num):
            self.SetPyToSimData(self.blue[i],self.red,1,self.timer,i+1)
            
        for i in range(self.red_num):
            self.SetPyToSimData(self.red[i],self.red,1,self.timer,i+self.blue_num+1)

        # print("UAVsim calc start @ "+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        self.perf_time = time.perf_counter()
        while not self.sim_comp:
            self.sim_time = 0
            for i in range(self.blue_num):
                self.GetSimToPyData(self.blue,self.red,i+1,"blue",i)
                
            for i in range(self.red_num):
                self.GetSimToPyData(self.blue,self.red,i+self.blue_num+1,"red",i)
                
            if self.sim_time == (self.blue_num + self.red_num):
                self.sim_comp = True
        else:
            self.sim_comp = False
            # print("UAVsim calc complete @ "+ str(time.perf_counter() - self.perf_time) + datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
            # print(self.blue[0].pos)
        
#python ⇒ uavsim

        
        for i in range(self.blue_num):
            if self.blue[i].hitpoint == 0:
                self.blue[i].pos = np.array([2*self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon/2, 0])
            else:
                self.blue[i].pos = self.limit_calc(self.blue[i].pos, [self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon, self.WINDOW_SIZE_alt])

        for i in range(self.red_num):
            if self.red[i].hitpoint == 0:
                self.red[i].pos = np.array([-self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon/2, 0])
            else:
                self.red[i].pos = self.limit_calc(self.red[i].pos, [self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon, self.WINDOW_SIZE_alt])

        for i in range(self.mrm_num):
            self.mrm[i].update_status(self.sim_dt)
            if np.linalg.norm(self.mrm[i].tgt.pos - self.mrm[i].pos) < self.mrm[i].destruction_range:
                self.mrm[i].tgt.hitpoint = 0
                self.mrm[i].hitpoint = 0
                if self.mrm[i].tgt.faction == "red":
                    self.mrm[i].tgt.pos = np.array([-self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon/2, 0])
                    self.mrm[i].pos = np.array([2*self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon/2, 0])
                    self.reward_k = self.reward_k + 1
                    reward_k_temp["blue_" + str(self.mrm[i].parent.id)] = 1
                    print(str(self.timer) + " blue_" + str(self.mrm[i].parent.id) + " Splash :" + "red_"+ str(self.mrm[i].tgt.id))

                if self.mrm[i].tgt.faction == "blue":
                    self.mrm[i].tgt.pos = np.array([2*self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon/2, 0])
                    self.mrm[i].pos = np.array([-self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon/2, 0])
                    self.reward_d = self.reward_d + 1
                    reward_d_temp["blue_" + str(self.mrm[i].tgt.id)] = 1
                    print(str(self.timer) + " red_" + str(self.mrm[i].parent.id) + " Splash :" + "blue_"+ str(self.mrm[i].tgt.id))

        # self.s_blue,self.s_red = self.sensor.update(self.blue,self.red)
        # self.blue = copy.deepcopy(self.s_blue)
        # self.red = copy.deepcopy(self.s_red)

        penalty_fire = {}
        
        for i in range(self.blue_num):
            #発射判定
            penalty_fire["blue_"+str(i)] = 0
            if self.blue[i].hitpoint != 0:
                if self.blue[i].tgt_inrange():
                    if self.blue[i].can_launch_ML():
                        self.mrm[self.mrm_num] = missile_3d(self.blue[i])
                        launch_distance = np.linalg.norm(self.blue[i].tgt.pos - self.blue[i].pos)
                        rewards_fire["blue_" + str(i)] = 1-(launch_distance/self.blue[i].radar_range)**2
                        self.reward_fire = self.reward_fire + 1
                        
                        for j in range(self.mrm_num):
                            # print(self.mrm[j].parent.faction,self.blue[i].tgt.id,self.red[j].tgt.id)
                            if self.mrm[j].tgt.faction == "red" and self.blue[i].tgt.id == self.mrm[j].tgt.id:
                                penalty_fire["blue_"+str(i)] = penalty_fire["blue_"+str(i)] - 1
                                print("Same tgt shoot")
                        self.mrm_num = self.mrm_num + 1
                        print(str(self.timer) + " blue_" + str(i) + " Shoot at " + "red_" + str(self.blue[i].tgt.id),
                              "launch distance : " + str(launch_distance),self.blue[i].inrange,self.blue[i].fire)
                    else:
                        if action_dict["blue_" + str(i)]["fire"] == 1:
                            penalty_trigger["blue_" + str(i)] = -1
                            # print(str(self.timer) +" Penalty!! cannot launch " + "Blue_"+str(i))
                else:
                    if action_dict["blue_" + str(i)]["fire"] == 1:
                        penalty_trigger["blue_" + str(i)] = -1
                        # print(str(self.timer) +" Penalty!! not inrange " + "Blue_"+str(i))

        for i in range(self.red_num):
            #発射判定
            if self.red[i].tgt_inrange():
                if self.red[i].can_launch():
                    self.mrm[self.mrm_num] = missile_3d(self.red[i])
                    self.mrm_num = self.mrm_num + 1
                    print(str(self.timer) + " red_" + str(i) + " Shoot at " + "blue_" + str(self.red[i].tgt.id))

        reward_evade = {}
        for i in range(self.mrm_num):
            for j in range(self.blue_num):
                self.blue[j].MAWS_ML(self.mrm[i])
                if self.blue[j].detect_launch_ML:
                    reward_evade["blue_" + str(j)] = 1
            for j in range(self.red_num):
                self.red[j].MAWS(self.mrm[i])



        for i in range(self.red_num):
            for j in range(self.blue_num):
                self.red[i].tgt_update(self.blue[j])

        self.action_dict_c = action_dict
        obs = get_obs(self)

        for i in range(self.blue_num):
            rewards_inrage["blue_" + str(i)] = 0
            rewards_inraged["blue_" + str(i)] = 0
            if self.blue[i].inrange:
                # if self.blue[i].role == "shooter":
                rewards_inrage["blue_" + str(i)] = 1

        for i in range(self.red_num):
            if self.red[i].inrange:
                # if self.blue[i].role == "shooter":
                rewards_inraged["blue_" + str(self.red[i].tgt.id)] =\
                    rewards_inraged["blue_" + str(self.red[i].tgt.id)] -1
                # if self.blue[i].role == "decoy":
                #     rewards_inraged["blue_" + str(self.red[i].tgt.id)] =\
                #         rewards_inraged["blue_" + str(self.red[i].tgt.id)] +1
                        
        penalty_role = {}
        if self.blue[0].role == self.blue[1].role:
            penalty_role["blue_" + str(0)] = 1
            penalty_role["blue_" + str(1)] = 1
            
        penalty_shooter = {}
        penalty_decoy = {}
        for i in range(self.blue_num):
            if self.blue[i].role == "shooter" and np.linalg.norm(self.blue[i].pos - self.blue[i].tgt.pos) > self.blue[i].radar_range:
                penalty_shooter["blue_" + str(i)] = 1
            if self.blue[i].id == 0:
                budy_id = 1
            else:
                budy_id = 0
            if self.blue[i].role == "decoy" and ("blue_" + str(budy_id) in rewards_inraged):
                penalty_decoy["blue_" + str(i)] = 1
                
        # tgt_id = np.zeros(4)
        # for i in range(self.blue_num):
        #     reward_tgt = np.count_nonzero(tgt_id)
        reward_touchdown = {}

        for i in range(self.blue_num):
            # 終了判定
            if np.linalg.norm(self.blue[i].pos - self.blue[i].tgt.pos) < 10*1000:
                print("Too Close Finish")
                is_fin = True
            if self.blue[i].pos[2] == 0 and self.blue[i].hitpoint != 0:
                self.blue[i].hitpoint = 0
                reward_touchdown["blue_"+str(i)] = -10

                print(self.timer,"blue_"+str(i),"DOWN")
                # self.reward_d = self.reward_d + 1
                is_touch = True
            elif self.blue[i].pos[2] > 0:

                reward_touchdown["blue_"+str(i)] = 0.005*0
        
        is_mrm_zero = 0
        for i in range(self.blue_num):
            if self.blue[i].mrm_num == 0:
                is_mrm_zero = is_mrm_zero + 1
        for i in range(self.red_num):
            if self.red[i].mrm_num == 0:
                is_mrm_zero = is_mrm_zero + 1
        if is_mrm_zero == self.blue_num + self.red_num:
            is_mrm_zero = 0
            for i in range(self.mrm_num):
                if self.mrm[i].hitpoint == 0:
                    is_mrm_zero = is_mrm_zero + 1
            if is_mrm_zero == len(self.mrm)+1:
                is_fin = True
                
        if self.reward_k == self.red_num:
            for i in range(self.mrm_num):
                if not self.mrm[i].hitpoint == 0:
                    is_fin = False
                else:
                    is_fin = True

        if self.reward_d == self.blue_num:
            is_fin = True
            
        # for i in range(self.blue_num):
        if self.blue[0].hitpoint == 0 and self.blue[1].hitpoint == 0:
            is_fin = True
        else:
            is_fin = False

        if is_fin or self.timer >= self.time_limit:
            dones["__all__"] = True
            for i in range(self.blue_num):
                self.SetPyToSimData(self.blue[i],self.red,1,-1,i+1)
            for i in range(self.red_num):
                self.SetPyToSimData(self.red[i],self.red,1,-1,i+self.blue_num+1)
            time.sleep(0.5)
            if (not self.reward_d == self.blue_num and self.reward_k == self.red_num):
                self.reward_win = 100
                print("WIN")
            elif is_touch:
                self.reward_win = 0
                print("DOWN LOSE")
            elif self.reward_k != self.red_num or self.timer >= self.time_limit:
                self.reward_win = -10*0 + 0.0*self.reward_k
                print("TIME LIMIT LOSE")

                

        # reward_ng = reward_calc.reward_ng(self.blue, self.ng_area_lat, self.ng_area_lon, self.ng_area_alt, self.ng_range)
        distance = {}
        for i in range(self.blue_num):
            distance["blue_" +str(i)] = np.linalg.norm(self.blue[i].pos - self.red[i].pos)
            reward_distance["blue_" +str(i)] = self.before_distance["blue_" +str(i)] - distance["blue_" +str(i)]
        reward_temp = self.reward_win
        reward = reward_temp
        self.before_distance = distance
        self.reward_total = self.reward_total + reward

        for i in range(self.blue_num):
            rewards["blue_" + str(i)] = 0

            if not self.blue[i].hitpoint == 0 or dones["__all__"]:
                #
                if "blue_" + str(i) in rewards_fire:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 20.0*rewards_fire["blue_" + str(i)]
                if "blue_" + str(i) in rewards_inrage:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 0.01*rewards_inrage["blue_" + str(i)]
                if "blue_" + str(i) in rewards_inraged:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 0.01*rewards_inraged["blue_" + str(i)]*0
                if "blue_" + str(i) in reward_k_temp:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 40.0*reward_k_temp["blue_" + str(i)]
                if "blue_" + str(i) in reward_d_temp:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] - 10.0*reward_d_temp["blue_" + str(i)]
                if "blue_" + str(i) in reward_distance:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 0.00001*reward_distance["blue_" + str(i)]*0
                if "blue_" + str(i) in penalty_role:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] - 0.05*penalty_role["blue_" + str(i)]*0  
                if "blue_" + str(i) in penalty_shooter:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] - 0.0015*penalty_shooter["blue_" + str(i)]*0
                if "blue_" + str(i) in penalty_decoy:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] - 0.0015*penalty_decoy["blue_" + str(i)]*0 
                if "blue_" + str(i) in reward_evade:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 0.001*reward_evade["blue_" + str(i)]
                if "blue_" + str(i) in penalty_trigger:
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 0.001*penalty_trigger["blue_" + str(i)]
                if "blue_" + str(i) in penalty_fire: #”有効弾”が”目標”に対して”飛翔中”に”目標”に”射撃”するとペナルティ "有効弾"：エネルギー残量で判断したいところ
                    rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + 2.0*penalty_fire["blue_" + str(i)]



            if "blue_" + str(i) in reward_touchdown:
                rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + reward_touchdown["blue_" + str(i)]
            rewards["blue_" + str(i)] = rewards["blue_" + str(i)] + reward
            self.rewards_total["blue_" + str(i)] = self.rewards_total["blue_" + str(i)] + rewards["blue_" + str(i)]
            # print("blue_" + str(i),is_touch,is_fin,self.timer,rewards["blue_" + str(i)],self.rewards_total["blue_" + str(i)],
            #       np.rad2deg(self.blue[i].psi),np.rad2deg(self.blue[i].gam))
            if dones["__all__"]:
                print("blue_" + str(i),is_touch,is_fin,self.timer,rewards["blue_" + str(i)],self.rewards_total["blue_" + str(i)])
        self.reward_num = rewards

        for i in range(self.blue_num):
            # if self.blue[i].hitpoint == 0 or dones["__all__"]:
            if dones["__all__"]:
                dones["blue_" + str(i)] = True
            else:
                dones["blue_" + str(i)] = False

        # info
        for i in range(self.blue_num):
            if self.blue[i].hitpoint == 0:
                infos["blue_" + str(i)] = {}

        return obs, rewards, dones, infos
    # @jit
    def limit_calc(self, pos_c, limit):
        pos = pos_c
        for i in range(len(limit)):
            if pos_c[i] <= 0:
                pos[i] = 0

            if pos_c[i] >= limit[i]:
                pos[i] = limit[i]

        return pos
    #------masubuchi add------
    def SetPyToSimData(self,blue,red,flag,stepcnt,memno):
        #shmem2 = mmap.mmap(0,232,"PYTHON_SIM_MEM")
        shmem2 = mmap.mmap(0,232,"PYTHON_SIM_MEM" + str(memno))
        BlueData = [[0] * 8,[0] * 8]
        RedData = [[0] * 8,[0] * 8]

        #初期化時
        BlueData[0] = [blue.pos[0], blue.pos[1], blue.pos[2], blue.V, np.rad2deg(blue.gam), np.rad2deg(blue.psi), blue.fire, blue.fire]
        # for i in range(len(red)):
        #     RedData[i] = [red[i].pos[0], red[i].pos[1], red[i].pos[2], red[i].V*0+250, np.rad2deg(red[i].gam)*0, np.rad2deg(red[i].psi),red[i].fire,blue[i].fire]

        #Step更新時
        if stepcnt > 0:
            BlueData[0][3] = blue.V_ref
            BlueData[0][4] = np.rad2deg(blue.gam_ref)
            BlueData[0][5] = np.rad2deg(blue.psi_ref)
            if blue.fire:
                BlueData[0][6] = 1
                BlueData[0][7] = 1


            # for i in range(len(red)):
            #     RedData[i][3] = red[i].V_ref*0+250
            #     RedData[i][4] = np.rad2deg(red[i].gam_ref)
            #     RedData[i][5] = np.rad2deg(red[i].psi_ref)
            #     if red[i].fire:
            #         RedData[i][6] = 1
            #         RedData[i][7] = 1


        #blue1-2,red1-2
        #double lat,lon,alt,dVc,dGamC,dPsi
        #int Python指令 1:初期化中 2:初期化完了 3:Sim終了
        #int stepカウント(初期化時：0,step実行毎にカウント 1up sim側は値変化で動作させる)
        OutValue = pack('ddddddiiddddddiiddddddiiddddddiiii',
                        BlueData[0][0], BlueData[0][1], BlueData[0][2], BlueData[0][3], BlueData[0][4], BlueData[0][5], BlueData[0][6],BlueData[0][7],
                        BlueData[1][0], BlueData[1][1], BlueData[1][2], BlueData[1][3], BlueData[1][4], BlueData[1][5], BlueData[0][6],BlueData[0][7],
                        RedData[0][0], RedData[0][1], RedData[0][2], RedData[0][3], RedData[0][4], RedData[0][5], RedData[0][6],RedData[0][7],
                        RedData[1][0], RedData[1][1], RedData[1][2], RedData[1][3], RedData[1][4], RedData[1][5], RedData[1][6],RedData[1][7],
                        flag,stepcnt)
        # print("    ","Bule[1] lat",BlueData[1][0],"lon",BlueData[1][1],"alt",BlueData[1][2],"Vc",BlueData[1][3],"Gamc",BlueData[1][4],"Psic",BlueData[1][5])
        # print("init","Red[0] lat",RedData[0][0],"lon",RedData[0][1],"alt",RedData[0][2],"Vc",RedData[0][3],"Gamc",RedData[0][4],"Psic",RedData[0][5])
        # print("    ","Red[1] lat",RedData[1][0],"lon",RedData[1][1],"alt",RedData[1][2],"Vc",RedData[1][3],"Gamc",RedData[1][4],"Psic",RedData[1][5])

        shmem2.write(OutValue)

    def GetSimToPyData(self,blue,red,memno,faction,uav_id):
        # print("GetSIM-->")
        #shmem = mmap.mmap(0,296,"UAV_SIM_MEM")
        shmem=(mmap.mmap(0,296,"UAV_SIM_MEM"),
               mmap.mmap(0,296,"UAV_SIM_MEM1"),
               mmap.mmap(0,296,"UAV_SIM_MEM2"),
               mmap.mmap(0,296,"UAV_SIM_MEM3"),
               mmap.mmap(0,296,"UAV_SIM_MEM4"),
               mmap.mmap(0,296,"UAV_SIM_MEM5"))

        #Uav[0] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #Uav[1] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #Uav[2] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #Uav[3] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #int Python指令
        #InValue = unpack('ddddddddddddddddddddddddddddddddddddii',shmem)
        InValue = unpack('ddddddddddddddddddddddddddddddddddddii',shmem[memno])
        #print(InValue)
        # print("Uav[0]:Vc",InValue[0],"VGam",InValue[1],"Psi",InValue[2],"Roll",InValue[3],"Pitch",InValue[4],"Yaw",InValue[5],"N",InValue[6],"E",InValue[7],"H",InValue[8])
        # print("Uav[1]:Vc",InValue[9],"VGam",InValue[10],"Psi",InValue[11],"Roll",InValue[12],"Pitch",InValue[13],"Yaw",InValue[14],"N",InValue[15],"E",InValue[16],"H",InValue[17])
        # print("Uav[2]:Vc",InValue[18],"VGam",InValue[19],"Psi",InValue[20],"Roll",InValue[21],"Pitch",InValue[22],"Yaw",InValue[23],"N",InValue[24],"E",InValue[25],"H",InValue[26])
        # print("Uav[3]:Vc",InValue[27],"VGam",InValue[28],"Psi",InValue[29],"Roll",InValue[30],"Pitch",InValue[31],"Yaw",InValue[32],"N",InValue[33],"E",InValue[34],"H",InValue[35])

        data_length = int((len(InValue)-2)/4)
        # print(self.timer,InValue[37],self.sim_time,self.sim_comp,datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

        if (InValue[37] - self.timer) == 0:
            self.sim_time = self.sim_time + 1
        # print(faction,uav_id,self.timer,InValue[37],self.sim_time)

        # self.sim_time = InValue[37]
        if faction == "blue":
            self.blue[uav_id].pos[0] = copy.deepcopy(InValue[6])
            self.blue[uav_id].pos[1] = copy.deepcopy(InValue[7])
            self.blue[uav_id].pos[2] = copy.deepcopy(InValue[8])
            self.blue[uav_id].V = copy.deepcopy(InValue[0])
            self.blue[uav_id].gam = np.deg2rad(copy.deepcopy(InValue[1]))
            self.blue[uav_id].psi = np.deg2rad(copy.deepcopy(InValue[2]))
            #rad→°
            self.blue[uav_id].phi = copy.deepcopy(InValue[3])
            self.blue[uav_id].pitch = copy.deepcopy(InValue[4])
            self.blue[uav_id].yaw = copy.deepcopy(InValue[5])
        if faction == "red":
            self.red[uav_id].pos[0] = copy.deepcopy(InValue[6])
            self.red[uav_id].pos[1] = copy.deepcopy(InValue[7])
            self.red[uav_id].pos[2] = copy.deepcopy(InValue[8])
            self.red[uav_id].V = copy.deepcopy(InValue[0])
            self.red[uav_id].gam = np.deg2rad(copy.deepcopy(InValue[1]))
            self.red[uav_id].psi = np.deg2rad(copy.deepcopy(InValue[2]))
            #rad→°
            self.red[uav_id].phi = copy.deepcopy(InValue[3])
            self.red[uav_id].pitch = copy.deepcopy(InValue[4])
            self.red[uav_id].yaw = copy.deepcopy(InValue[5])
            # self.red[uav_id].pos[0] = InValue[24]
            # self.red[uav_id].pos[1] = InValue[25]
            # self.red[uav_id].pos[2] = InValue[26]
            # self.red[uav_id].V = InValue[18];
            # self.red[uav_id].gam = np.deg2rad(InValue[19])
            # self.red[uav_id].psi = np.deg2rad(InValue[20])
            # #rad→°
            # self.red[uav_id].phi = InValue[21]
            # self.red[uav_id].pitch = InValue[22]
            # self.red[uav_id].yaw = InValue[23]

#------masubuchi add------
# #------masubuchi add------
#     def SetPyToSimData(self,blue,red,flag,stepcnt,memno):
#         #shmem2 = mmap.mmap(0,232,"PYTHON_SIM_MEM")
#         shmem2 = mmap.mmap(0,232,"PYTHON_SIM_MEM" + str(memno))
#         BlueData = [[0.0] * 8,[0.0] * 8]
#         RedData = [[0.0] * 8,[0.0] * 8]

#         #初期化時
#         for i in range(len(blue)):
#             BlueData[i] = [blue[i].pos[0], blue[i].pos[1], blue[i].pos[2], blue[i].V*0+340, np.rad2deg(blue[i].gam)*0, np.rad2deg(blue[i].psi), blue[i].fire, blue[i].fire]
#         for i in range(len(red)):
#             RedData[i] = [red[i].pos[0], red[i].pos[1], red[i].pos[2], red[i].V*0+250, np.rad2deg(red[i].gam)*0, np.rad2deg(red[i].psi),red[i].fire,blue[i].fire]

#         #Step更新時
#         if stepcnt > 0:
#             for i in range(len(blue)):
#                 BlueData[i][3] = blue[i].V_ref
#                 BlueData[i][4] = np.rad2deg(blue[i].gam_ref)
#                 BlueData[i][5] = np.rad2deg(blue[i].psi_ref)
#                 if blue[i].fire:
#                     BlueData[i][6] = 1
#                     BlueData[i][7] = 1


#             for i in range(len(red)):
#                 RedData[i][3] = red[i].V_ref*0+250
#                 RedData[i][4] = np.rad2deg(red[i].gam_ref)
#                 RedData[i][5] = np.rad2deg(red[i].psi_ref)
#                 if red[i].fire:
#                     RedData[i][6] = 1
#                     RedData[i][7] = 1


#         #blue1-2,red1-2
#         #double lat,lon,alt,dVc,dGamC,dPsi
#         #int Python指令 1:初期化中 2:初期化完了 3:Sim終了
#         #int stepカウント(初期化時：0,step実行毎にカウント 1up sim側は値変化で動作させる)
#         OutValue = pack('ddddddiiddddddiiddddddiiddddddiiii',
#                         BlueData[0][0], BlueData[0][1], BlueData[0][2], BlueData[0][3], BlueData[0][4], BlueData[0][5], BlueData[0][6],BlueData[0][7],
#                         BlueData[1][0], BlueData[1][1], BlueData[1][2], BlueData[1][3], BlueData[1][4], BlueData[1][5], BlueData[0][6],BlueData[0][7],
#                         RedData[0][0], RedData[0][1], RedData[0][2], RedData[0][3], RedData[0][4], RedData[0][5], RedData[0][6],RedData[0][7],
#                         RedData[1][0], RedData[1][1], RedData[1][2], RedData[1][3], RedData[1][4], RedData[1][5], RedData[1][6],RedData[1][7],
#                         flag,stepcnt)
#         # print("    ","Bule[1] lat",BlueData[1][0],"lon",BlueData[1][1],"alt",BlueData[1][2],"Vc",BlueData[1][3],"Gamc",BlueData[1][4],"Psic",BlueData[1][5])
#         # print("init","Red[0] lat",RedData[0][0],"lon",RedData[0][1],"alt",RedData[0][2],"Vc",RedData[0][3],"Gamc",RedData[0][4],"Psic",RedData[0][5])
#         # print("    ","Red[1] lat",RedData[1][0],"lon",RedData[1][1],"alt",RedData[1][2],"Vc",RedData[1][3],"Gamc",RedData[1][4],"Psic",RedData[1][5])

#         shmem2.write(OutValue)

#     def GetSimToPyData(self,blue,red,memno):
#         # print("GetSIM-->")
#         #shmem = mmap.mmap(0,296,"UAV_SIM_MEM")
#         shmem=(mmap.mmap(0,296,"UAV_SIM_MEM"),
#                mmap.mmap(0,296,"UAV_SIM_MEM1"),
#                mmap.mmap(0,296,"UAV_SIM_MEM2"),
#                mmap.mmap(0,296,"UAV_SIM_MEM3"),
#                mmap.mmap(0,296,"UAV_SIM_MEM4"),
#                mmap.mmap(0,296,"UAV_SIM_MEM5"))

#         #Uav[0] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
#         #Uav[1] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
#         #Uav[2] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
#         #Uav[3] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
#         #int Python指令
#         #InValue = unpack('ddddddddddddddddddddddddddddddddddddii',shmem)
#         InValue = unpack('ddddddddddddddddddddddddddddddddddddii',shmem[memno])
#         #print(InValue)
#         # print("Uav[0]:Vc",InValue[0],"VGam",InValue[1],"Psi",InValue[2],"Roll",InValue[3],"Pitch",InValue[4],"Yaw",InValue[5],"N",InValue[6],"E",InValue[7],"H",InValue[8])
#         # print("Uav[1]:Vc",InValue[9],"VGam",InValue[10],"Psi",InValue[11],"Roll",InValue[12],"Pitch",InValue[13],"Yaw",InValue[14],"N",InValue[15],"E",InValue[16],"H",InValue[17])
#         # print("Uav[2]:Vc",InValue[18],"VGam",InValue[19],"Psi",InValue[20],"Roll",InValue[21],"Pitch",InValue[22],"Yaw",InValue[23],"N",InValue[24],"E",InValue[25],"H",InValue[26])
#         # print("Uav[3]:Vc",InValue[27],"VGam",InValue[28],"Psi",InValue[29],"Roll",InValue[30],"Pitch",InValue[31],"Yaw",InValue[32],"N",InValue[33],"E",InValue[34],"H",InValue[35])

#         data_length = int((len(InValue)-2)/4)
#         # print(self.timer,InValue[37],self.sim_time,self.sim_comp,datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
#         # print(self.timer,InValue[37])
#         if (InValue[37] - self.sim_time) != 0:
#             self.sim_comp = True
#         self.sim_time = InValue[37]

#         for i in range(len(blue)):
#             self.blue[i].pos[0] = InValue[i*data_length + 6]
#             self.blue[i].pos[1] = InValue[i*data_length + 7]
#             self.blue[i].pos[2] = InValue[i*data_length + 8]
#             self.blue[i].V = InValue[i*data_length + 0]
#             self.blue[i].gam = np.deg2rad(InValue[i*data_length + 1])
#             self.blue[i].psi = np.deg2rad(InValue[i*data_length + 2])
#             #rad→°
#             self.blue[i].phi = InValue[i*data_length + 3]
#             self.blue[i].pitch = InValue[i*data_length + 4]
#             self.blue[i].yaw = InValue[i*data_length + 5]

#         for i in range(len(red)):
#             self.red[i].pos[0] = InValue[i*data_length + 24]
#             self.red[i].pos[1] = InValue[i*data_length + 25]
#             self.red[i].pos[2] = InValue[i*data_length + 26]
#             self.red[i].V = InValue[i*data_length + 18];
#             self.red[i].gam = np.deg2rad(InValue[i*data_length + 19])
#             self.red[i].psi = np.deg2rad(InValue[i*data_length + 20])
#             #rad→°
#             self.red[i].phi = InValue[i*data_length + 21]
#             self.red[i].pitch = InValue[i*data_length + 22]
#             self.red[i].yaw = InValue[i*data_length + 23]

# #------masubuchi add------
    # def get_rotation_matrix(self, rad):
    #     rot = np.array([[np.cos(rad), -np.sin(rad)],
    #                     [np.sin(rad), np.cos(rad)]])
    #     return rot



    # def distances_calc(self, position):
    #     tmp_index = np.arange(position.shape[0])
    #     xx, yy = np.meshgrid(tmp_index, tmp_index)
    #     distances = np.linalg.norm(position[xx]-position[yy], axis=2)


    #     return distances

    # def angles_calc(self, position):
    #     tmp_index = np.arange(position.shape[0])
    #     xx, yy = np.meshgrid(tmp_index, tmp_index)
    #     distances = np.linalg.norm(np.arctan2(position[xx],position[yy]), axis=2)

    #     return distances
