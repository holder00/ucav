# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:11:44 2021

@author: Takumi
"""
import mmap
import time
from datetime import datetime
from struct import *
import ctypes
from multiprocessing import Process
import subprocess

Kernel32 = ctypes.windll.Kernel32
mutex = Kernel32.CreateMutexA(0,0,"Global/UAV_MUTEX")


import gym
from gym import spaces
import numpy as np
import math
from settings.initial_settings import *  # Import settings  # For training
# from settings.test_settings import *  # Import settings  # For testing
from settings.reset_conditions import reset_conditions
from modules.resets import reset_red, reset_blue, reset_block
from modules.observations import get_observation
from modules.rewards import get_reward
import cv2
from status.get_obs_3d import get_obs
from status.reward_calc import reward_calc
from status.init_space import init_space
from UAV.uav_3d import uav_3d
from weapon.missile_3d import missile_3d
from utility.terminate_uavsimproc import teminate_proc
from SENSOR.SENSOR_MAIN import sensor



class MyEnv():
    def __init__(self, config={}):
        super(MyEnv, self).__init__()
        np.set_printoptions(precision=2,suppress=True)
        self.WINDOW_SIZE_lat = 100*1000 #画面サイズの決定
        self.WINDOW_SIZE_lon = 100*1000 #画面サイズの決定
        self.WINDOW_SIZE_alt = 12*1000

        self.GOAL_RANGE = 1500 #ゴールの範囲設定
        self.timer = 0
        self.sim_dt = 1
        self.sim_freq = 1
        self.time_limit = 600

        self.blue_num = 2

        self.blue_side = int(self.WINDOW_SIZE_lat/4)
        self.blue_safe_area = [self.WINDOW_SIZE_lat-self.blue_side,self.WINDOW_SIZE_lat]

        self.red_num = 2
        self.red_side = int(self.WINDOW_SIZE_lat/4)
        self.red_safe_area = [0,self.red_side]
        self.ng_range = 250
        self.ng_area_lat = [self.ng_range, self.WINDOW_SIZE_lat-self.ng_range]
        self.ng_area_lon = [self.ng_range, self.WINDOW_SIZE_lon-self.ng_range]
        self.ng_area_alt = [self.ng_range, self.WINDOW_SIZE_alt-self.ng_range]


        # self.action_dict_c = {}
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

        #exeのパス
        if not self.init_flag:
            #UCAV.exeが起動している場合、プロセスキルする。
            teminate_proc.UAVsimprockill(proc_name="UCAV.exe")
            subprocess.Popen(r"./UCAV.exe")
            #exe起動待ち
            time.sleep(3)

            #共有メモリに初期値設定
            self.SetPyToSimData(self.blue,self.red,1,0)

        return obs

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}

        self.timer = self.timer + self.sim_dt

#python ⇒ uavsim
        # if self.timer > 0:
        self.sim_cnt = self.sim_cnt + 1
        self.SetPyToSimData(self.blue,self.red,1,self.sim_cnt)


        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+ " @ "+ "============" + "UAVsim calc start" + "=============")
        self.perf_time = time.perf_counter()
        while not self.sim_comp:
            self.GetSimToPyData(self.blue,self.red)
            if self.sim_comp:
                self.sim_comp = False
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + " @ "+ "============" + "UAVsim calc complete"+ "============" + str(time.perf_counter() - self.perf_time))
                print(self.blue[0].pos)
                break
        if self.timer >= self.time_limit:
            dones["__all__"] = True
            self.SetPyToSimData(self.blue,self.red,1,-1)
        else:
            dones["__all__"] = False


        # info
        for i in range(self.blue_num):
            if self.blue[i].hitpoint == 0:
                infos["blue_" + str(i)] = {}

        return obs, rewards, dones, infos

    def limit_calc(self, pos_c, limit):
        pos = pos_c
        for i in range(len(pos_c)):
            if pos_c[i] <= 0:
                pos[i] = 0

            if pos_c[i] >= limit[i]:
                pos[i] = limit[i]

        return pos
    #------masubuchi add------
    def SetPyToSimData(self,blue,red,flag,stepcnt):
        shmem2 = mmap.mmap(0,232,"PYTHON_SIM_MEM")
        BlueData = [[0.0] * 8,[0.0] * 8]
        RedData = [[0.0] * 8,[0.0] * 8]

        #初期化時
        for i in range(len(blue)):
            BlueData[i] = [blue[i].pos[0], blue[i].pos[1], blue[i].pos[2], blue[i].V*0+340, np.rad2deg(blue[i].gam)*0, np.rad2deg(blue[i].psi), blue[i].fire, blue[i].fire]
        for i in range(len(red)):
            RedData[i] = [red[i].pos[0], red[i].pos[1], red[i].pos[2], red[i].V*0+250, np.rad2deg(red[i].gam)*0, np.rad2deg(red[i].psi),red[i].fire,blue[i].fire]

        #Step更新時
        if stepcnt > 0:
            for i in range(len(blue)):
                BlueData[i][3] = blue[i].V_ref
                BlueData[i][4] = np.rad2deg(blue[i].gam_ref)
                BlueData[i][5] = np.rad2deg(blue[i].psi_ref)
                if blue[i].fire:
                    BlueData[i][6] = 1
                    BlueData[i][7] = 1
                else:
                    BlueData[i][6] = 1
                    BlueData[i][7] = 1

            for i in range(len(red)):
                RedData[i][3] = red[i].V_ref*0+250
                RedData[i][4] = np.rad2deg(red[i].gam_ref)
                RedData[i][5] = np.rad2deg(red[i].psi_ref)
                if red[i].fire:
                    RedData[i][6] = 1 #0:未発射＝False
                    RedData[i][7] = 1 #ダミー
                else:
                    RedData[i][6] = 1
                    RedData[i][7] = 1

        #blue1-2,red1-2
        #double lat,lon,alt,dVc,dGamC,dPsi
        #int Python指令 1:初期化中 2:初期化完了 3:Sim終了
        #int stepカウント(初期化時：0,step実行毎にカウント 1up sim側は値変化で動作させる)
        # print(BlueData)
        # print(BlueData[0][7])
        OutValue = pack('ddddddiiddddddiiddddddiiddddddiiii',
                        BlueData[0][0], BlueData[0][1], BlueData[0][2], BlueData[0][3], BlueData[0][4], BlueData[0][5], BlueData[0][6], BlueData[0][7],
                        BlueData[1][0], BlueData[1][1], BlueData[1][2], BlueData[1][3], BlueData[1][4], BlueData[1][5], BlueData[1][6], BlueData[1][7],
                        RedData[0][0], RedData[0][1], RedData[0][2], RedData[0][3], RedData[0][4], RedData[0][5], RedData[0][6],RedData[0][7],
                        RedData[1][0], RedData[1][1], RedData[1][2], RedData[1][3], RedData[1][4], RedData[1][5], RedData[1][6],RedData[1][7],
                        flag,stepcnt)    
        # print("init","Bule[0] lat",BlueData[0][0],"lon",BlueData[0][1],"alt",BlueData[0][2],"Vc",BlueData[0][3],"Gamc",BlueData[0][4],"Psic",BlueData[0][5])
        # print("    ","Bule[1] lat",BlueData[1][0],"lon",BlueData[1][1],"alt",BlueData[1][2],"Vc",BlueData[1][3],"Gamc",BlueData[1][4],"Psic",BlueData[1][5])
        # print("init","Red[0] lat",RedData[0][0],"lon",RedData[0][1],"alt",RedData[0][2],"Vc",RedData[0][3],"Gamc",RedData[0][4],"Psic",RedData[0][5])
        # print("    ","Red[1] lat",RedData[1][0],"lon",RedData[1][1],"alt",RedData[1][2],"Vc",RedData[1][3],"Gamc",RedData[1][4],"Psic",RedData[1][5])

        shmem2.write(OutValue)

    def GetSimToPyData(self,blue,red):
        # print("GetSIM-->")
        shmem = mmap.mmap(0,296,"UAV_SIM_MEM")
        #Uav[0] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #Uav[1] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #Uav[2] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #Uav[3] dVc,dGamC,dPsiC,dRoll,dPitch,dYaw
        #int Python指令
        InValue = unpack('ddddddddddddddddddddddddddddddddddddii',shmem)
        #print(InValue)
        # print("Uav[0]:Vc",InValue[0],"VGam",InValue[1],"Psi",InValue[2],"Roll",InValue[3],"Pitch",InValue[4],"Yaw",InValue[5],"N",InValue[6],"E",InValue[7],"H",InValue[8])
        # print("Uav[1]:Vc",InValue[9],"VGam",InValue[10],"Psi",InValue[11],"Roll",InValue[12],"Pitch",InValue[13],"Yaw",InValue[14],"N",InValue[15],"E",InValue[16],"H",InValue[17])
        # print("Uav[2]:Vc",InValue[18],"VGam",InValue[19],"Psi",InValue[20],"Roll",InValue[21],"Pitch",InValue[22],"Yaw",InValue[23],"N",InValue[24],"E",InValue[25],"H",InValue[26])
        # print("Uav[3]:Vc",InValue[27],"VGam",InValue[28],"Psi",InValue[29],"Roll",InValue[30],"Pitch",InValue[31],"Yaw",InValue[32],"N",InValue[33],"E",InValue[34],"H",InValue[35])

        data_length = int((len(InValue)-2)/4)
        # print(self.timer,InValue[37],self.sim_time,self.sim_comp,datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        if (InValue[37] - self.sim_time) != 0:
            self.sim_comp = True
        self.sim_time = InValue[37]

        for i in range(len(blue)):
            self.blue[i].pos[0] = InValue[i*data_length + 6]
            self.blue[i].pos[1] = InValue[i*data_length + 7]
            self.blue[i].pos[2] = InValue[i*data_length + 8]
            self.blue[i].V = InValue[i*data_length + 0]
            self.blue[i].gam = np.deg2rad(InValue[i*data_length + 1])
            self.blue[i].psi = np.deg2rad(InValue[i*data_length + 2])
            #rad→°
            self.blue[i].phi = InValue[i*data_length + 3]
            self.blue[i].pitch = InValue[i*data_length + 4]
            self.blue[i].yaw = InValue[i*data_length + 5]

        for i in range(len(red)):
            self.red[i].pos[0] = InValue[i*data_length + 24]
            self.red[i].pos[1] = InValue[i*data_length + 25]
            self.red[i].pos[2] = InValue[i*data_length + 26]
            self.red[i].V = InValue[i*data_length + 18]
            self.red[i].gam = np.deg2rad(InValue[i*data_length + 19])
            self.red[i].psi = np.deg2rad(InValue[i*data_length + 20])
            #rad→°
            self.red[i].phi = InValue[i*data_length + 21]
            self.red[i].pitch = InValue[i*data_length + 22]
            self.red[i].yaw = InValue[i*data_length + 23]

#------masubuchi add------
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
