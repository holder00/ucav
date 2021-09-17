# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:24:56 2021

@author: Takumi
"""

import gym
import numpy as np
import cv2
from reward_calc import reward_calc
from uav import uav
from missile import missile



class MyEnv(gym.Env):
    def __init__(self):
        self.WINDOW_SIZE_lat = 1000 #画面サイズの決定
        self.WINDOW_SIZE_lon = 500 #画面サイズの決定

        self.GOAL_RANGE = 5 #ゴールの範囲設定
        self.timer = 0
        self.sim_dt = 0.1
        self.time_limit = 150

        self.blue_num = 1
        
        self.blue_side = int(self.WINDOW_SIZE_lat/5)
        self.blue_safe_area = [self.WINDOW_SIZE_lat-self.blue_side,self.WINDOW_SIZE_lat]
        
        self.red_num = self.blue_num
        self.red_side = int(self.WINDOW_SIZE_lat/5)
        self.red_safe_area = [0,self.red_side]
        self.ng_range = 200
        self.ng_area_lat = [self.ng_range, self.WINDOW_SIZE_lat-self.ng_range]
        self.ng_area_lon = [self.ng_range, self.WINDOW_SIZE_lon-self.ng_range]
        # アクション数定義
        # ACTION_MAP = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
        # ACTION_MAP = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
        # ACTION_MAP = np.arange(self.red_num)
        ac_tgt_id = np.arange(self.red_num)
        ac_fire = np.arange(2)
        ac_evade = np.arange(2)
        # ac_aa = np.arange(8)
        ACTION_MAP = np.array(np.meshgrid(ac_tgt_id, ac_fire, ac_evade)).T.reshape(-1,3)
        # self.ACTION_MAP = np.tile(ACTION_MAP,(self.blue_num,1,1))
        self.ACTION_MAP = np.tile(ACTION_MAP,(self.blue_num,1,1))
        ACTION_NUM = ACTION_MAP.shape[1]
        # self.action_space = gym.spaces.MultiDiscrete(list((self.red_num,2,2,8)*self.blue_num))
        self.action_space = gym.spaces.MultiDiscrete(list((self.red_num,2,2)*self.blue_num))

        # 状態の範囲を定義
        self.ofs_num = self.red_num + self.blue_num
        LOW = np.tile(np.append([0,0,0,0,-1,-1], [0]*(self.blue_num+self.red_num)),(self.ofs_num))   #被射撃、HP、距離
        HIGH = np.tile(np.append([1,2,1,1,1,1],[1]*(self.blue_num+self.red_num)),(self.ofs_num))
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH)
        
        self.reset()

    def reset(self):
        self.blue = [uav(self.WINDOW_SIZE_lat-self.blue_side, self.WINDOW_SIZE_lon, self.blue_safe_area,"blue",0,1)]*self.blue_num
        self.red = [uav(self.red_side, self.WINDOW_SIZE_lon, self.red_safe_area,"red",0,1)]*self.red_num
        self.before_distance = np.zeros(self.blue_num)
        self.mrm = [0]*(self.blue_num*self.blue[0].mrm_num + self.red_num*self.red[0].mrm_num)
        self.mrm_num = 0
        observation = np.zeros(self.observation_space.shape)
        ops = np.zeros(self.observation_space.shape)
        
        obs_index_distance = (self.observation_space.shape[0]-(self.red_num+self.blue_num))*0+5
        position = np.zeros([self.blue_num+self.red_num,2])
        self.reward_k = 0
        self.reward_d = 0
        self.reward_missile_lost = 0
        self.reward_fire = 0
        self.reward_win = 0
        self.reward_total = 0
        
        self.index_id = int(self.observation_space.shape[0]/(self.blue_num+self.red_num))
        for i in range(self.blue_num):
            self.blue[i] = uav(self.WINDOW_SIZE_lat-self.blue_side, self.WINDOW_SIZE_lon, self.blue_safe_area,"blue",i,0)
            
        for i in range(self.red_num):
            self.red[i] = uav(self.red_side, self.WINDOW_SIZE_lon, self.red_safe_area,"red",i,0)
            
        for i in range(self.blue_num):
            self.blue[i].tgt_update(self.red[i])
            
        for i in range(self.red_num):
            self.red[i].tgt_update(self.blue[i])
                
        for i in range(self.blue_num):
            # 状態の作成
            index = i*self.index_id
            # position[i] = self.blue[i].pos
            # observation[index:index+2] = position[i]
            observation[index] = self.blue[i].hitpoint
            
        # print(observation)    
        for i in range(self.red_num):
            # 状態の作成
            index = (i+self.blue_num)*self.index_id
            # position[i+self.blue_num] = self.red[i].pos
            # observation[index:index+2] = position[i+self.blue_num]
            observation[index] = self.red[i].hitpoint
            # observation[i+self.blue_num*2] = np.array([self.blue[i].hitpoint, self.red[i].hitpoint])
            self.before_distance[i] = np.linalg.norm(self.blue[i].pos - self.red[i].pos)
            
        distances = self.distances_calc(position)
        angles = self.angles_calc(position)

        for i in range(self.blue_num):
            index = i*self.index_id
            print(observation[index+5:index+self.index_id])
            print("0000000000")
            print(distances[i,:])
            # observation[index+5:index+self.index_id] = np.append(distances[i,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon),angles[i,:])
            observation[index+6:index+self.index_id] = distances[i,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon)
        for i in range(self.red_num):
            index = (i+self.blue_num)*self.index_id
            # observation[index+5:index+self.index_id] = np.append(distances[i+self.blue_num,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon),angles[i+self.blue_num,:])
            observation[index+6:index+self.index_id] = distances[i+self.blue_num,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon)
        # observation[0:self.blue_num+self.red_num,obs_index_distance:obs_index_distance+(self.red_num+self.blue_num)] = distances[0:self.blue_num+self.red_num,0:self.blue_num+self.red_num]
        self.temp = observation   
            
        # observation[self.ofs_num-2] = np.array(self.ng_area_lat)
        # observation[self.ofs_num-1] = np.array(self.ng_area_lon)
        # print(observation)
        return observation

    def step(self, action_index):
        distance_temp = [0]*self.blue_num
        reward_temp = [0]*self.blue_num
        reward_fw_blue = [0]*self.blue_num
        reward_fw_red = [0]*self.red_num
        reward_fw = [0]*self.blue_num
        reward_k_temp = 0
        reward_d_temp = 0
        self.reward_fire = 0
        self.reward_missile_lost = 0
        # reward_ng = [0]*self.blue_num
        observation = np.zeros(self.observation_space.shape)
        ops = np.zeros(self.observation_space.shape)
        obs_index_distance = (self.observation_space.shape[0]-(self.red_num+self.blue_num))*0+5
        position = np.zeros([self.blue_num+self.red_num,2])

        is_touch = False
        is_red_reach = False
        is_blue_reach = False
        is_fin = False
        done = False
        reward = 0
        reward_detect = 0
        reward_inrange = 0
        penalty = 0
        
        for i in range(self.blue_num):
            self.blue[i].pos_update_ctrl(self.sim_dt)
            self.blue[i].pos = self.limit_calc(self.blue[i].pos, [self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon])
            
        for i in range(self.red_num):
            self.red[i].pos_update_ctrl(self.sim_dt)
            self.red[i].pos = self.limit_calc(self.red[i].pos, [self.WINDOW_SIZE_lat, self.WINDOW_SIZE_lon])
        
        for i in range(self.mrm_num):
            self.mrm[i].update_status(self.sim_dt)
            if self.mrm[i].missile_tgt_lost:
                self.reward_missile_lost = self.reward_missile_lost + 1
                self.mrm[i].missile_tgt_lost = False
                # print(self.reward_missile_lost)
                
            if np.linalg.norm(self.mrm[i].tgt.pos - self.mrm[i].pos) < self.mrm[i].destruction_range:
                self.mrm[i].tgt.hitpoint = 0
                self.mrm[i].hitpoint = 0
                if self.mrm[i].tgt.faction == "red":
                    self.reward_k = self.reward_k + 1
                    reward_k_temp = reward_k_temp + 1
                elif self.mrm[i].tgt.faction == "blue":
                    self.reward_d = self.reward_d + 1
                    reward_d_temp = reward_d_temp + 1
        
        for i in range(self.blue_num):
            #発射判定
            if self.blue[i].tgt_inrange():
                if self.blue[i].can_launch_ML():
                    self.mrm[self.mrm_num] = missile(self.blue[i])
                    self.reward_fire = self.reward_fire + 1
                    self.mrm_num = self.mrm_num + 1
                
        for i in range(self.red_num):
            # print(self.red[i].detect_launch)
            #発射判定 uavクラスのステータス更新部分に移行予定
            if self.red[i].tgt_inrange():
                if self.red[i].can_launch():
                    self.mrm[self.mrm_num] = missile(self.red[i])
                    self.mrm_num = self.mrm_num + 1
                    
        
        for i in range(self.mrm_num):
            for j in range(self.blue_num):
                self.blue[j].MAWS_ML(self.mrm[i])
            for j in range(self.red_num):
                self.red[j].MAWS(self.mrm[i])
        
        for i in range(self.blue_num):
            self.blue[i].tgt_update_ML(self.red[action_index[0+i*3]])
            if action_index[1+i*3] == 0:
                self.blue[i].detect_launch = True
            else:
                self.blue[i].detect_launch = False
            if action_index[2+i*3] == 0:
                self.blue[i].cool_down = 0
            else:
                self.blue[i].cool_down = self.blue[i].cool_down_limit   
            # self.blue[i].ref_aa = np.deg2rad(action_index[3+i*4]*45)
            
                
        for i in range(self.red_num):
            for j in range(self.blue_num):
                self.red[i].tgt_update(self.blue[j])
        
        for i in range(self.blue_num):
            # 状態の作成
            index = i*self.index_id
            position[i] = self.blue[i].pos
            # observation[index:index+2] = position[i]
            observation[index] = self.blue[i].hitpoint        
            observation[index+1] = self.blue[i].mrm_num
            if self.blue[i].inrange:
                observation[index+2] = 0
            else:
                observation[index+2] = 1
            if self.blue[i].detect_launch_ML:
                observation[index+3] = 0
            else:
                observation[index+3] = 1
            # observation[index+4] = self.blue[i].aa
            observation[index+4] = np.cos(self.blue[i].ops_az())
            observation[index+5] = np.sin(self.blue[i].ops_az())
                
        for i in range(self.red_num):
            # 状態の作成
            index = (i+self.blue_num)*self.index_id
            position[i+self.blue_num] = self.red[i].pos
            # observation[index:index+2] = position[i+self.blue_num]
            observation[index] = self.red[i].hitpoint
            observation[index+1] = self.red[i].mrm_num
            if self.red[i].inrange:
                observation[index+2] = 0
            else:
                observation[index+2] = 1
            if self.red[i].detect_launch:
                observation[index+3] = 0
            else:
                observation[index+3] = 1
            # observation[index+4] = self.red[i].aa
            observation[index+4] = np.cos(self.red[i].ops_az())
            observation[index+5] = np.sin(self.red[i].ops_az())
        distances = self.distances_calc(position)
        angles = self.angles_calc(position)
        
        for i in range(self.blue_num):
            index = i*self.index_id
            # observation[index+5:index+self.index_id] = np.append(distances[i,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon),angles[i,:])
            observation[index+6:index+self.index_id] = distances[i,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon)
        
        for i in range(self.red_num):
            index = (i+self.blue_num)*self.index_id
            observation[index+6:index+self.index_id] = distances[i+self.blue_num,:]/np.sqrt(self.WINDOW_SIZE_lat*self.WINDOW_SIZE_lat+self.WINDOW_SIZE_lon*self.WINDOW_SIZE_lon)
        
        for i in range(self.blue_num):
            if self.blue[i].inrange:
                reward_inrange = reward_inrange + 1
        for i in range(self.red_num):
            if self.red[i].inrange:
                reward_inrange = reward_inrange - 2
                
        tgt_id = np.zeros(4)
        for i in range(self.blue_num):
            # tgt_id[i] = self.ACTION_MAP[i,action_index[i],0]
            reward_tgt = np.count_nonzero(tgt_id)
        
        for i in range(self.blue_num):
            # 終了判定
            if distance_temp[i] < self.GOAL_RANGE:
                # is_touch = True
                is_touch = False
            if self.red[i].pos[0] > self.blue_safe_area[0]:
                is_red_reach = True
            # if self.blue[i].pos[0] < self.red_safe_area[1]:
                # is_blue_reach = True
                # is_reach = False

            self.before_distance[i] = distance_temp[i]
        
        if self.reward_k == self.red_num or self.reward_d == self.blue_num:
            for i in range(self.mrm_num):
                if not self.mrm[i].hitpoint == 0:
                    is_fin = False
                else:
                    is_fin = True
            if self.reward_d == self.blue_num:
                is_fin = True
        if is_touch or is_red_reach or is_blue_reach or is_fin or self.timer >= self.time_limit:
            done = True
            self.timer = 0
            if (not is_red_reach and not self.reward_d == self.blue_num) or is_blue_reach:
                self.reward_win = 1
            elif is_red_reach or self.reward_d == self.blue_num or not(self.reward_k == self.red_num) or self.timer >= self.time_limit:
                self.reward_win = -4
        self.center_line = (np.min(reward_fw_blue) + np.max(reward_fw_red))/2

        self.blue_line = np.average(reward_fw_blue)
        reward_blue_line = (self.WINDOW_SIZE_lat- self.blue_line)/self.WINDOW_SIZE_lat
        reward_f = (self.WINDOW_SIZE_lat/2 - self.center_line)/(self.WINDOW_SIZE_lat/2)
        reward_reduce = [0]*self.red_num
            
        reward_ng = reward_calc.reward_ng(self.blue, self.ng_area_lat, self.ng_area_lon, self.ng_range)

        #報酬は事象と状況によって区別すべき、事象は発生した一瞬に対して報酬が変化、状況は継続的に報酬が変化
        #状況は今後の行動で好転させることができるが、事象は今後の行動で好転させることができない
        #ex)撃墜された”事象”＝発生した瞬間マイナス、しかし今後の行動で好転させることができない　前線が押されている”状況”＝現在はマイナス継続するとマイナス、しかし今後の行動で打開できる

        reward_temp = 0.0001*reward_inrange -reward_tgt*0 - 0*self.reward_fire -0.0001*np.sum(reward_ng) + 0*reward_k_temp*self.reward_k + self.reward_win
        reward = reward_temp
        
        self.reward_total = self.reward_total + reward 
        self.timer = self.timer + self.sim_dt 
        self.reward_num = reward
           
        return observation, reward, done, {}
    
    
    def render(self):
        # opencvで描画処理
        img = np.zeros((self.WINDOW_SIZE_lon, self.WINDOW_SIZE_lat, 3)) #画面初期化
        for i in range(self.blue_num):
            self.render_craft(img,self.blue[i])
            self.render_craft(img,self.red[i])
            self.render_radar(img,self.blue[i])
            self.tgt_lines(img,self.blue[i])
            # self.tgt_lines(img,self.red[i])
        for i in range(self.mrm_num):
            self.render_missile(img, self.mrm[i])
            # self.tgt_lines(img,self.mrm[i])

        # cv2.rectangle(img,(int(self.center_line),0),(int(self.center_line), self.WINDOW_SIZE_lon),(255,0,255),3)
        # cv2.rectangle(img,(int(self.blue_line),0),(int(self.blue_line), self.WINDOW_SIZE_lon),(255,255,255),3)
        cv2.rectangle(img,(0,0),(self.blue_side, self.WINDOW_SIZE_lon),(0,0,255),3)
        
        cv2.rectangle(img,(self.WINDOW_SIZE_lat,0),(self.WINDOW_SIZE_lat-self.red_side, self.WINDOW_SIZE_lon),(255,0,0),3)
        
        cv2.putText(img, format(self.timer, '.1f'), (self.WINDOW_SIZE_lat-500, self.WINDOW_SIZE_lon-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(img, format(self.reward_num, '.3f'), (self.WINDOW_SIZE_lat-500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.putText(img, format(self.reward_total, '.3f'), (self.WINDOW_SIZE_lat-700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def render_craft(self, img, temp):
        if temp.faction == "red":
            # cv2.putText(img, format(temp.id, 'd'), (int(temp.pos[0]), int(temp.pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            color_num = (0,0,255)
            # if temp.detect_launch:
            #     color_num = (255,255,255)
        elif temp.faction =="blue":
            # cv2.putText(img, format(temp.id, 'd'), (int(temp.pos[0]), int(temp.pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            color_num = (255,0,0)
            # if temp.detect_launch:
            #     color_num = (255,255,255)
        elif temp.faction == "mrm":
            color_num = (0,255,0)
        cv2.polylines(img,[self.craft(temp.pos,temp.vec)],True,color_num)

        cv2.circle(img,  (int(temp.pos[0]), int(temp.pos[1])), 5, color_num, thickness=-1)
        
    def render_missile(self, img, temp):
        if temp.faction == "red":
            color_num = (0,0,255)
        elif temp.faction =="blue":
            color_num = (255,0,0)
        elif temp.faction == "mrm":
            color_num = (0,255,0)
        cv2.polylines(img,[self.craft_missile(temp.pos,temp.vec)],True,color_num)

        cv2.circle(img,  (int(temp.pos[0]), int(temp.pos[1])), 1, color_num, thickness=-1)
        
    def tgt_lines(self, img, temp):
        if temp.hitpoint > 0:
            if temp.faction == "red":
                color_num = (0,0,255)
            elif temp.faction =="blue":
                color_num = (255,0,0)
            elif temp.faction == "mrm":
                color_num = (0,255,0)
                
            cv2.line(img,(int(temp.pos[0]), int(temp.pos[1])),(int(temp.tgt.pos[0]), int(temp.tgt.pos[1])),color_num,1)
        
    def render_radar(self, img, temp):
        # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_8, shift=0)
        if temp.inrange:
            color_num = (255,0,0)
        else:
            color_num = (255,255,255)
        cv2.ellipse(img, (int(temp.pos[0]), int(temp.pos[1])), (int(temp.radar_range+1), int(temp.radar_range+1)), -90-np.rad2deg(np.arctan2(temp.vec[0], temp.vec[1])), 180-np.rad2deg(temp.sensor_az), 180+np.rad2deg(temp.sensor_az), color_num)

   
    def craft_missile(self, pos, vec):
        pos = np.array(pos)
        t = -np.arctan2(vec[0], vec[1])
        rot = self.get_rotation_matrix(t)

        top = pos + np.dot(rot,np.array([0, 5]))
        btm_l = pos + np.dot(rot,np.array([-1, -1]))
        btm_r = pos + np.dot(rot,np.array([1, -1]))

        
        pts = np.array([top,btm_l,btm_r], np.int32)
        pts = pts.reshape((-1,1,2))
        
        return pts
     
    def craft(self, pos, vec):
        pos = np.array(pos)
        t = -np.arctan2(vec[0], vec[1])
        rot = self.get_rotation_matrix(t)

        top = pos + np.dot(rot,np.array([0, 15]))
        btm_l = pos + np.dot(rot,np.array([-5, -5]))
        btm_r = pos + np.dot(rot,np.array([5, -5]))

        
        pts = np.array([top,btm_l,btm_r], np.int32)
        pts = pts.reshape((-1,1,2))
        
        return pts
    
    def get_rotation_matrix(self, rad):
        rot = np.array([[np.cos(rad), -np.sin(rad)],
                        [np.sin(rad), np.cos(rad)]])
        return rot
    
    def limit_calc(self, pos_c, limit):
        pos = pos_c
        if pos_c[0] <= 0:
            pos[0] = 0
        
        if pos_c[0] >= limit[0]:
            pos[0] = limit[0]
            
        if pos_c[1] <= 0:
            pos[1] = 0
        
        if pos_c[1] >= limit[1]:
            pos[1] = limit[1]
        
        return pos
    
    def distances_calc(self, position):
        tmp_index = np.arange(position.shape[0])
        xx, yy = np.meshgrid(tmp_index, tmp_index)
        distances = np.linalg.norm(position[xx]-position[yy], axis=2)
        
        
        return distances
    
    def angles_calc(self, position):
        tmp_index = np.arange(position.shape[0])
        xx, yy = np.meshgrid(tmp_index, tmp_index)
        distances = np.linalg.norm(np.arctan2(position[xx],position[yy]), axis=2)
        
        return distances
        