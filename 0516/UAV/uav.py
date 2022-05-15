# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:32:17 2021

@author: Takumi
"""
# import environment
import numpy as np


class uav:
    def __init__(self, lat, lon, safe_area, faction, uav_id, tgt):
        self.pos = np.array([np.random.randint(safe_area[0], safe_area[1]), np.random.randint(lon/4, 3*lon/4)])
        self.safe_area = safe_area
        self.lat = lat
        self.lon = lon
        self.faction = faction
        if self.faction == "blue":
            # self.vec = np.array([-1,0])
            self.az = np.deg2rad(180)
            self.Izz = 50
            self.thrust = 1
        elif self.faction == "red":
            # self.vec = np.array([1,0])
            self.az = np.deg2rad(0)
            self.Izz = 50
            self.thrust = 1
        self.ref_aa = 0
        self.aa = 0
        self.com = np.array([0,0])
        self.id = uav_id
        self.mass = 1
        self.tgt = tgt
        self.vel = 2
        self.sensor_mode = False
        self.sensor_az = np.deg2rad(60)
        self.detect_launch = False
        self.detect_launch_ML = False
        self.inrange = False
        self.mrm_num = 2
        self.vel_limit = 15
        self.vec = self.az2vec(self.az)
        self.hitpoint = 1.0
        
        self.cool_down_limit = 200
        self.cool_down = self.cool_down_limit
        self.radar_range = 500
        self.mrm_range = 500
        
        # masu add 0930
        # 機体姿勢、高度用
        # 後で機体simモデルと整合させる
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.alt = 0
        
    def tgt_update(self, tgt):
        if self.tgt == 0:
            self.tgt = tgt
        elif self.detect_launch == False:
            if np.linalg.norm(self.pos - self.tgt.pos) > np.linalg.norm(self.pos - tgt.pos) and self.faction != "mrm" and tgt.hitpoint > 0:
                # print(self.faction, "TGT CHANGE")
                self.tgt = tgt
                
    def tgt_update_ML(self, tgt):
        if self.tgt == 0:
            self.tgt = tgt
        elif self.detect_launch == False:
            if self.faction != "mrm" and tgt.hitpoint > 0:
                # print(self.faction, "TGT CHANGE")
                self.tgt = tgt
                
    def tgt_inrange(self):
        # self.radar_range = self.mrm_range*(np.abs(self.aa)/np.pi-1/2)+10
        if self.hitpoint > 0 and self.tgt.hitpoint != 0 and np.linalg.norm(self.pos - self.tgt.pos) < self.radar_range and np.abs(self.ops_az()) < self.sensor_az:
            self.inrange = True
        else:
            self.inrange = False
        return self.inrange  
    
    def can_launch(self):
        self.cool_down = self.cool_down + 1
        launch = False
        if self.inrange and self.cool_down > self.cool_down_limit and self.mrm_num > 0:
            # print(self.faction, "MRM LAUNCH!!")
            self.cool_down = 0
            self.mrm_num = self.mrm_num - 1
            launch = True
        return launch
    
    def can_launch_ML(self):
        launch = False
        if self.inrange and self.cool_down == self.cool_down_limit and self.mrm_num > 0:
            # print(self.faction, "MRM LAUNCH!!")
            self.cool_down = 0
            self.mrm_num = self.mrm_num - 1
            launch = True
        return launch
    
    def MAWS(self, mrm):
        if self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint > 0 and self.hitpoint > 0:
            self.detect_launch = True
            # print(self.faction, "MRM DETECTED!!")
        elif self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint == 0:
            self.detect_launch = False
            
            
    def MAWS_ML(self, mrm):
        if self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint > 0 and self.hitpoint > 0:
            self.detect_launch_ML = True
            # print(self.faction, "MRM DETECTED!!")
        elif self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint == 0:
            self.detect_launch_ML = False

            
    def pos_update(self, sim_dt):
        
        self.vel = self.vel + self.thrust/self.mass*sim_dt
        if self.vel > self.vel_limit:
            self.vel = self.vel_limit
        # self.vec = self.vec + np.array([np.random.randint(0, 3), np.random.randint(-1, 1)])
        # # self.vec[1] = np.random.randint(-2, 2)
        self.pos = self.pos + self.vel*self.vec/np.linalg.norm(self.vec)*sim_dt

    
    def pos_update_ctrl(self, sim_dt):
        if self.tgt.hitpoint == 0 and self.faction == "mrm":
            
            if self.tgt.faction == "red":
                self.tgt.pos = np.array([-1000, self.lon/2])
            elif self.tgt.faction == "blue":
                self.tgt.pos = np.array([2000, self.lon/2])
            self.com = np.array([0,0])
        if self.hitpoint > 0:
            self.guidance_law(sim_dt)
            self.vec = self.vec_norm(self.vec + 0.01*self.Izz*self.com*self.thrust/self.mass*sim_dt)
    
            self.vel = self.vel + self.thrust/self.mass*sim_dt
            if self.vel > self.vel_limit:
                self.vel = self.vel_limit
    
            self.pos = self.pos + self.vel*self.vec*sim_dt

        else:
            if self.faction == "mrm":
                if self.parent.faction == "red":
                    self.pos = np.array([-self.parent.safe_area[1], self.lon/2])
                elif self.parent.faction == "blue":
                    self.pos = np.array([2*self.parent.safe_area[1], self.lon/2])
                    self.az = self.vec2az(self.vec)
        self.calc_aa()            
                
        
    def pos_update_ML(self, sim_dt, action):
        self.vec = self.vec_norm(self.vec + 0.01*self.Izz*action*self.thrust/self.mass*sim_dt)

        self.vel = self.vel + self.thrust/self.mass*sim_dt
        if self.vel > self.vel_limit:
            self.vel = self.vel_limit
        
        self.pos = self.pos + self.vel*self.vec*sim_dt
        self.az = self.vec2az(self.vec)
        self.calc_aa()
        
    def guidance_law(self, sim_dt):
        if self.detect_launch:
            self.com = -1*self.vec_norm((self.tgt.pos - self.pos))  
        else:
            self.com = self.vec_norm((self.tgt.pos - self.pos)) -2*self.az2vec(self.ref_aa - self.aa)
            
        
    def az2vec(self,az):
        vec = self.vel*np.array([int(np.cos(np.deg2rad(az))),int(np.sin(np.deg2rad(az)))])
        
        return vec
    
    def vec_norm(self, vec):
        az = self.vec2az(vec)
        vec = np.array([np.cos(az), np.sin(az)])
        
        return vec
    
    def vec2az(self,vec):
        az = np.arctan2(vec[1],vec[0])

        return az
    
    def ops_az(self):
        tgt_az = self.vec2az(self.tgt.pos - self.pos) - self.vec2az(self.vec)
        if tgt_az > np.pi:
            tgt_az = tgt_az - 2*np.pi
        elif tgt_az < -np.pi:
            tgt_az = tgt_az + 2*np.pi
                

        return tgt_az
    
    def calc_aa(self):
        self.aa = -self.vec2az(self.pos - self.tgt.pos) + np.pi - self.vec2az(self.tgt.vec)
        if self.aa > np.pi:
            self.aa = self.aa - 2*np.pi
        elif self.aa < -np.pi:
            self.aa = self.aa + 2*np.pi
#    def detect_enem(self, )


    
    