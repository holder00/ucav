# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:35:21 2021

@author: Takumi
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class uav_3d:
    def __init__(self, lat, lon, safe_area, faction, uav_id, tgt):
        self.pos = np.array([np.random.randint(safe_area[0], safe_area[1]), np.random.randint(0, lon), np.random.randint(0, 7500)])
        self.safe_area = safe_area
        self.lon = lon
        self.faction = faction
        self.mass = 10000 #kg
        self.g = 9.80665 #m/ss
        self.Cd2 = 0.3
        self.Cd0 = 0.1
        self.Cl1 = 0.1
        self.Cl0 = 2
        
        if self.faction == "blue":
            # self.vec = np.array([-1,0])
            self.az = np.deg2rad(180)
            self.Izz = 50
            self.thrust = 1
        elif self.faction == "red":
            # self.vec = np.array([1,0])
            self.az = np.deg2rad(0)
            self.Izz = 25*2
            self.thrust = 1
        self.ref_aa = 0
        self.aa = 0
        self.com = np.array([0,0])
        self.id = uav_id
        self.mass = 1
        self.tgt = tgt
        self.vel = 2
        self.sensor_mode = False
        self.sensor_az = np.deg2rad(30)
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
    
    def state(self,state0,t,T,alp,phi):
        x,y,z,V,psi,gam = state0[0],state0[1],state0[2],state0[3],state0[4],state0[5]
        D = (self.Cd2*alp*alp+self.Cd0)*V*V
        L = (self.Cl1*alp+self.Cl0)*V*V
        alp = gam
        v = [0]*6
        v[0] = V*np.cos(gam)*np.sin(psi)
        v[1] = V*np.cos(gam)*np.cos(psi)
        v[2] = V*np.sin(gam)
        v[3] = T/self.mass*np.cos(alp) -D/self.mass -self.g*np.sin(gam)
        v[4] = L/(self.mass*V*np.cos(gam))*np.sin(phi)
        v[5] = (T*np.sin(alp) + L)/(self.mass*V)*np.cos(phi)-self.g/V*np.cos(gam)
        return v
    
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
    #    def detect_enem(self, )        
    # x = 0
    # y = 0
    # z = 1000
    
    # sim_dt = 0.1
    



    

    
    def pos_update_ctrl(self, sim_dt):
        if self.tgt.hitpoint == 0 and self.faction == "mrm":
            
            if self.tgt.faction == "red":
                self.tgt.pos = np.array([-1000, self.lon/2, 0])
            elif self.tgt.faction == "blue":
                self.tgt.pos = np.array([2000, self.lon/2, 0])
            self.com = np.array([0,0,0])
        if self.hitpoint > 0:
            self.guidance_law(sim_dt)
            self.vec = self.vec_norm(self.vec + 0.01*self.Izz*self.com*self.thrust/self.mass*sim_dt)
    
            self.vel = self.vel + self.thrust/self.mass*sim_dt
            if self.vel > self.vel_limit:
                self.vel = self.vel_limit
            t = np.arange(0, sim_dt, sim_dt/100)
                #state
            self.V = 250 #m/s
            self.gam = np.deg2rad(0)
            self.psi = np.deg2rad(0)
            
            self.T = 5000
            self.alp = np.deg2rad(0)
            self.phi = np.deg2rad(0)
            state0 = [self.pos[0],self.pos[1],self.pos[2], self.V, self.psi, self.gam]
            v = odeint(self.state, state0, t, args=(self.T, self.alp, self.phi))
            # self.pos[0],self.pos[1],self.pos[2], self.V, self.psi, self.gam = v[-1,:]
            # self.pos = self.pos + self.vel*self.vec*sim_dt
            if self.faction == "blue":
                print(v[-1,0:3])
            self.pos = v[-1,0:3]

        else:
            if self.faction == "mrm":
                if self.parent.faction == "red":
                    self.pos = np.array([-self.parent.safe_area[1], self.lon/2,0])
                elif self.parent.faction == "blue":
                    self.pos = np.array([2*self.parent.safe_area[1], self.lon/2,0])
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

