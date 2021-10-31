# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:35:21 2021

@author: Takumi
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from get_rotation_matrix_3d import get_rotation_matrix_3d_phi
import copy

class uav_3d:
    def __init__(self, lat, lon, safe_area, faction, uav_id, tgt):
        self.pos = np.array([np.random.randint(safe_area[0], safe_area[1]), np.random.randint(0, lon),
                             np.random.randint(5*1000, 5.001*1000)])
        self.safe_area = safe_area
        self.lon = lon
        self.faction = faction
        self.mass = 4000 #kg
        self.g = 9.80665 #m/ss
        self.Cd2 = 3.0132
        self.Cd1 = 0.0517
        self.Cd0 = 0.0249
        self.Cl1 = 3.2417
        self.Cl0 = -0.0184
        self.Cb = -0.8594
        self.T = 16000
        self.S = 18
        
        
        self.V = 250 #m/s
        self.gam = np.deg2rad(0)

        if self.faction == "blue":
            # self.vec = np.array([-1,0])
            self.az = np.deg2rad(180)
            self.psi = np.deg2rad(-90)
            self.phi = np.deg2rad(0)
            self.alp = np.deg2rad(0)
            self.Izz = 50
            self.thrust = 1
            self.mrm_num = 1
        elif self.faction == "red":
            # self.vec = np.array([1,0])
            self.az = np.deg2rad(0)
            self.psi = np.deg2rad(90)
            self.phi = np.deg2rad(0)
            self.alp = np.deg2rad(1)
            self.Izz = 25*2
            self.thrust = 1
            self.mrm_num = 0
            
        self.V_p = self.V
        self.gam_p = self.gam
        self.psi_p = self.psi
        
        self.ref_aa = 0
        self.aa = 0
        self.com = np.array([0,0])
        self.id = uav_id
        # self.mass = 1
        self.tgt = tgt
        self.vel = 2
        self.sensor_mode = False
        self.sensor_az = np.deg2rad(30)
        self.detect_launch = False
        self.detect_launch_ML = False
        self.inrange = False
        
        self.vel_limit = 15
        self.vec = self.az2vec(self.az)
        self.hitpoint = 1.0

        self.cool_down_limit = 200
        self.cool_down = self.cool_down_limit
        self.radar_range = 40*1000
        self.mrm_range = 40*1000
    
    
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

    def state_calc(self,state0,t,T,alp,phi):
        x,y,z,V,psi,gam = state0[0],state0[1],state0[2],state0[3],state0[4],state0[5]
        D = 0.5*1.29*self.S*(self.Cd2*alp*alp+ self.Cd1*alp +self.Cd0)*V*V
        L = 0.5*1.29*self.S*(self.Cl1*alp+self.Cl0)*V*V
        v = [0]*6
        # v[3] = V + (T/self.mass*np.cos(alp) -D/self.mass -self.g*np.sin(gam))*t[1]
        # v[4] = psi + ((L/(self.mass*V*np.cos(gam))*np.sin(phi)))*t[1]
        # v[5] = gam + (((T*np.sin(alp) + L)/(self.mass*V)*np.cos(phi)-self.g/V*np.cos(gam)))*t[1]
        
        v[3] = V + (T*np.cos(alp) -D -self.mass*self.g*np.sin(gam) )/(self.mass)*t[1]
        v[4] = psi + (L*np.sin(phi))/(self.mass*V*np.cos(gam))*t[1]
        v[5] = gam + (T*np.sin(alp)+L*np.cos(phi)-self.mass*self.g*np.cos(gam))/(self.mass*V)*t[1]
 
        if v[5] >= np.pi/2:
            v[4] = v[4] + np.pi
            v[5] = np.pi -v[5] 
        elif v[5] <= -np.pi/2:
            v[4] = v[4] - np.pi
            v[5] = - np.pi -v[5]
            
        if v[4] >= 2*np.pi:
            v[4]= v[4]%(2*np.pi)
        elif v[4] <= 0:
            v[4]= v[4]%(2*np.pi) 
        
        v[0] = x+ v[3]*np.cos(v[5])*np.sin(v[4])*t[1]
        v[1] = y+ v[3]*np.cos(v[5])*np.cos(v[4])*t[1]
        v[2] = z+ v[3]*np.sin(v[5])*t[1]
        return v
        
    def state(self,state0,t,T,alp,phi):
        x,y,z,V,psi,gam = state0[0],state0[1],state0[2],state0[3],state0[4],state0[5]
        v = np.zeros([len(t),len(state0)])
        for i in range(len(t)):
            if i == 0:
                v[i,:] = self.state_calc(state0, t, T, alp, phi)
            else:
                state0 = v[i-1,:]
                v[i,:] = self.state_calc(state0, t, T, alp, phi)
        return v
    
    def pos_update_ctrl(self, sim_dt):
        if self.tgt.hitpoint == 0 and self.faction == "mrm":      
            if self.tgt.faction == "red":
                self.tgt.pos = np.array([-1000, self.lon/2, 0])
            elif self.tgt.faction == "blue":
                self.tgt.pos = np.array([2000, self.lon/2, 0])
            self.com = np.array([0,0,0])
        if self.hitpoint > 0:
            self.guidance_law(sim_dt)
            t = np.arange(0, sim_dt, sim_dt/10)
                #state

            state0 = [self.pos[0],self.pos[1],self.pos[2], self.V, self.psi, self.gam]
            # v = odeint(self.state, state0, t, args=(self.T, self.alp, self.phi))
            v = self.state(state0,t,self.T,self.alp,self.phi)

            self.pos[0],self.pos[1],self.pos[2], self.V, self.psi, self.gam = v[-1,:]
            
            if self.faction == "red":
                self.aa = self.aa + np.deg2rad(5)
                # self.pos = np.array([50*1000, 50*1000, 5*1000])
                # self.psi = np.deg2rad(90)
                # self.gam = np.deg2rad(0)
                # self.phi = 0
            if self.faction == "blue":
                self.aa = self.aa + np.deg2rad(5)
                # self.pos = np.array([51*1000, 50*1000, 7.5*1000])
                # self.psi = self.psi*0 + np.deg2rad(-90)
                # self.gam = np.deg2rad(70)
                # self.phi = 0
            self.check_eular()
        else:
            if self.faction == "mrm":
                if self.parent.faction == "red":
                    self.pos = np.array([-self.parent.safe_area[1], self.lon/2,0])
                elif self.parent.faction == "blue":
                    self.pos = np.array([2*self.parent.safe_area[1], self.lon/2,0])
                    self.az = self.vec2az(self.vec)
    def check_eular(self):
        if (self.gam//(np.pi/2))%4 == 0:
            self.gam = self.gam%(np.pi/2)
        elif(self.gam//(np.pi/2))%4 == 1:
            self.gam = -self.gam%(np.pi/2)
            self.psi = self.psi + np.pi
            self.phi = self.phi + np.pi
        elif(self.gam//(np.pi/2))%4 == 2:
            self.gam = -(self.gam%(np.pi/2))
            self.psi = self.psi + np.pi
            self.phi = self.phi + np.pi
        elif(self.gam//(np.pi/2))%4 == 3:
            self.gam = -(np.pi/2) + self.gam%(np.pi/2)
            
        if self.psi > np.pi:
            self.psi = self.psi - 2*np.pi
        elif self.psi < -np.pi:
            self.psi = self.psi + 2*np.pi  
        if self.phi > np.pi:
            self.phi = self.phi - 2*np.pi
        elif self.phi < -np.pi:
            self.phi = self.phi + 2*np.pi  
            
    def pos_update_ML(self, sim_dt, action):
        self.vec = self.vec_norm(self.vec + 0.01*self.Izz*action*self.thrust/self.mass*sim_dt)

        self.vel = self.vel + self.thrust/self.mass*sim_dt
        if self.vel > self.vel_limit:
            self.vel = self.vel_limit
        
        self.pos = self.pos + self.vel*self.vec*sim_dt
        self.az = self.vec2az(self.vec)

        
    def guidance_law(self, sim_dt):
        if self.detect_launch:
            # self.com = -1*self.vec_norm((self.tgt.pos - self.pos))  
            psi_sgn = np.pi
        else:
            psi_sgn = 0

        tgt_vector = self.tgt.pos+ 0*np.array([self.pos[0]*0+50000, self.pos[1]*0 +50000, self.pos[2]*0+75000]) - self.pos
        psi_ref = self.ops_az()
        gam_ref = self.ops_gam()
        
        if self.faction == "blue" or self.faction == "mrm" or self.faction == "red":
            V_ref = 240
            Kp_v = 1000
            Kd_v = 500
            
            V_err = (V_ref - self.V)
            V_err_d = (self.V_p - self.V)/sim_dt
            T_com = Kp_v*V_err + Kd_v*V_err_d
            self.T = self.T +  + T_com
            
            Kp_p = 0.1
            Kp_d = 1.0
            phi_lim = np.deg2rad(90)
            psi_err = psi_ref - self.psi*0
            # if psi_err > np.pi:
            #     psi_err = -2*np.pi + psi_err
            # elif psi_err < -np.pi:
            #     psi_err = 2*np.pi + psi_err
            psi_err_d = (self.psi_p - self.psi)/sim_dt
            phi_com = Kp_p*psi_err + Kp_d*psi_err_d
            self.phi = self.phi + phi_com
            if self.phi >= phi_lim:
                self.phi = phi_lim
            elif self.phi <= -phi_lim:
                self.phi = -phi_lim

            if self.pos[2] <  tgt_vector[2] and gam_ref < 0:
                gam_ref = -gam_ref
            Kp_a = 0.1
            Kd_a = 0.1
            alp_lim = np.deg2rad(5)
            gam_err = gam_ref - self.gam*0
            gam_err_d = (self.gam_p - self.gam)/sim_dt
            alp_com = Kp_a*gam_err + Kd_a*gam_err_d
            
            self.alp = self.alp + alp_com
            # self.alp = self.alp*np.cos(self.phi)
            # self.alp = alp_lim
            if self.alp >= alp_lim:
                self.alp = alp_lim
            elif self.alp <= -alp_lim:
                self.alp = -alp_lim
            if self.faction == "blue":
                print(np.rad2deg([psi_ref,gam_ref,self.psi,self.gam,self.phi,self.alp]))
            
            # self.phi = np.arctan2(tgt_vector[2], tgt_vector[1])
            # rot_phi = get_rotation_matrix_3d_phi(-self.phi)
            # tgt_vector = np.dot(rot_phi,tgt_vector)
            
            self.V_p = self.V
            self.gam_p = self.gam
            self.psi_p = self.psi

            
        
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
        tgt_pos = self.tgt.pos - self.pos
        tgt_az = np.arctan2(tgt_pos[0],tgt_pos[1]) - self.psi
        # tgt_az = self.vec2az(self.tgt.pos - self.pos) - self.vec2az(self.vec)
        if tgt_az > np.pi:
            tgt_az = tgt_az - 2*np.pi
        elif tgt_az < -np.pi:
            tgt_az = tgt_az + 2*np.pi
        return tgt_az
    
    def ops_gam(self):
        tgt_pos = self.tgt.pos - self.pos
        tgt_az = np.arctan2(tgt_pos[0],tgt_pos[1]) - self.psi
        rot_psi = get_rotation_matrix_3d_psi(self.psi) 
        tgt_vec = np.dot(rot_psi,tgt_pos)
        tgt_gam = np.arctan2(tgt_vec[2],np.abs(tgt_vec[0])) - self.gam
        if tgt_gam > np.pi:
            tgt_gam = tgt_gam - 2*np.pi
        elif tgt_gam < -np.pi:
            tgt_gam = tgt_gam + 2*np.pi
        return tgt_gam
    
    def calc_aa(self):
        self.aa = -self.vec2az(self.pos - self.tgt.pos) + np.pi - self.vec2az(self.tgt.vec)
        if self.aa > np.pi:
            self.aa = self.aa - 2*np.pi
        elif self.aa < -np.pi:
            self.aa = self.aa + 2*np.pi

