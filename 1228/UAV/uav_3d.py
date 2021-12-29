# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:35:21 2021

@author: Takumi
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_phi
from utility.ATTITUDE_CONVERT import ATTITUDE_CONVERT
#座標系クラスのインポート
from utility.COORDINATION_SYSTEMS import ECEF, LLA, NED, FRD, AC_ATTITUDE, LOCAL, SENSOR_ATTITUDE, FOCAL_POSITION, FOCAL_ANGLE
from utility.harf_angle import harf_angle
import copy
# from numba import jit
class uav_3d:
    def __init__(self, lat_lim, lon_lim, safe_area, faction, uav_id, tgt):
        self.pos = np.array([np.random.randint(safe_area[0], safe_area[1]), np.random.randint(lon_lim/4, 3*lon_lim/4), np.random.randint(7000, 12000)])
        self.safe_area = safe_area
        self.lon_lim = lon_lim
        self.lat_lim = lat_lim
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
        self.rho = 0.5
        self.fire = False
        self.V = 250 #m/s
        self.gam = np.deg2rad(0)

        self.id = uav_id
        if self.faction == "blue":
            self.az = np.deg2rad(180)
            self.psi = np.deg2rad(180)
            self.phi = np.deg2rad(0)
            self.alp = np.deg2rad(0)
            # self.pos = np.array([80*1000, 50*1000, 5*1000])
            self.mrm_num = 2
            self.radar_range = 100*1000
            self.mrm_range = 50*1000

            # if self.id == 0:
            #     self.mrm_num = 1
            # self.pos = np.array([150*1000, 100*1000, 5*1000])
            # else:
            #     self.mrm_num = 0

        elif self.faction == "red":
            # self.pos = np.array([100*1000, 100*1000, 5*1000])
            self.az = np.deg2rad(0)
            self.psi = np.deg2rad(0)
            self.phi = np.deg2rad(0)
            self.alp = np.deg2rad(1)
            self.mrm_num = 2
            self.radar_range = 100*1000
            self.mrm_range = 60*1000
            
        # self.opp_ecef = NED(self.pos[0],self.pos[1],self.pos[2])
        self.V_p = self.V
        self.gam_p = self.gam
        self.psi_p = self.psi

        self.V_ref = self.V
        self.psi_ref = self.psi
        self.gam_ref = self.gam

        self.aa = 0
        self.tgt = tgt
        self.role = "shooter" #shooter or decoy

        self.sensor_az = np.deg2rad(180)
        self.detect_launch = False
        self.detect_launch_ML = False

        self.hitpoint = 1.0
        self.inrange = False
        self.cool_down_limit = 200
        self.cool_down = self.cool_down_limit


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
        self.inrange = False
        if self.hitpoint > 0 and self.tgt.hitpoint != 0 and np.linalg.norm(self.pos - self.tgt.pos) <= self.radar_range:
            temp_az = self.ops_az()-self.psi
            temp_az=harf_angle(temp_az)

            temp_gam = self.ops_gam()-self.gam
            temp_gam=harf_angle(temp_gam)

            if np.abs(temp_az) <= self.sensor_az and np.abs(temp_gam) <= self.sensor_az:
                self.inrange = True
        else:
            self.inrange = False
        return self.inrange

    def can_launch(self):
        self.cool_down = self.cool_down + 1
        launch = False
        self.fire = False
        if self.inrange and self.cool_down > self.cool_down_limit and self.mrm_num > 0 and np.linalg.norm(self.pos - self.tgt.pos) <= self.mrm_range:
            self.cool_down = 0
            self.mrm_num = self.mrm_num - 1
            launch = True
            self.fire = True
        return launch

    def can_launch_ML(self):
        launch = False
        self.fire = False
        if self.inrange and self.cool_down == self.cool_down_limit and self.mrm_num > 0 and np.linalg.norm(self.pos - self.tgt.pos) <= self.mrm_range*10:
            self.cool_down = 0
            self.mrm_num = self.mrm_num - 1
            launch = True
            self.fire = True
        return launch

    def MAWS(self, mrm):
        if self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint > 0 and self.hitpoint > 0:
            self.detect_launch = True
        elif self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint == 0:
            self.detect_launch = False


    def MAWS_ML(self, mrm):
        if self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint > 0 and self.hitpoint > 0:
            self.detect_launch_ML = True
        elif self.faction == mrm.tgt.faction and self.id == mrm.tgt.id and mrm.hitpoint == 0:
            self.detect_launch_ML = False

    def state_calc(self,state0,t,T,alp,phi):
        x,y,z,V,psi,gam = state0[0],state0[1],state0[2],state0[3],state0[4],state0[5]
        D = 0.5*self.rho*self.S*(self.Cd2*alp*alp+ self.Cd1*alp +self.Cd0)*V*V
        L = 0.5*self.rho*self.S*(self.Cl1*alp+self.Cl0)*V*V
        v = [0]*6

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
        # ATTITUDE_CONVERT.frd2ned()
        if self.hitpoint > 0:
            self.guidance_law(sim_dt)

            t = np.arange(0, sim_dt, sim_dt/10)
                #state


            state0 = [self.pos[0],self.pos[1],self.pos[2], self.V, self.psi, self.gam]
            # v = odeint(self.state, state0, t, args=(self.T, self.alp, self.phi))
            v = self.state(state0,t,self.T,self.alp,self.phi)

            # self.pos[0],self.pos[1],self.pos[2], self.V, self.psi, self.gam = v[-1,:]
            self.V = v[-1,3]
            if self.faction == "red":
                self.aa = self.aa + np.deg2rad(5)
                # self.pos = np.array([50*1000, 50*1000, 5*1000])
                # self.psi = np.deg2rad(90)
                # self.gam = np.deg2rad(0)
                # self.phi = 0
            if self.faction == "blue":
                self.aa = self.aa + np.deg2rad(5)

                # self.pos = np.array([75*1000, 50*1000, 5*1000])
                # self.psi = self.psi*0 + np.deg2rad(-90+30)
                # self.gam = np.deg2rad(30)
                # self.phi = 0
            self.check_eular()

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

    def ops_az(self):
        tgt_pos = np.empty(self.pos.shape,np.float64)
        tgt_pos = self.tgt.pos - self.pos
        tgt_az = np.arctan2(tgt_pos[1],tgt_pos[0])
        tgt_az = harf_angle(tgt_az)

        return tgt_az

    def ops_gam(self):
        tgt_pos = np.empty(self.pos.shape,np.float64)

        tgt_pos = self.tgt.pos - self.pos
        tgt_az = np.arctan2(tgt_pos[1],tgt_pos[0])
        rot_psi = get_rotation_matrix_3d_psi(-tgt_az)
        tgt_vec = np.dot(rot_psi,tgt_pos)
        tgt_gam = np.arctan2(tgt_vec[2],np.abs(tgt_vec[0]))
        tgt_gam = harf_angle(tgt_gam)
        return tgt_gam

