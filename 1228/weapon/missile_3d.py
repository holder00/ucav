# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 23:24:23 2021

@author: Takumi
"""
from UAV.uav_3d import uav_3d
import numpy as np
import copy
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_phi

from scipy.interpolate import interp1d

class missile_3d(uav_3d):
    def __init__(self, parent):
        # super(missile, self).__init__(0, 10, [0,10])
        self.parent = parent
        self.safe_area = parent.safe_area
        self.lon_lim = copy.deepcopy(parent.lon_lim)
        self.lat_lim = copy.deepcopy(parent.lat_lim)
        self.pos = copy.deepcopy(parent.pos)
        self.faction = "mrm"
        self.destruction_range = 250
        self.destruction_angle = 250
        self.tgt = parent.tgt
        self.mass = 200 #kg
        self.g = 9.80665 #m/ss
        self.Cd2 = 3.0132
        self.Cd1 = 0.0517
        self.Cd0 = 0.0249
        self.Cl1 = 3.2417
        self.Cl0 = -0.0184
        self.Cb = -0.8594
        self.T = 6000*self.g
        self.S = 0.025
        self.V = parent.V
        self.rho = 0.5
        self.psi = copy.deepcopy(parent.psi)
        self.gam = copy.deepcopy(parent.gam)
        self.alp = np.deg2rad(0)
        self.phi = np.deg2rad(0)
        self.V_p = self.V
        self.gam_p = self.gam
        self.psi_p = self.psi

        self.inrange = False
        self.detect_launch = False
        self.hitpoint_ini = 300
        self.hitpoint = 300
        self.thrust_dt = 500
        self.sensor_az = np.deg2rad(180)
        self.radar_range = 10*1000
        self.CD0_table = np.array([[0,0.75,1,1.1,1.2,1.3,1.5,2,3,4],[0.42,0.45,0.55,0.68,0.96,0.94,0.88,0.74,0.6,0.49]])
        self.CD0 = 0.45

    def calc_temp(self,alt):
        t = 15-0.0065*alt
        return t

    def calc_mach(self,vel):
        mach0 = 331.5 + 0.61*self.calc_temp(self.pos[2])
        mach = self.V/mach0
        return mach


    def update_status(self, sim_dt):
        if self.hitpoint > 0:

            self.hitpoint = self.hitpoint - 1
            if (self.hitpoint_ini -self.hitpoint) < 2:
                self.T = 6000*self.g
                self.mass = 200
            elif (self.hitpoint_ini -self.hitpoint) < 10:
                self.T = 920*self.g
                self.mass = 157
            else:
                self.T = 0.0
                self.mass = 130
                K = 0.05
            self.CD = self.CD0 + 0.05*0.1*self.mass/(self.rho*(self.V**2)*self.S)

            self.V = self.V +(self.T-0.5*self.rho*(self.V**2)*self.CD*self.S)/(self.mass)*sim_dt
            if (self.tgt_inrange() or self.parent.tgt_inrange()) and not self.tgt.hitpoint == 0:

                # self.pos_update_ctrl(sim_dt)

                K = 1.0
                self.gam = K*self.ops_gam() + self.gam*0
                self.psi = K*self.ops_az() + self.psi*0

                # print(self.pos,self.tgt.pos,np.rad2deg(self.gam),np.rad2deg(self.ops_gam()))
                # print(self.V,self.gam,self.psi)
                self.pos[0] = self.pos[0] + self.V*np.cos(self.gam)*np.cos(self.psi)*sim_dt
                self.pos[1] = self.pos[1] + self.V*np.cos(self.gam)*np.sin(self.psi)*sim_dt
                self.pos[2] = self.pos[2] + self.V*np.sin(self.gam)*sim_dt
            else:
                K = 0
                # self.psi = K*self.ops_az() + self.psi_p*0
                # self.gam = K*self.ops_gam() + self.gam_p*0

                self.pos[0] = self.pos[0] + self.V*np.cos(self.gam)*np.cos(self.psi)*sim_dt
                self.pos[1] = self.pos[1] + self.V*np.cos(self.gam)*np.sin(self.psi)*sim_dt
                self.pos[2] = self.pos[2] + self.V*np.sin(self.gam)*sim_dt
            self.phi = 0
            if self.V < 0:
                self.hitpoint = 0
            # print(self.V,self.T,self.hitpoint_ini,self.hitpoint,(self.hitpoint_ini -self.hitpoint)>2)
            # self.hitpoint = self.hitpoint - 1

    def guidance_law(self, sim_dt):
        if self.tgt.hitpoint == 0 or not self.tgt_inrange():
            psi_ref = 0
            gam_ref = 0
        else:
            psi_ref = self.ops_az()
            if psi_ref > np.pi:
                psi_ref = psi_ref - 2*np.pi
            elif psi_ref < -np.pi:
                psi_ref = psi_ref + 2*np.pi
            gam_ref = self.ops_gam()
            # if self.pos[2] <  self.tgt_vector[2] and gam_ref < 0:
                # gam_ref = -gam_ref

        V_ref = 1360
        Kp_v = 1000
        Kd_v = 500

        V_err = (V_ref - self.V)
        V_err_d = (self.V_p - self.V)/sim_dt
        T_com = Kp_v*V_err + Kd_v*V_err_d
        self.T = self.T - self.thrust_dt + T_com*0

        Kp_p = 0.5
        Kp_d = 0.1
        phi_lim = np.deg2rad(90)
        psi_err = psi_ref - self.psi*0

        psi_err_d = (self.psi_p - self.psi)/sim_dt
        phi_com = Kp_p*psi_err + Kp_d*psi_err_d
        self.phi = self.phi + phi_com
        if self.phi >= phi_lim:
            self.phi = phi_lim
        elif self.phi <= -phi_lim:
            self.phi = -phi_lim
        # if self.pos[2] <  self.tgt_vector[2] and gam_ref < 0:
            # gam_ref = -gam_ref
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

        # print(np.rad2deg(psi_ref),np.rad2deg(gam_ref))
        self.V_p = self.V
        self.gam_p = self.gam
        self.psi_p = self.psi


