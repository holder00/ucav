# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 23:24:23 2021

@author: Takumi
"""
from uav_3d import uav_3d
import numpy as np
import copy
from get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from get_rotation_matrix_3d import get_rotation_matrix_3d_phi

class missile_3d(uav_3d):
    def __init__(self, parent):
        # super(missile, self).__init__(0, 10, [0,10])
        self.parent = parent
        self.safe_area = parent.safe_area
        self.lon = copy.deepcopy(parent.lon)
        self.pos = copy.deepcopy(parent.pos)
        self.vec = copy.deepcopy(parent.vec)
        self.vel = copy.deepcopy(parent.vel)
        self.faction = "mrm"
        self.destruction_range = 500
        self.destruction_angle = 500
        self.tgt = parent.tgt
        self.vel_limit = 100
        self.Izz = 50
        self.thrust = 1.5
        self.detect_launch = False
        self.missile_tgt_lost = False
        self.mass = 1
        self.hitpoint = 75
        self.thrust_dt = 2000
        self.sensor_az = np.deg2rad(60)
        self.radar_range = 500
        self.mass = 4000 #kg
        self.g = 9.80665 #m/ss
        self.Cd2 = 3.0132
        self.Cd1 = 0.0517
        self.Cd0 = 0.0249
        self.Cl1 = 3.2417
        self.Cl0 = -0.0184
        self.Cb = -0.8594
        self.T = 16000*10
        self.T_lim = self.T
        self.S = 18
        self.V = parent.V

        self.psi = copy.deepcopy(parent.psi)
        self.gam = copy.deepcopy(parent.gam)
        self.alp = np.deg2rad(0)
        self.phi = np.deg2rad(0)
        self.V_p = self.V
        self.gam_p = self.gam
        self.psi_p = self.psi
        
    def update_status(self, sim_dt):
        if self.hitpoint > 0:
            self.pos_update_ctrl(sim_dt)
            self.hitpoint = self.hitpoint - 1
            self.thrust = self.thrust - self.thrust_dt
            if self.thrust < 0:
               self.thrust = 0 
        else:
            if self.parent.faction == "blue" and self.pos[0] != np.array(2*self.parent.safe_area[1]):
                self.missile_tgt_lost = True
            self.pos_update_ctrl(sim_dt)
            
    def guidance_law(self, sim_dt):
        if self.tgt.hitpoint == 0 or not self.tgt_inrange():
            self.com = np.array([0,0])
            tgt_vector = self.tgt.pos+ 0*np.array([self.pos[0]*0+50000, self.pos[1]*0 +50000, self.pos[2]*0+75000]) - self.pos
        else:
            tgt_vector = self.tgt.pos+ 0*np.array([self.pos[0]*0+50000, self.pos[1]*0 +50000, self.pos[2]*0+75000]) - self.pos
            
        psi_ref = self.ops_az()
        gam_ref = self.ops_gam()
        
        self.com = self.vec_norm((self.tgt.pos - self.pos))
        V_ref = 1360
        Kp_v = 1000
        Kd_v = 500
        
        V_err = (V_ref - self.V)
        V_err_d = (self.V_p - self.V)/sim_dt
        T_com = Kp_v*V_err + Kd_v*V_err_d
        self.T = self.T - self.thrust_dt
        if self.T < 0:
            self.T = 0

            
            

        
        Kp_p = 0.2
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

        
        # # self.phi = np.arctan2(tgt_vector[2], tgt_vector[1])
        # rot_phi = get_rotation_matrix_3d_phi(-self.phi)
        # tgt_vector = np.dot(rot_phi,tgt_vector)
        
        self.V_p = self.V
        self.gam_p = self.gam
        self.psi_p = self.psi
