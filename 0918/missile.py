# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 23:02:41 2021

@author: Takumi
"""
from uav import uav
import numpy as np

class missile(uav):
    def __init__(self, parent):
        # super(missile, self).__init__(0, 10, [0,10])
        self.parent = parent
        self.safe_area = parent.safe_area
        self.lon = parent.lon
        self.pos = parent.pos
        self.vec = parent.vec
        self.vel = parent.vel
        self.faction = "mrm"
        self.destruction_range = 10
        self.destruction_angle = 10
        self.tgt = parent.tgt
        self.vel_limit = 100
        self.Izz = 50
        self.thrust = 1.5
        self.detect_launch = False
        self.missile_tgt_lost = False
        self.mass = 1
        self.hitpoint = 250
        self.thrust_dt = 0.001
        self.sensor_az = np.deg2rad(60)
        self.radar_range = 500
        
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
        else:
            self.com = self.vec_norm((self.tgt.pos - self.pos))
    # def pos_update_ctrl(self, sim_dt, tgt):
    #     if self.hitpoint > 0:
    #         self.hitpoint = self.hitpoint - 1
    #         self.guidance_law(sim_dt, tgt)
    #         self.vec = self.vec_norm(self.vec + 0.01*self.Izz*self.com*self.thrust/self.mass*sim_dt)
    
    #         self.vel = self.vel + self.thrust/self.mass*sim_dt
    #         if self.vel > self.vel_limit:
    #             self.vel = self.vel_limit
    
    #         self.pos = self.pos + self.vel*self.vec*sim_dt
    #     else:
    #         self.pos = np.array([-1000, -1000])
    