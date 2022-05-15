# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:51:40 2021

@author: Takumi
"""
import numpy as np

def get_rotation_matrix_3d_psi(rad):
    rad = rad+np.pi*0
    rot = np.array([[np.cos(rad), -np.sin(rad), 0],
                    [np.sin(rad), np.cos(rad), 0],
                    [0,           0,           1]])
    return rot

def get_rotation_matrix_3d_gam(rad):
    rad = -rad
    rot = np.array([[np.cos(rad), 0, np.sin(rad)],
                    [0,           1,           0],                    
                    [-np.sin(rad), 0, np.cos(rad)]])
    return rot

def get_rotation_matrix_3d_phi(rad):
    rad = -rad
    rot = np.array([[1,          0,            0],
                    [0,np.cos(rad), -np.sin(rad)],
                    [0,np.sin(rad),  np.cos(rad)]])
    return rot