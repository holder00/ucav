# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:57:36 2021

@author: OokiT1
"""

import numpy as np

def harf_angle(angle):
    
    if angle < -np.pi:
        angle = angle + 2*np.pi
    if angle > np.pi:
        angle = angle - 2*np.pi
        
    return angle