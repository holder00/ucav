# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:19:06 2021

@author: Takumi
"""

import argparse
import gym
import datetime
import os
import random
import tempfile
import numpy as np
import pickle

a = np.arange(1,13).reshape(3,4)

# b = a[[2,1,0], : ]
b = {}
for i in range(len(a)):
    # temp = a[i,:]
    # oth = np.delete(a,i,0)
    ob = np.vstack([a[i,:],np.delete(a,i,0)])
    b['blue_' + str(i)] = np.vstack([a[i,:],np.delete(a,i,0)]).astype(np.float32)