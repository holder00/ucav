# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:15:08 2021

@author: Takumi
"""

import glob
import os
import copy



def latest_learned_file_path(path):
    list_of_folder = glob.glob(path)
    time_p = 0
    res = []

    for file in list_of_folder:
        list_of_files = glob.glob(file + "\*")
    
        l_in = [s for s in list_of_files if not 'tune' in s]
        time = os.path.getmtime(l_in[0])
        
        if time_p < time:
            # print(l_in[0])

            res = copy.deepcopy(l_in[0])
        time_p = time
    
    return res
    
# model_path = latest_learned_file_path('./UCAV/checkpoints/test_2/*')