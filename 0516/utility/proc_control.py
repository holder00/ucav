# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:14:04 2022

@author: OokiT1
"""
import psutil
import subprocess
import time

class proc_ctrl:

    def process_kill(proc_name):
        # print(proc_name)
        dict_pids = {
            p.info["pid"]: p.info["name"]
            for p in psutil.process_iter(attrs=["pid", "name"])
        }
        # for dict_pids in 
        for key, value in dict_pids.items():
            if value == proc_name:
                psutil.Process(key).kill()
                time.sleep(0.05)
                
    def process_start(memno):
        # subprocess.Popen(r"./UCAV_vec.exe /BLUE 1 /RED 0 /MEM "+str(memno)+" /LOG")
        subprocess.Popen(r"./EXE/UCAV"+str(memno)+".exe /BLUE 1 /RED 0 /MEM "+str(memno))
        time.sleep(0.05)
        
    def process_kill_all():
        for i in range(100):
            dict_pids = {
                p.info["pid"]: p.info["name"]
                for p in psutil.process_iter(attrs=["pid", "name"])
            }
            # for dict_pids in 
            for key, value in dict_pids.items():
                if value == "UCAV"+str(i)+".exe":
                    psutil.Process(key).kill()
