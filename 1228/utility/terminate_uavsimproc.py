# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 08:24:26 2021

@author: ookit1
"""

import psutil

class teminate_proc:
    def UAVsimprockill(proc_name):
        dict_pids = {
            p.info["pid"]: p.info["name"]
            for p in psutil.process_iter(attrs=["pid", "name"])
        }
        # for dict_pids in 
        for key, value in dict_pids.items():
            if value == proc_name:
                psutil.Process(key).kill()