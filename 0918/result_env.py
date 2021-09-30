# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 20:29:48 2021

@author: Takumi
"""

import numpy as np
import matplotlib.pyplot as plt

class render_env:

    def copy_from_env(env):
        blue_pos = [env.timer]
        red_pos = [env.timer]
        mrm_pos = [env.timer]
        blue_mrm_num = 0
        red_mrm_num = 0
        for i in range(env.blue_num):
            blue_pos.append(np.append(env.blue[i].pos, env.blue[i].hitpoint))
            # blue_pos = np.append(blue_pos,env.blue[i].pos)
            # blue_pos = np.append(blue_pos,env.blue[i].hitpoint)
            blue_mrm_num = blue_mrm_num + env.blue[i].mrm_num
            
        for i in range(env.red_num):
            red_pos.append(np.append(env.red[i].pos, env.red[i].hitpoint))
            # red_pos = np.append(red_pos,env.red[i].pos)
            # red_pos = np.append(red_pos,env.red[i].hitpoint)
            red_mrm_num = red_mrm_num + env.red[i].mrm_num
        if env.mrm_num > 0:
            for i in range(env.mrm_num):
                mrm_pos.append(np.append(env.mrm[i].pos, env.mrm[i].hitpoint))
                # mrm_pos = np.append(mrm_pos,env.mrm[i].pos)
                # mrm_pos = np.append(mrm_pos,env.mrm[i].hitpoint)
        for i in range(blue_mrm_num + red_mrm_num):
            # mrm_pos = np.append(mrm_pos,np.zeros([3,1]))
            mrm_pos.append(np.zeros([3]))
        return blue_pos, red_pos, mrm_pos


    def rend(env,hist_pos,f_color , fig):
        time = hist_pos[:,0]
        num = hist_pos.shape[1]-1
        
        for i in range(num):
            # pos = np.vstack(hist_pos[:,i+1])
            pos_temp = np.vstack(hist_pos[:,i+1].tolist())
            pos = pos_temp[np.all(pos_temp > 0, axis=1)]
            fig1 = plt.figure(fig)
            plt.plot(pos[:,0],env.WINDOW_SIZE_lon-pos[:,1],'-',color = f_color)
        plt.grid('on')
        plt.xlim(0,env.WINDOW_SIZE_lat)
        plt.ylim(0,env.WINDOW_SIZE_lon)
        plt.show()
        plt.axes().set_aspect('equal')
        
    def rend_3d(env,hist_pos,f_color , fig):
        time = hist_pos[:,0]
        num = hist_pos.shape[1]-1
        
        for i in range(num):
            # pos = np.vstack(hist_pos[:,i+1])
            pos_temp = np.vstack(hist_pos[:,i+1].tolist())
            pos = pos_temp[np.all(pos_temp > 0, axis=1)]
            fig1 = plt.figure(fig)
            ax = fig1.gca(projection='3d')
            X = pos[:,0]
            Y = pos[:,1]
            Z = pos[:,2]
            # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5
            # mid_x = (X.max()+X.min()) * 0.5
            # mid_y = (Y.max()+Y.min()) * 0.5
            # mid_z = (Z.max()+Z.min()) * 0.5
            # ax.set_xlim(mid_x - max_range, mid_x + max_range)
            # ax.set_ylim(mid_y - max_range, mid_y + max_range)
            # ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.plot(X, Y, Z)
            # plt.plot(pos[:,0],env.WINDOW_SIZE_lon-pos[:,1],'-',color = f_color)
            
            
        
        
        

        
        
        plt.grid('on')
        ax.set_xlim(0,env.WINDOW_SIZE_lat)
        ax.set_xlim(0,env.WINDOW_SIZE_lon)
        ax.set_xlim(0,env.WINDOW_SIZE_alt)
        plt.show()