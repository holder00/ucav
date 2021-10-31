# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 20:29:48 2021

@author: Takumi
"""

import numpy as np
import matplotlib.pyplot as plt
from get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from get_rotation_matrix_3d import get_rotation_matrix_3d_phi
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

class render_env:
    # def __init__(self):
    #     pass
    
    def copy_from_env(env):
        blue_pos = [env.timer]
        red_pos = [env.timer]
        mrm_pos = [env.timer]
        blue_mrm_num = 0
        red_mrm_num = 0
        for i in range(env.blue_num):
            temp = np.append(env.blue[i].pos,env.blue[i].psi)
            temp = np.append(temp,env.blue[i].gam)
            temp = np.append(temp,env.blue[i].phi)
            blue_pos.append(np.append(temp, env.blue[i].hitpoint))
            # blue_pos = np.append(blue_pos,env.blue[i].pos)
            # blue_pos = np.append(blue_pos,env.blue[i].hitpoint)
            blue_mrm_num = blue_mrm_num + env.blue[i].mrm_num
            
        for i in range(env.red_num):
            temp = np.append(env.red[i].pos,env.red[i].psi)
            temp = np.append(temp,env.red[i].gam)
            temp = np.append(temp,env.red[i].phi)
            red_pos.append(np.append(temp, env.red[i].hitpoint))
            # red_pos = np.append(red_pos,env.red[i].pos)
            # red_pos = np.append(red_pos,env.red[i].hitpoint)
            red_mrm_num = red_mrm_num + env.red[i].mrm_num
        if env.mrm_num > 0:
            for i in range(env.mrm_num):
                temp = np.append(env.mrm[i].pos,env.mrm[i].psi)
                temp = np.append(temp,env.mrm[i].gam)
                temp = np.append(temp,env.mrm[i].phi)
                mrm_pos.append(np.append(temp, env.mrm[i].hitpoint))
                # mrm_pos = np.append(mrm_pos,env.mrm[i].pos)
                # mrm_pos = np.append(mrm_pos,env.mrm[i].hitpoint)
        for i in range(blue_mrm_num + red_mrm_num):
            # mrm_pos = np.append(mrm_pos,np.zeros([3,1]))
            mrm_pos.append(np.zeros([7]))
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
        plt.zlim(0,env.WINDOW_SIZE_alt)
        plt.show()
        plt.axes().set_aspect('equal')
        

    
    def rend_3d(env,hist_pos,f_color , fig):
        
        time = hist_pos[:,0]
        num = hist_pos.shape[1]-1
        
        for i in range(num):
            # pos = np.vstack(hist_pos[:,i+1])

            

            # pos = pos_temp[np.all(pos_temp[:,-1] != 0, axis=1),:]
            if hist_pos.shape[0] < 25:
                pos_temp = np.vstack(hist_pos[:,i+1].tolist())
            else:
                pos_temp = np.vstack(hist_pos[-25:,i+1].tolist())
            pos = pos_temp[~(pos_temp[:,-1] == 0)]
            fig1 = plt.figure(fig)
            ax = fig1.gca(projection='3d')
            if pos.size != 0:
                if pos.shape[0] < 25:
                    X = pos[:,0]
                    Y = pos[:,1]
                    Z = pos[:,2]
                else:
                    X = pos[-25:,0]
                    Y = pos[-25:,1]
                    Z = pos[-25:,2]
                rot_psi = get_rotation_matrix_3d_psi(pos[-1,3])
                
                rot_gam = get_rotation_matrix_3d_gam(pos[-1,4])
                
                rot_phi = get_rotation_matrix_3d_phi(pos[-1,5])
                vecter = np.array([1, 0, 0])*10000
                vecter = np.dot(rot_gam,vecter)
                vecter = np.dot(rot_psi,vecter)
                
                
                vecter_phi = np.array([0, 1, 0])*5000
                vecter_phi = np.dot(rot_phi,vecter_phi)
                vecter_phi = np.dot(rot_gam,vecter_phi)
                vecter_phi = np.dot(rot_psi,vecter_phi)
                # print(vecter)
                # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5
                # mid_x = (X.max()+X.min()) * 0.5
                # mid_y = (Y.max()+Y.min()) * 0.5
                # mid_z = (Z.max()+Z.min()) * 0.5
                # ax.set_xlim(mid_x - max_range, mid_x + max_range)
                # ax.set_ylim(mid_y - max_range, mid_y + max_range)
                # ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.plot([X[-1],X[-1]+vecter[0]], [Y[-1],Y[-1]+vecter[1]], [Z[-1],Z[-1]+vecter[2]],color = f_color)
                ax.plot([X[-1]-vecter_phi[0],X[-1]+vecter_phi[0]],
                        [Y[-1]-vecter_phi[1],Y[-1]+vecter_phi[1]],
                        [Z[-1]-vecter_phi[2],Z[-1]+vecter_phi[2]],color = f_color)
                ax.plot(X, Y, Z,color = f_color)
                ax.plot(X, Y, 0,color = f_color)
                ax.plot([X[-1],X[-1]], [Y[-1],Y[-1]], [0,Z[-1]],'--',color = f_color)
                # ax.scatter(X, Y, Z,'o',color = f_color)
                
                # plt.plot(pos[:,0],env.WINDOW_SIZE_lon-pos[:,1],'-',color = f_color)
                
        plt.grid('on')
        ax.set_xlim(0,env.WINDOW_SIZE_lat)
        ax.set_ylim(0,env.WINDOW_SIZE_lon)
        ax.set_zlim(0,env.WINDOW_SIZE_alt)
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, env.WINDOW_SIZE_alt/env.WINDOW_SIZE_lat, 1]))
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_zlabel("Altitude [m]")
        plt.show()
        