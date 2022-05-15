# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 20:29:48 2021

@author: Takumi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_phi
# from stl import mesh

class render_env:
    # def __init__(self):
    #     pass
    def copy_from_env_mod(env):
        # for i in range(env.blue_num):
        #     blue_pos = np.array([env.timer,env.blue[i].pos,pos,env.blue[i].psi,env.blue[i].gam,env.blue[i].phi,env.blue[i].hitpoint])
        # red_pos = [env.timer]
        # mrm_pos = [env.timer]
        status_num = 13
        blue_mrm_num = 0
        red_mrm_num = 0
        blue_pos = np.zeros(status_num*env.blue_num)
        red_pos = np.zeros(status_num*env.blue_num)
                                                           
        for i in range(env.blue_num):
            if env.blue[i].detect_launch_ML:
                detect = 0
            else:
                detect = 1
            blue_pos[status_num*i:status_num*(i+1)] = np.concatenate([np.array([env.timer]),
                                       env.blue[i].pos,
                                       np.array([env.blue[i].psi]),
                                       np.array([env.blue[i].gam]),
                                       np.array([env.blue[i].phi]),
                                       np.array([env.blue[i].V]),
                                       np.array([detect]),
                                       np.array([env.reward_num["blue_" + str(i)]]),
                                       # np.array([0]),
                                       np.array([env.rewards_total["blue_" + str(i)]]),
                                       np.array([env.blue[i].tgt.id]),
                                       np.array([env.blue[i].hitpoint])
                                       ])
            blue_mrm_num = blue_mrm_num + env.blue[i].mrm_num
            
        for i in range(env.red_num):
            if env.red[i].detect_launch:
                detect = 0
            else:
                detect = 1
            red_pos[status_num*i:status_num*(i+1)] = np.concatenate([np.array([env.timer]),
                                       env.red[i].pos,
                                       np.array([env.red[i].psi]),
                                       np.array([env.red[i].gam]),
                                       np.array([env.red[i].phi]),
                                       np.array([env.red[i].V]),
                                       np.array([detect]),
                                       np.array([0]),
                                       np.array([0]),
                                       np.array([env.red[i].tgt.id]),
                                       np.array([env.red[i].hitpoint])
                                       ])
            red_mrm_num = red_mrm_num + env.red[i].mrm_num
        mrm_pos = np.concatenate([np.zeros((status_num)*(blue_mrm_num + red_mrm_num+env.mrm_num))])
        if env.mrm_num > 0:
            for i in range(env.mrm_num):
                mrm_pos[status_num*i:status_num*(i+1)] = np.concatenate([np.array([env.timer]),
                                           env.mrm[i].pos,
                                           np.array([env.mrm[i].psi]),
                                           np.array([env.mrm[i].gam]),
                                           np.array([env.mrm[i].phi]),
                                           np.array([env.mrm[i].V]),
                                           np.array([0]),
                                           np.array([0]),
                                           np.array([0]),
                                           np.array([env.mrm[i].tgt.id]),
                                           np.array([env.mrm[i].hitpoint])
                                           ])
        return blue_pos, red_pos, mrm_pos
    
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
        
    def rend_3d_mod3(time,hist_pos,f_color,fig,info,uav_id,traj_length,radar_plot):
        status_num = 13
        traj_length = int(traj_length)
        if time == 1:
            num = int((np.shape(hist_pos)[0])/status_num)#hist_pos.shape[1]-1
        else:
            num = int((np.shape(hist_pos)[1])/status_num)#hist_pos.shape[1]-1

        for i in range(num):
            # pos = np.vstack(hist_pos[:,i+1])

            
            if time < traj_length:
                if time == 1:
                    pos_temp = hist_pos[i*status_num+1:(i+1)*status_num]
                    state = hist_pos[i*status_num+1:(i+1)*status_num]
                else:
                    pos_temp = hist_pos[:,i*status_num+1:(i+1)*status_num]
                    state = hist_pos[-1,i*status_num+1:(i+1)*status_num]
            else:
                pos_temp = hist_pos[-traj_length:,i*status_num+1:(i+1)*status_num]
                state = hist_pos[-1,i*status_num+1:(i+1)*status_num]
            
            if f_color == "b":
                rend_faction = "blue"
            elif f_color == "r":
                rend_faction = "red"
            else:
                rend_faction = "mrm"

            if time == 1:
                pos = pos_temp[~(pos_temp[-1] == 0)]
            else:
                pos = pos_temp[~(pos_temp[:,-1] == 0)]
            fig1 = plt.figure(fig)
            pos = np.array(pos)

            ax = fig1.gca(projection='3d')
            if pos.size != 0:
                if pos.shape[0] < traj_length:
                    if time == 1:
                        pos = pos[0]
                        X = pos[0]/1000
                        Y = pos[1]/1000
                        Z = pos[2]/1000
                    else:
                        X = pos[:,0]/1000
                        Y = pos[:,1]/1000
                        Z = pos[:,2]/1000
                else:
                    X = pos[-traj_length:,0]/1000
                    Y = pos[-traj_length:,1]/1000
                    Z = pos[-traj_length:,2]/1000

                if rend_faction =="blue":
                    # plot_obs = env.blue[i]
                    sensor_az = info["blue_sensor_az"]
                    radar_range = info["blue_radar_range"]
                    mrm_range = info["blue_mrm_range"]
                    if uav_id == 0:
                        f_color_p = "g"
                        roles = "Decoy "+ "TGT: "+ str(int(state[-2]))
                    else:
                        f_color_p = "b"
                        roles = "Shooter "+ "TGT: "+ str(int(state[-2]))
                elif rend_faction =="red":
                    sensor_az = info["red_sensor_az"]
                    radar_range = info["red_radar_range"]
                    mrm_range = info["red_mrm_range"]
                    # plot_obs = env.red[i]
                    f_color_p = "r"
                    roles = "red_"+str(uav_id) +" "+ "TGT: "+ str(int(state[-2]))
                else:
                    # plot_obs = env.mrm[i]
                    sensor_az = info["mrm_sensor_az"]
                    radar_range = info["mrm_radar_range"]
                    f_color_p = "k"   
                    roles = ""

                rot_psi = get_rotation_matrix_3d_psi(state[3])
                rot_gam = get_rotation_matrix_3d_gam(state[4])
                rot_phi = get_rotation_matrix_3d_phi(state[5])
                if rend_faction == "mrm":
                    vecter = np.array([1, 0, 0])*5000/1000
                else:
                    vecter = np.array([1, 0, 0])*7000/1000
                    
                
                vecter = np.dot(rot_gam,vecter)
                vecter = np.dot(rot_psi,vecter)
                if rend_faction == "mrm":
                    vecter_gam = np.array([0, 0, 1])*2000/1000
                else:
                    vecter_gam = np.array([0, 0, 1])*3000/1000

                vecter_gam = np.dot(rot_phi,vecter_gam)
                vecter_gam = np.dot(rot_gam,vecter_gam)
                vecter_gam = np.dot(rot_psi,vecter_gam)
                if rend_faction == "mrm":
                    vecter_phi = np.array([0, 1, 0])*2000/1000
                else:
                    vecter_phi = np.array([0, 1, 0])*4000/1000
                vecter_phi = np.dot(rot_phi,vecter_phi)
                vecter_phi = np.dot(rot_gam,vecter_phi)
                vecter_phi = np.dot(rot_psi,vecter_phi)

                limit = [info["WINDOW_SIZE_lat"], info["WINDOW_SIZE_lon"], info["WINDOW_SIZE_alt"]]
                if time == 1:
                    obs_pos = [X,Y,Z]
                else:
                    obs_pos = [X[-1],Y[-1],Z[-1]]
                rotation = [rot_phi,rot_gam,rot_psi] 

                if radar_plot:
                    x,y,z = create_surfs(sensor_az,radar_range/1000,limit,obs_pos,rotation)
                    ax.plot_surface(x,y,z,color=f_color_p, antialiased=True,alpha=0.2,shade=False)
                    if rend_faction == "blue" or rend_faction == "red":
                        x,y,z = create_surfs(sensor_az,mrm_range/1000,limit,obs_pos,rotation)
                        ax.plot_surface(x,y,z,color=f_color_p, antialiased=True,alpha=0.05,shade=False)
                
                ax.text(obs_pos[0]+1, obs_pos[1]+1, obs_pos[2]+1, roles ,color=f_color_p,size=8)
                if rend_faction =="blue":
                    if state[7] == 0:
                        detect = " Detect"
                    else:
                        detect = " Nomal"
                    plot_text = format(time,"3.0f")+" "+ roles+" "+format(state[8],".5f")+" "+format(state[9],".2f")+\
                        " vel: " + format(state[6],".1f") + detect + " Azm: " + format(np.rad2deg(state[3]),".1f")
                    ax.text(10, 10+uav_id*20, 10, plot_text, color=f_color_p,size=8)
                
                ax.plot([obs_pos[0],obs_pos[0]+vecter[0]],
                        [obs_pos[1],obs_pos[1]+vecter[1]],
                        [obs_pos[2],obs_pos[2]+vecter[2]],color = f_color_p)
                
                ax.plot([obs_pos[0]-vecter_phi[0],obs_pos[0]+vecter_phi[0]],
                                 [obs_pos[1]-vecter_phi[1],obs_pos[1]+vecter_phi[1]],
                                 [obs_pos[2]-vecter_phi[2],obs_pos[2]+vecter_phi[2]],color = f_color_p)
                ax.plot([obs_pos[0],obs_pos[0]+vecter_gam[0]],
                                 [obs_pos[1],obs_pos[1]+vecter_gam[1]],
                                 [obs_pos[2],obs_pos[2]+vecter_gam[2]],color = f_color_p)
                ax.plot(X, Y, Z,'--',color = f_color_p, linewidth = 1.0)
                ax.plot(X, Y, 0,'--',color = f_color_p, linewidth = 1.0)
                ax.plot([obs_pos[0],obs_pos[0]], [obs_pos[1],obs_pos[1]], [0,obs_pos[2]],'o-',color = f_color_p,linewidth = 1.0,markersize=0.5)

            ax.set_box_aspect((info["WINDOW_SIZE_lat"],info["WINDOW_SIZE_lon"],info["WINDOW_SIZE_alt"]))

            ax.set_xlabel("North [km]")
            ax.set_ylabel("East [km]")
            ax.set_zlabel("Altitude [km]")


            # ax.view_init(elev=10, azim=-45)

            
            return ax
 
    
    
    
    def rend_3d_mod2(time,hist_pos,f_color,fig,info,plot_radar):
        status_num = 13
        if time == 1:
            num = int((np.shape(hist_pos)[0])/status_num)#hist_pos.shape[1]-1
        else:
            num = int((np.shape(hist_pos)[1])/status_num)#hist_pos.shape[1]-1
        
        for i in range(num):
            
            
            # pos = np.vstack(hist_pos[:,i+1])
            traj_length = 200
            
            if time < traj_length:
                if time == 1:
                    pos_temp = hist_pos[i*status_num+1:(i+1)*status_num]
                    state = hist_pos[i*status_num+1:(i+1)*status_num]
                else:
                    pos_temp = hist_pos[:,i*status_num+1:(i+1)*status_num]
                    state = hist_pos[-1,i*status_num+1:(i+1)*status_num]
            else:
                pos_temp = hist_pos[-traj_length:,i*status_num+1:(i+1)*status_num]
                state = hist_pos[-1,i*status_num+1:(i+1)*status_num]
            
            if f_color == "b":
                rend_faction = "blue"
            elif f_color == "r":
                rend_faction = "red"
            else:
                rend_faction = "mrm"

            if time == 1:
                pos = pos_temp[~(pos_temp[-1] == 0)]
            else:
                pos = pos_temp[~(pos_temp[:,-1] == 0)]
            fig1 = plt.figure(fig)
            pos = np.array(pos)

            ax = fig1.gca(projection='3d')
            if pos.size != 0:
                if pos.shape[0] < traj_length:
                    if time == 1:
                        pos = pos[0]
                        X = pos[0]/1000
                        Y = pos[1]/1000
                        Z = pos[2]/1000
                    else:
                        X = pos[:,0]/1000
                        Y = pos[:,1]/1000
                        Z = pos[:,2]/1000
                else:
                    X = pos[-traj_length:,0]/1000
                    Y = pos[-traj_length:,1]/1000
                    Z = pos[-traj_length:,2]/1000

                if rend_faction =="blue":
                    # plot_obs = env.blue[i]
                    sensor_az = info["blue_sensor_az"]
                    radar_range = info["blue_radar_range"]
                    mrm_range = info["blue_mrm_range"]
                    if i == 0:
                        f_color_p = "g"
                        roles = "Decoy "+ "TGT: "+ str(int(state[-2]))
                    else:
                        f_color_p = "b"
                        roles = "Shooter "+ "TGT: "+ str(int(state[-2]))
                elif rend_faction =="red":
                    sensor_az = info["red_sensor_az"]
                    radar_range = info["red_radar_range"]
                    mrm_range = info["red_mrm_range"]
                    # plot_obs = env.red[i]
                    f_color_p = "r"
                    roles = "red_"+str(i) +" "+ "TGT: "+ str(int(state[-2]))
                else:
                    # plot_obs = env.mrm[i]
                    sensor_az = info["mrm_sensor_az"]
                    radar_range = info["mrm_radar_range"]
                    f_color_p = "k"   
                    roles = ""

                rot_psi = get_rotation_matrix_3d_psi(state[3])
                rot_gam = get_rotation_matrix_3d_gam(state[4])
                rot_phi = get_rotation_matrix_3d_phi(state[5])
                if rend_faction == "mrm":
                    vecter = np.array([1, 0, 0])*5000/1000
                else:
                    vecter = np.array([1, 0, 0])*7000/1000
                    
                
                vecter = np.dot(rot_gam,vecter)
                vecter = np.dot(rot_psi,vecter)
                if rend_faction == "mrm":
                    vecter_gam = np.array([0, 0, 1])*2000/1000
                else:
                    vecter_gam = np.array([0, 0, 1])*3000/1000

                vecter_gam = np.dot(rot_phi,vecter_gam)
                vecter_gam = np.dot(rot_gam,vecter_gam)
                vecter_gam = np.dot(rot_psi,vecter_gam)
                if rend_faction == "mrm":
                    vecter_phi = np.array([0, 1, 0])*2000/1000
                else:
                    vecter_phi = np.array([0, 1, 0])*4000/1000
                vecter_phi = np.dot(rot_phi,vecter_phi)
                vecter_phi = np.dot(rot_gam,vecter_phi)
                vecter_phi = np.dot(rot_psi,vecter_phi)

                limit = [info["WINDOW_SIZE_lat"], info["WINDOW_SIZE_lon"], info["WINDOW_SIZE_alt"]]
                if time == 1:
                    obs_pos = [X,Y,Z]
                else:
                    obs_pos = [X[-1],Y[-1],Z[-1]]
                rotation = [rot_phi,rot_gam,rot_psi] 

                if plot_radar:
                    x,y,z = create_surfs(sensor_az,radar_range/1000,limit,obs_pos,rotation)
                    ax.plot_surface(x,y,z,color=f_color_p, antialiased=True,alpha=0.2,shade=False)
                    if rend_faction == "blue" or rend_faction == "red":
                        x,y,z = create_surfs(sensor_az,mrm_range/1000,limit,obs_pos,rotation)
                        ax.plot_surface(x,y,z,color=f_color_p, antialiased=True,alpha=0.05,shade=False)
                
                ax.text(obs_pos[0]+1, obs_pos[1]+1, obs_pos[2]+1, roles ,color=f_color_p,size=8)
                if rend_faction =="blue":
                    if state[7] == 0:
                        detect = " Detect"
                    else:
                        detect = " Nomal"
                    plot_text = format(time,"3.0f")+" "+ roles+" "+format(state[8],".5f")+" "+format(state[9],".2f")+\
                        " vel: " + format(state[6],".1f") + detect + " Azm: " + format(np.rad2deg(state[3]),".1f")
                    ax.text(10, 10+i*20, 10, plot_text, color=f_color_p,size=8)
                
                ax.plot([obs_pos[0],obs_pos[0]+vecter[0]],
                        [obs_pos[1],obs_pos[1]+vecter[1]],
                        [obs_pos[2],obs_pos[2]+vecter[2]],color = f_color_p)
                
                ax.plot([obs_pos[0]-vecter_phi[0],obs_pos[0]+vecter_phi[0]],
                                 [obs_pos[1]-vecter_phi[1],obs_pos[1]+vecter_phi[1]],
                                 [obs_pos[2]-vecter_phi[2],obs_pos[2]+vecter_phi[2]],color = f_color_p)
                ax.plot([obs_pos[0],obs_pos[0]+vecter_gam[0]],
                                 [obs_pos[1],obs_pos[1]+vecter_gam[1]],
                                 [obs_pos[2],obs_pos[2]+vecter_gam[2]],color = f_color_p)
                ax.plot(X, Y, Z,'--',color = f_color_p, linewidth = 1.0)
                ax.plot(X, Y, 0,'--',color = f_color_p, linewidth = 1.0)
                ax.plot([obs_pos[0],obs_pos[0]], [obs_pos[1],obs_pos[1]], [0,obs_pos[2]],'o-',color = f_color_p,linewidth = 1.0,markersize=0.5)

            ax.set_box_aspect((info["WINDOW_SIZE_lat"],info["WINDOW_SIZE_lon"],info["WINDOW_SIZE_alt"]))

            ax.set_xlabel("North [km]")
            ax.set_ylabel("East [km]")
            ax.set_zlabel("Altitude [km]")


            # ax.view_init(elev=10, azim=-45)
            ax.view_init(elev=90, azim=-90)
            ax.set_xlim(0,info["WINDOW_SIZE_lat"]/1000)
            ax.set_ylim(0,info["WINDOW_SIZE_lon"]/1000)
            ax.set_zlim(0,info["WINDOW_SIZE_alt"]/1000)
            ax.set_xticks(np.linspace(0,info["WINDOW_SIZE_lat"]/1000,9))
            ax.set_yticks(np.linspace(0,info["WINDOW_SIZE_lon"]/1000,9))
            ax.set_zticks(np.linspace(0,info["WINDOW_SIZE_alt"]/1000,9))
            
            # return ax
 

    def rend_3d(env,hist_pos,f_color , fig):
        
        time = hist_pos[:,0]
        num = hist_pos.shape[1]-1
        
        for i in range(num):
            # pos = np.vstack(hist_pos[:,i+1])
            traj_length = 600
            if hist_pos.shape[0] < traj_length:
                pos_temp = np.vstack(hist_pos[:,i+1].tolist())
            else:
                pos_temp = np.vstack(hist_pos[-traj_length:,i+1].tolist())
            
            if f_color == "b":
                rend_faction = "blue"
            elif f_color == "r":
                rend_faction = "red"
            else:
                rend_faction = "mrm"
                
            # pos = pos_temp[np.all(pos_temp[:,-1] != 0, axis=1),:]
            pos = pos_temp[~(pos_temp[:,-1] == 0)]
            fig1 = plt.figure(fig)

            # fig, ax = fig.add_subplot(111, projection='3d')
            ax = fig1.gca(projection='3d')
            if pos.size != 0:
                if pos.shape[0] < traj_length:
                    X = pos[:,0]/1000
                    Y = pos[:,1]/1000
                    Z = pos[:,2]/1000
                else:
                    X = pos[-traj_length:,0]/1000
                    Y = pos[-traj_length:,1]/1000
                    Z = pos[-traj_length:,2]/1000

                if rend_faction =="blue":
                    plot_obs = env.blue[i]
                    if env.blue[i].role == "decoy":
                        f_color_p = "g"
                        roles = "Decoy "+ "TGT: "+ str(plot_obs.tgt.id)
                    else:
                        f_color_p = "b"
                        roles = "Shooter "+ "TGT: "+ str(plot_obs.tgt.id)
                elif rend_faction =="red":
                    plot_obs = env.red[i]
                    f_color_p = "r"
                    roles = "red_"+str(i) +" "+ "TGT: "+ str(plot_obs.tgt.id)
                else:
                    plot_obs = env.mrm[i]
                    f_color_p = "k"   
                    roles = ""

                rot_psi = get_rotation_matrix_3d_psi(pos[-1,3])
                rot_gam = get_rotation_matrix_3d_gam(pos[-1,4])
                rot_phi = get_rotation_matrix_3d_phi(pos[-1,5])
                if plot_obs.faction == "mrm":
                    vecter = np.array([1, 0, 0])*5000/1000
                else:
                    vecter = np.array([1, 0, 0])*7000/1000
                    
                
                vecter = np.dot(rot_gam,vecter)
                vecter = np.dot(rot_psi,vecter)
                if plot_obs.faction == "mrm":
                    vecter_gam = np.array([0, 0, 1])*2000/1000
                else:
                    vecter_gam = np.array([0, 0, 1])*3000/1000

                vecter_gam = np.dot(rot_phi,vecter_gam)
                vecter_gam = np.dot(rot_gam,vecter_gam)
                vecter_gam = np.dot(rot_psi,vecter_gam)
                if plot_obs.faction == "mrm":
                    vecter_phi = np.array([0, 1, 0])*2000/1000
                else:
                    vecter_phi = np.array([0, 1, 0])*4000/1000
                vecter_phi = np.dot(rot_phi,vecter_phi)
                vecter_phi = np.dot(rot_gam,vecter_phi)
                vecter_phi = np.dot(rot_psi,vecter_phi)
                
                
                
                # ax.add_collection3d(mplot3d.art3d.Poly3DCollection(env.your_mesh.vectors))
                limit = [env.WINDOW_SIZE_lat, env.WINDOW_SIZE_lon, env.WINDOW_SIZE_alt]
                obs_pos = [X[-1],Y[-1],Z[-1]]
                rotation = [rot_phi,rot_gam,rot_psi] 
                
                x,y,z = create_surfs(plot_obs.sensor_az,plot_obs.radar_range/1000,limit,obs_pos,rotation)
                ax.plot_surface(x,y,z,color=f_color_p, antialiased=True,alpha=0.2,shade=False)
                if plot_obs.faction == "blue" or plot_obs.faction == "red":
                    x,y,z = create_surfs(plot_obs.sensor_az,plot_obs.mrm_range/1000,limit,obs_pos,rotation)
                    ax.plot_surface(x,y,z,color=f_color_p, antialiased=True,alpha=0.05,shade=False)
                
                ax.text(X[-1]+1, Y[-1]+1, Z[-1]+1, roles ,color=f_color_p,size=8)
                if rend_faction =="blue":
                    if plot_obs.detect_launch_ML:
                        detect = " Detect"
                    else:
                        detect = " Nomal"
                    plot_text = format(env.timer,"3d")+" "+ roles+" "+format(env.reward_num["blue_" + str(i)],".5f")+" "+format(env.rewards_total["blue_" + str(i)],".2f")+\
                        " vel: " + format(plot_obs.V,".1f") + detect + " Azm: " + format(np.rad2deg(plot_obs.psi),".1f")
                    ax.text(10, 10+i*20, 10, plot_text, color=f_color_p,size=8)
                
                ax.plot([X[-1],X[-1]+vecter[0]], [Y[-1],Y[-1]+vecter[1]], [Z[-1],Z[-1]+vecter[2]],color = f_color_p)
                ax.plot([X[-1]-vecter_phi[0],X[-1]+vecter_phi[0]],
                                 [Y[-1]-vecter_phi[1],Y[-1]+vecter_phi[1]],
                                 [Z[-1]-vecter_phi[2],Z[-1]+vecter_phi[2]],color = f_color_p)
                ax.plot([X[-1],X[-1]+vecter_gam[0]],
                                 [Y[-1],Y[-1]+vecter_gam[1]],
                                 [Z[-1],Z[-1]+vecter_gam[2]],color = f_color_p)
                ax.plot(X, Y, Z,'--',color = f_color_p, linewidth = 1.0)
                ax.plot(X, Y, 0,'--',color = f_color_p, linewidth = 1.0)
                ax.plot([X[-1],X[-1]], [Y[-1],Y[-1]], [0,Z[-1]],'o-',color = f_color_p,linewidth = 1.0,markersize=0.5)
                
                # ax.scatter(X, Y, Z,'o',color = f_color)
                # return body_x, body_y, body_z, trajectory, ground, height
                # plt.plot(pos[:,0],env.WINDOW_SIZE_lon-pos[:,1],'-',color = f_color)
                
        #     plt.grid('on')

            
        #     # max_range = np.array([env.WINDOW_SIZE_lat, env.WINDOW_SIZE_lon, env.WINDOW_SIZE_alt]).max() * 0.5
        #     # mid_x = (env.WINDOW_SIZE_lat) * 0.5
        #     # mid_y = (env.WINDOW_SIZE_lon) * 0.5
        #     # mid_z = (env.WINDOW_SIZE_alt) * 0.5
        #     # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #     # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #     # ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_box_aspect((env.WINDOW_SIZE_lat,env.WINDOW_SIZE_lon,env.WINDOW_SIZE_alt))
            # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 1.0, env.WINDOW_SIZE_alt/env.WINDOW_SIZE_lat, 1]))
            ax.set_xlabel("North [km]")
            ax.set_ylabel("East [km]")
            ax.set_zlabel("Altitude [km]")
        #     ax.set_xlabel("East [m]")
        #     ax.set_ylabel("North [m]")
        #     ax.set_zlabel("Altitude [m]")
        #     # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5
        #     # mid_x = (X.max()+X.min()) * 0.5
        #     # mid_y = (Y.max()+Y.min()) * 0.5
        #     # mid_z = (Z.max()+Z.min()) * 0.5
        #     # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #     # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #     # ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # ax.view_init(elev=10, azim=-45)
            ax.view_init(elev=90, azim=-90)
            ax.set_xlim(0,env.WINDOW_SIZE_lat/1000)
            ax.set_ylim(0,env.WINDOW_SIZE_lon/1000)
            ax.set_zlim(0,env.WINDOW_SIZE_alt/1000)
            ax.set_xticks(np.linspace(0,env.WINDOW_SIZE_lat/1000,9))
            ax.set_yticks(np.linspace(0,env.WINDOW_SIZE_lon/1000,9))
            ax.set_zticks(np.linspace(0,env.WINDOW_SIZE_alt/1000,9))
            # fig1.show()
            # fig.canvas.draw()
def create_surfs(az,plt_range,limit,obs_pos,rotation):
    x,y,z = surf_data(az,plt_range)
    x,y,z = surf_rot([x,y,z],rotation[0],rotation[1],rotation[2])
    X = clip_limmit(obs_pos[0]+x,limit[0])
    Y = clip_limmit(obs_pos[1]+y,limit[1])
    Z = clip_limmit(obs_pos[2]+z,limit[2])
    return X,Y,Z

def surf_data(az,radar_range):
    res = 20
    u = np.linspace(-az, az, res)

    v = np.linspace(np.pi/2-az, np.pi/2+az, res)
    x = radar_range * np.outer(np.cos(u), np.sin(v))
    y = radar_range * np.outer(np.sin(u), np.sin(v))
    z = radar_range * np.outer(np.ones(np.size(u)), np.cos(v))
    X = x.reshape([1, res**2])
    Y = y.reshape([1, res**2])
    Z = z.reshape([1, res**2])
    
    return X,Y,Z

def surf_rot(vec_surf,rot_phi,rot_gam,rot_psi):
    res = int(np.sqrt(vec_surf[0].shape[1]))
    vec_surf = np.vstack(vec_surf)
    vec_surf = np.dot(rot_phi,vec_surf)
    vec_surf = np.dot(rot_gam,vec_surf)
    vec_surf = np.dot(rot_psi,vec_surf)
    
    X = vec_surf[0].reshape([res,res])
    Y = vec_surf[1].reshape([res,res])
    Z = vec_surf[2].reshape([res,res])
    
    return X,Y,Z  

def clip_limmit(vec_surf,lim):
    temp_vec = vec_surf
    temp_vec = np.where(temp_vec <= 0, 0, temp_vec)
    Z = np.where(temp_vec >= lim/1000, lim/1000, temp_vec)

    
    return Z #X,Y,Z 