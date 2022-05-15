# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:51:09 2021

@author: Takumi
"""
import time
from datetime import datetime
import gym
import numpy as np
# from environment_rllib_3d_light import MyEnv
from environment_rllib_3d3 import MyEnv
# from modules.models import DenseNetModelLargeShare_3
from utility.result_env import render_env
from utility.terminate_uavsimproc import teminate_proc
import matplotlib.pyplot as plt
import matplotlib
import ctypes
import warnings
from matplotlib.animation import FuncAnimation
import cv2
from utility.save_logs import save_hists
from utility.save_logs import save_env_info
import pickle
# from PIL import Image
# from ray.rllib.agents.ppo import ppo


np.set_printoptions(precision=3, suppress=True)
real_time = 0
# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + " @ " +"============sim start=============")

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

#UCAV.exeが起動している場合、プロセスキルする。
teminate_proc.UAVsimprockill(proc_name="UCAV_vec.exe")

def getkey(key):
    return(bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000))
# def main():
# 環境の生成

env = MyEnv({"eval":True})
observations = env.reset()
# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + " @ " +"============sim reset=============")
# my_model = DenseNetModelLargeShare_3(env.observation_space,
#                                      env.action_space,
#                                      env.action_space.n,
#                                      {}, 'my_model')
# env_blue_pos = [0]
# env_red_pos = [0]
# env_mrm_pos = [0]
step_num = 0
fig = plt.figure(1,figsize=(8.0, 6.0))
record_mode = 1

if record_mode == 0:
    file_name = "test_num"
    video = cv2.VideoWriter(file_name+'.mp4',0x00000020,20.0,(800,600))
ESC = 0x1B          # ESCキーの仮想キーコード

plt.ion()           # 対話モードオン
save_env_info(env)
f = open("info"+".pkl",mode="rb")
info = pickle.load(f)
f.close()
total_time = 0
while True:
    perf_time = time.perf_counter()
    # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + " @ "+  "============sim steps:"+str(step_num)+ " start"+ "=============")
    action_dict = {}
    # for i in range(env.blue_num):
    #     action_dict['blue_' + str(i)] = env.action_space.sample()
    # for i in range(env.red_num):
    #     action_dict['red_' + str(i)] = env.action_space.sample()
    observations, rewards, dones, infos = env.step(action_dict)

    # env_blue_pos_temp, env_red_pos_temp, env_mrm_pos_temp= render_env.copy_from_env(env)
    env_blue_pos_temp_mod, env_red_pos_temp_mod, env_mrm_pos_temp_mod = render_env.copy_from_env_mod(env)
    if env.timer == 1:
        env_blue_pos_mod = env_blue_pos_temp_mod
        env_red_pos_mod = env_red_pos_temp_mod
        env_mrm_pos_mod = env_mrm_pos_temp_mod
    else:
        env_blue_pos_mod = np.vstack([env_blue_pos_mod,env_blue_pos_temp_mod])
        env_red_pos_mod = np.vstack([env_red_pos_mod,env_red_pos_temp_mod])
        env_mrm_pos_mod = np.vstack([env_mrm_pos_mod,env_mrm_pos_temp_mod])
    if env.timer>1:
        total_time += (time.perf_counter()-perf_time)
        print(env.timer,time.perf_counter()-perf_time,total_time/(env.timer-1))
        
    # env_blue_pos.append(env_blue_pos_temp)
    # env_red_pos.append(env_red_pos_temp)
    # env_mrm_pos.append(env_mrm_pos_temp)
    # print(observations)
    # print(f'env.steps: {env.steps}')
    # np.set_printoptions(precision=1)
    # print(f'red_force: {env.red.force}')
    # np.set_printoptions(precision=1)
    # print(f'blue_force: {env.blue.force}')
    # print(f'dones: {dones}')
    # np.set_printoptions(precision=3)
    # print(f'observations:{observations}')
    # np.set_printoptions(precision=3)
    # print(f'rewards: {rewards}')
    # print(f'infos: {infos}')

    # env.render()

    # if step_num == 0:
    #     del env_blue_pos[0]
    #     del env_red_pos[0]
    #     del env_mrm_pos[0]

    # hist_blue_pos = np.vstack(env_blue_pos)
    # hist_red_pos = np.vstack(env_red_pos)
    # hist_mrm_pos = np.vstack(env_mrm_pos)
    plt.clf()
    # body_x, body_y, body_z, trajectory, ground, height = render_env.rend_3d(env,hist_blue_pos,"b",1)
    # render_env.rend_3d(env,hist_blue_pos,"b",1)
    # render_env.rend_3d(env,hist_red_pos,"r",1)
    # render_env.rend_3d(env,hist_mrm_pos,"k",1)
    render_env.rend_3d_mod2(env.timer,env_blue_pos_mod,"b",1,info,plot_radar=False)
    render_env.rend_3d_mod2(env.timer,env_red_pos_mod,"r",1,info,plot_radar=False)
    render_env.rend_3d_mod2(env.timer,env_mrm_pos_mod,"k",1,info,plot_radar=False)
    plt.subplots_adjust(left=-0.1,right=1.1,bottom=-0.1,top=1.1)
    fig.canvas.draw()


    plt.pause(.05)
    if record_mode == 0:
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        video.write(img.astype('uint8'))
    elif record_mode == 1:
        pass
    # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + " @ "+  "============sim steps:"+str(step_num)+ " end"+ "=============")
    # print("clac time: "+ str(time.perf_counter() - perf_time))
    step_num = step_num + 1

    # エピソードの終了処理
    if dones['__all__'] or getkey(ESC):
        #UCAV.exeが起動している場合、プロセスキルする。
        teminate_proc.UAVsimprockill(proc_name="UCAV.exe")
        # print(f'all done at {env.steps}')
        save_hists("blue",0,env_blue_pos_mod)
        save_hists("red",0,env_red_pos_mod)
        save_hists("mrm",0,env_mrm_pos_mod)
        if record_mode == 0:
                video.release()
        break




# if __name__ == "__main__":
