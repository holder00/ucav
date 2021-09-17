# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:25:39 2021

@author: Takumi
"""

#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines import DQN
from environment import MyEnv
from stable_baselines.common.callbacks import CheckpointCallback
import cv2
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import csv
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np

def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    lates_modified_file_path = sorted(files, key = lambda files: files[1])[-1]
    return lates_modified_file_path[0] 

env_blue = MyEnv()
env_red = MyEnv()
# env = DummyVecEnv([lambda: env])
# import reward_calc

# reward_calc.reward_calc()
#env = MyEnv()

test_num = 1000
win = 0
# pathを指定して任意の重みをロードする
# model = PPO2.load("./save_weights/rl_model_150000_steps")
# pathhh = get_latest_modified_file_path("./save_weights")
# model = PPO2.load(pathhh)
model_blue = PPO2(MlpPolicy, env_blue, verbose=1, tensorboard_log="log")
model_red = PPO2(MlpPolicy, env_red, verbose=1, tensorboard_log="log")
winrate = np.tile([0.0,0.0],(test_num,1))
ER = np.tile([0.0,0.0],(test_num,1))
blue_num = env_blue.blue_num
red_num = env_red.blue_num
hist_obs = [0]*test_num
hist_action = [0]*test_num
# 10回試行する
for i in range(test_num):
    print('start learning')
    # pathhh = get_latest_modified_file_path("./save_weights")
    # model = PPO2.load(pathhh)
    obs_blue = env_blue.reset()
    obs_red = env_red.reset()
    obs_red = obs_blue[[4,5,6,7,0,1,2,3],:]
    plt.figure(1)
    fig1 = plt.figure(1)
    plt.plot(winrate[0:i,0],winrate[0:i,1],'o-')
    plt.plot(ER[0:i,0],ER[0:i,1],'o-')
    plt.grid('on')
    plt.draw()
    
    while True:

        action_blue, _states_blue = model_blue.predict(obs_blue)
        action_red, _states_red = model_red.predict(obs_red)
        hist_obs[i] = obs_blue
        hist_action[i] = action_blue
        # print(action)
        obs_blue, rewards_blue, dones_blue, info_blue = env_blue.step(action_blue)
        obs_red, rewards_red, dones_red, info_red = env_red.step(action_red)
        env_blue.red = env_red.red
        env_red.blue = env_blue.blue
        obs_red = obs_blue[[4,5,6,7,0,1,2,3],:]
        # print(obs)
        env_blue.render()
        if dones_blue or cv2.waitKey(1) == 13:
            break
    if np.all(obs_blue[blue_num:(blue_num+red_num),2]==0):
        win = win + 1
    ER[i] = np.array([i+1,np.sum(obs_blue[0:blue_num,2])/np.sum(obs_blue[blue_num:(blue_num+red_num),2])])
    winrate[i] = np.array([i+1,win/(i+1)])
    plt.clf()
