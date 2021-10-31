# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:51:09 2021

@author: Takumi
"""

import gym
import numpy as np
from environment_rllib_3d import MyEnv
from modules.models import DenseNetModelLargeShare_3
from result_env import render_env
import matplotlib.pyplot as plt
import ctypes
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
def getkey(key):
    return(bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000))
# def main():
# 環境の生成
env = MyEnv()
observations = env.reset()

# my_model = DenseNetModelLargeShare_3(env.observation_space,
#                                      env.action_space,
#                                      env.action_space.n,
#                                      {}, 'my_model')
env_blue_pos = [0]
env_red_pos = [0]
env_mrm_pos = [0]
step_num = 0
fig = plt.figure(1)
ESC = 0x1B          # ESCキーの仮想キーコード

plt.ion()           # 対話モードオン
while True:
    action_dict = {}
    for i in range(env.blue_num):
        action_dict['blue_' + str(i)] = env.action_space.sample()

    observations, rewards, dones, infos = env.step(action_dict)

    env_blue_pos_temp, env_red_pos_temp, env_mrm_pos_temp= render_env.copy_from_env(env)

    env_blue_pos.append(env_blue_pos_temp)
    env_red_pos.append(env_red_pos_temp)
    env_mrm_pos.append(env_mrm_pos_temp)
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
    
    if step_num == 0:
        del env_blue_pos[0]
        del env_red_pos[0]
        del env_mrm_pos[0]

    hist_blue_pos = np.vstack(env_blue_pos)
    hist_red_pos = np.vstack(env_red_pos)
    hist_mrm_pos = np.vstack(env_mrm_pos)
    plt.clf()
    render_env.rend_3d(env,hist_blue_pos,"b",1)
    render_env.rend_3d(env,hist_red_pos,"r",1)
    render_env.rend_3d(env,hist_mrm_pos,"k",1)
    plt.pause(.05)
    
    step_num = step_num + 1
    
    
    # エピソードの終了処理
    if dones['__all__'] or getkey(ESC):
        # print(f'all done at {env.steps}')
        break




# if __name__ == '__main__':
#     for _ in range(1):
#         main()
