# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:25:39 2021

@author: Takumi
"""
import gym
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines import DQN
from environment import MyEnv
from stable_baselines.common.callbacks import CheckpointCallback
import cv2
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import csv
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
from stable_baselines.bench import Monitor
from gym.envs.registration import register
from ray.rllib.agents import ppo
from result_env import render_env

import os
import tensorflow as tf
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    lates_modified_file_path = sorted(files, key = lambda files: files[1])[-1]
    return lates_modified_file_path[0] 

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128, 128],
                                                          vf=[128, 128, 128, 128])],
                                           feature_extraction="mlp")



env_id = 'UCAV-v0'

if env_id in gym.envs.registration.registry.env_specs.copy():
    del gym.envs.registration.registry.env_specs[env_id]
if not env_id in gym.envs.registration.registry.env_specs.copy():
    # del gym.envs.registration.registry.env_specs[env_id]
    register(id=env_id,entry_point='environment:MyEnv')


# env = MyEnv()
# env = DummyVecEnv([lambda: env])
# env = DummyVecEnv([lambda: env])
# import reward_calc
env = gym.make(env_id)
# reward_calc.reward_calc()
#env = MyEnv()
log_dir = './log/'


test_num = 1000
win = 0
# pathを指定して任意の重みをロードする
# model = PPO2.load("./save_weights/rl_model_150000_steps")
# pathhh = get_latest_modified_file_path("./save_weights")
# model = PPO2.load(pathhh)

winrate = np.tile([0.0,0.0],(test_num,1))
ER = np.tile([0.0,0.0],(test_num,1))
blue_num = env.blue_num
red_num = env.red_num
hist_obs = [0]*test_num
hist_action = [0]*test_num
rewards_total = np.tile([0.0,0.0],(test_num,1))
# 10回試行する
for i in range(test_num):
    print('start learning')
    pathhh = get_latest_modified_file_path("./save_weights")
    print(pathhh)
    model = PPO2.load(pathhh)
    # model = A2C.load(pathhh)
    # model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
    # model = ppo.PPOTrainer(MlpPolicy, env)
    obs = env.reset()
    
    plt.figure(1)
    # fig1 = plt.figure(1)
    # plt.plot(winrate[0:i,0],winrate[0:i,1],'o-')
    # plt.plot(ER[0:i,0],ER[0:i,1],'o-')
    plt.grid('on')
    
    # x, y = ts2xy(load_results(log_dir), 'timesteps')
    
    plt.draw()
    plt.figure(2)
    
    plt.plot(rewards_total[0:i,0],rewards_total[0:i,1],'o-')
    plt.grid('on')
    plt.draw()
    step_num = 0
    env_blue_pos = [0]
    env_red_pos = [0]
    env_mrm_pos = [0]
    while True:
        
        # print(obs)
        action, _states = model.predict(obs)
        hist_obs[i] = obs
        hist_action[i] = action
        # print(obs)
        obs, rewards, dones, info = env.step(action)

        env.render()
        env_blue_pos_temp, env_red_pos_temp, env_mrm_pos_temp= render_env.copy_from_env(env)
        env_blue_pos.append(env_blue_pos_temp)
        env_red_pos.append(env_red_pos_temp)
        env_mrm_pos.append(env_mrm_pos_temp)

        step_num = step_num + 1
        if dones or cv2.waitKey(1) == 13:
            break
    # if np.all(obs[blue_num:(blue_num+red_num),2]==0):
        win = win + 1
    # ER[i] = np.array([i+1,np.sum(obs[0:blue_num,2])/np.sum(obs[blue_num:(blue_num+red_num),2])])
    del env_blue_pos[0]
    del env_red_pos[0]
    del env_mrm_pos[0]
    fig1 = plt.figure(1)
    fig1.clf()
 
    hist_blue_pos = np.vstack(env_blue_pos)
    hist_red_pos = np.vstack(env_red_pos)
    hist_mrm_pos = np.vstack(env_mrm_pos)
    
    render_env.rend(env,hist_blue_pos,"b",1)
    render_env.rend(env,hist_red_pos,"r",1)
    render_env.rend(env,hist_mrm_pos,"k",1)
    
    fig2 = plt.figure(2)
    fig2.clf()  
    
    winrate[i] = np.array([i+1,win/(i+1)])
    rewards_total[i] = np.array([i+1,env.reward_total])
