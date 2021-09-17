# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:25:21 2021

@author: Takumi
"""

#from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench import Monitor
from environment import MyEnv
import os
from glob import glob

def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    lates_modified_file_path = sorted(files, key = lambda files: files[1])[-1]
    return lates_modified_file_path[0] 

log_dir = './log/'
env = MyEnv()
# env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log")

# pathhh = get_latest_modified_file_path("./save_weights")
# model = PPO2.load(pathhh)
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
print('start learning')
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./save_weights/', name_prefix='rl_model')
model.learn(total_timesteps=10000000, callback=checkpoint_callback)
print('finish learning')
