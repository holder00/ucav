# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:25:39 2021

@author: Takumi
"""

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

from environment import MyEnv



# 10回試行する
for i in range(200):
    env = MyEnv()

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log")
#    checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./save_weights/', name_prefix='rl_model')
#    model.learn(total_timesteps=500, callback=checkpoint_callback)
    # pathを指定して任意の重みをロードする
    dir_temp = "./save_weights/rl_model_{}_steps"
    dir_r = dir_temp.format(500*i+500)
    
    model = PPO2.load(dir_r)
    
    print('start learning')
    print(i)
    obs = env.reset()

    for step in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones or cv2.waitKey(1) == 13:
            break

