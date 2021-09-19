# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:54:50 2021

@author: Takumi
"""

import gym
import numpy as np
from multiprocessing import Process, freeze_support
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, LstmPolicy
from stable_baselines import ACKTR
from stable_baselines import PPO2
from stable_baselines import A2C
from gym.envs.registration import register
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.bench import Monitor


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

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128, 128],
                                                          vf=[128, 128, 128, 128])],
                                           feature_extraction="mlp")

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


env_id = 'UCAV-v0'
log_dir = './log/'
# if env_id in gym.envs.registration.registry.env_specs.copy():
#     del gym.envs.registration.registry.env_specs[env_id]
# if not env_id in gym.envs.registration.registry.env_specs.copy():
#     # del gym.envs.registration.registry.env_specs[env_id]
#     register(id=env_id,entry_point='environment:MyEnv')

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        if env_id in gym.envs.registration.registry.env_specs.copy():
            del gym.envs.registration.registry.env_specs[env_id]
        if not env_id in gym.envs.registration.registry.env_specs.copy():
            # del gym.envs.registration.registry.env_specs[env_id]
            register(id=env_id,entry_point='environment:MyEnv')
        env = gym.make(env_id)
        env = Monitor(env, log_dir, allow_early_resets=True)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
# 
    # model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="log")
    # model = PPO2(CustomLSTMPolicy, env, verbose=1, tensorboard_log="log")
    # model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=dict(layers=[256, 256, 256, 256]), tensorboard_log="log")
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log")
    # model.learn(total_timesteps=25000)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./save_weights/', name_prefix='rl_model')
    model.learn(total_timesteps=10000000, callback=checkpoint_callback)

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()