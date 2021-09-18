# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:00:02 2021

@author: Takumi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:45:04 2021

@author: Takumi
"""
# TensorFlowのバージョンの警告をフィルタリング
import os

# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings

# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


from environment import MyEnv
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from gym.envs.registration import register

def make_env(env_id, rank, seed=0):
    """
    マルチプロセス環境のユーティリティ関数。

    :param env_id: (str) 環境ID
    :param num_env: (int) サブプロセスに含める環境の数
    :param seed: (int) RNGの初期シード
    :param rank: (int) サブプロセスのインデックス
    """
    def _init():
        env = gym.make(env_id)
        # 【重要】環境ごとに異なるシードを使用してください
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


register(
    id='UCAV-v0',
    entry_point='environment:MyEnv'
)

# 環境
env_id = 'UCAV-v0'

# プロセス数
PROCESSES_TO_TEST = [4]

# RLアルゴリズムは不安定になることが多いため、実験を複数回実行
# （https://arxiv.org/abs/1709.06560を参照）
NUM_EXPERIMENTS = 3 
TRAIN_STEPS = 5000

# 評価するエピソード数
EVAL_EPS = 20

# 評価するアルゴリズム
ALGO = PPO2

# # エージェントを評価するための環境の生成
# eval_env = DummyVecEnv([lambda: gym.make(env_id)])

nproc= 2

reward_averages = []
reward_std = []
training_times = []
total_procs = 0
# envs= [make_env(env_id, seed) for seed in range(nproc)]
env = SubprocVecEnv([make_env(env_id, i) for i in range(nproc)])
# envs= SubprocVecEnv(envs)

# for n_procs in PROCESSES_TO_TEST:
#    total_procs += n_procs
#    print('Running for n_procs = {}'.format(n_procs))
#    if n_procs == 1:
#        # プロセスが1つしかない場合は、マルチプロセッシングを使用する必要はない
#        train_env = DummyVecEnv([lambda: gym.make(env_id)])
#    else:
#        # プロセスを起動するために「spawn」を使用。詳細については、ドキュメントを参照
#        train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)], start_method='spawn')
#        print("0000000000")
#    rewards = []
#    times = []

#    for experiment in range(NUM_EXPERIMENTS):
#        # 結果のばらつきのため、いくつかの実験を実行することをお勧めする
#        train_env.reset()
#        model = ALGO('MlpPolicy', train_env, verbose=0)
#        start = time.time()
#        model.learn(total_timesteps=TRAIN_STEPS)
#        times.append(time.time() - start)
#        mean_reward = evaluate(model, eval_env, num_episodes=EVAL_EPS)
#        rewards.append(mean_reward)

#    # 重要：サブプロセスを使用する場合は、サブプロセスを閉じることを忘れないように。
#    # 多くの実験を実行すると、メモリの問題が発生する可能性がある
#    train_env.close()
#    reward_averages.append(np.mean(rewards))
#    reward_std.append(np.std(rewards))
#    training_times.append(np.mean(times))