"""
Compute success ratio by using learned model
"""
import pygame
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
import myenv_2D_R
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.vec_env import VecVideoRecorder
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import SAC


N_EVAL_EPISODES = 1000
ID = 2

ALGORITHM = 'sac'  # 'sac'
MODEL_NAME = 'sac_32_w2_' + str(ID) + '/best_model.zip'


def main():
    model_name = './models/myenv_2D_R-v0-sac/' + MODEL_NAME

    """ Generate & Check environment """
    env_name = 'myenv_2D_R-v0'
    env = gym.make(env_name)
    # env = Monitor(env, 'logs')
    # check_env(env)
    mission_probability = env.mission_probability

    """ Vectorize environment """
    # env = DummyVecEnv([lambda: env])

    """ Load model and set environment """
    if ALGORITHM == 'sac':
        model = SAC.load(model_name)
    else:
        raise Exception('Load error.  Specify proper name')

    """ Perform simulaion    """
    episode_rewards = evaluate_policy(model=model,
                                      env=env,
                                      n_eval_episodes=N_EVAL_EPISODES,
                                      return_episode_rewards=True)
    mean_reward = np.mean(np.array(episode_rewards[0]))
    std_reward = np.std(np.array(episode_rewards[0]))

    n_success = np.sum(np.array(episode_rewards[0]) >0)
    n_fail = np.sum(np.array(episode_rewards[0]) <= 0)
    if n_success + n_fail != N_EVAL_EPISODES:
        raise Exception('Error in counting successes and fails')

    """ Summarize results """
    print('==================== Summary of the results ====================')
    print(f'Model_name : {model_name}')
    print(f'Mission conditions = w1 : w2 : w3 : l1 : l2 = '
          f'{mission_probability[0][0]:.3f} : {mission_probability[0][1]:.3f} : '
          f'{mission_probability[0][2]:.3f} : '
          f'{mission_probability[0][3]:.3f} : {mission_probability[0][4]:.3f}')
    print(f'   Model is < {MODEL_NAME} >')
    print(f'   Number of success missions: {round(n_success)} / {N_EVAL_EPISODES},  '
          f'   Number of failed missions {round(n_fail)} / {N_EVAL_EPISODES}')
    print(f'   Mean rewards: {mean_reward:.2f} +/- {std_reward:.2f}')
    print(f'   Success percentage: {n_success / N_EVAL_EPISODES * 100: .2f} %')
    print(f'   Fail percentage: {n_fail / N_EVAL_EPISODES * 100: .2f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
