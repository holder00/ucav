# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 00:02:58 2021

@author: Takumi
"""
from environment import MyEnv
from ray.rllib.agents import ppo
import ray
from ray.rllib import agents

def env_creator(env_name):
    if env_name == 'MyEnv-v0':
        from environment import MyEnv as env
    elif env_name == 'MyEnv-v1':
        from custom_gym.envs.custom_env import MyEnv as env
    else:
        raise NotImplementedError
    return env

env_name = 'MyEnv-v0'
config = {
    # Whatever config settings you'd like...
    }
trainer = agents.ppo.PPOTrainer(
    env=env_creator(env_name), 
    config=config)
max_training_episodes = 10000
while True:
    results = trainer.train()
    # Enter whatever stopping criterion you like
    if results['episodes_total'] >= max_training_episodes:
        break
print('Mean Rewards:\t{:.1f}'.format(results['episode_reward_mean']))