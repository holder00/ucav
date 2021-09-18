# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 22:33:18 2021

@author: Takumi
"""

from ray.tune.registry import register_env

def env_creator(env_config):
    return MyEnv(...)  # return an env instance

register_env("my_env", env_creator)
trainer = ppo.PPOTrainer(env="my_env")