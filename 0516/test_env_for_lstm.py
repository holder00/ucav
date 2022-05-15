# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:11:44 2021

@author: Takumi
"""
import gym
import mmap
import time
# from datetime import datetime
from struct import *
import ctypes
from multiprocessing import Process
import subprocess
import copy
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
Kernel32 = ctypes.windll.Kernel32
mutex = Kernel32.CreateMutexA(0,0,"Global/UAV_MUTEX")


# import gym
# from gym import spaces
import numpy as np
# import math
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from settings.initial_settings import *  # Import settings  # For training
# from settings.test_settings import *  # Import settings  # For testing
# from settings.reset_conditions import reset_conditions
# from modules.resets import reset_red, reset_blue, reset_block
# from modules.observations import get_observation
# from modules.rewards import get_reward
# import cv2
from status.get_obs_3d import get_obs
# from status.reward_calc import reward_calc
from status.init_space import init_space
from UAV.uav_3d import uav_3d
from weapon.missile_3d import missile_3d
from utility.terminate_uavsimproc import teminate_proc
from utility.harf_angle import harf_angle
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_psi
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_gam
from utility.get_rotation_matrix_3d import get_rotation_matrix_3d_phi
# from SENSOR.SENSOR_MAIN import sensor
# from numba import jit


class MyEnv(MultiAgentEnv):
    def __init__(self, num):
        self.agents = [MockEnv(25) for _ in range(2)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    def reset(self):
        self.resetted = True
        self.dones = set()
        return {"blue_"+str(i): a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = 1.0, 1.0, False, {}
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

