# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:53:03 2021

@author: ookit1
"""
import gym
from gym import spaces
import numpy as np
class init_space:
    def action(env):
        # action_space = gym.spaces.Dict({"tgt_id": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                       "fire": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                       "vector_psi_x": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                         # "vector_psi_y": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                       "vector_gam_x": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                        # "vector_gam_y": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                       "velocity": gym.spaces.Box(low=-1,high=1,shape=(1,))})
        action_space = gym.spaces.Dict({"tgt_id": gym.spaces.Discrete(env.red_num),
                                              "fire": gym.spaces.Discrete(2),
                                              "vector_psi_x": gym.spaces.Box(low=-1,high=1,shape=(1,)),
                                                # "vector_psi_y": gym.spaces.Box(low=-1,high=1,shape=(1,)),
                                               "vector_gam_x": gym.spaces.Box(low=-1,high=1,shape=(1,)),
                                                # "vector_gam_y": gym.spaces.Box(low=-1,high=1,shape=(1,)),
                                              "velocity": gym.spaces.Box(low=-1,high=1,shape=(1,))})
        # action_space = gym.spaces.Dict({"tgt_id": gym.spaces.Discrete(env.red_num),
        #                                 #"role": gym.spaces.Discrete(2), #role 0:shooter 1:decoy
        #                                 "vector_psi_x": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                 "vector_psi_y": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                 "vector_gam_x": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                 "vector_gam_y": gym.spaces.Box(low=-1,high=1,shape=(1,)),
        #                                 "velocity": gym.spaces.Box(low=0,high=1,shape=(1,)),
        #                                 })  
        return action_space

    def observation(env):
        env.obs_dict = gym.spaces.Dict({"hitpoint": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "mrm_num": gym.spaces.Discrete(3),
                                    "inrange": gym.spaces.Discrete(2),
                                    "detect": gym.spaces.Discrete(2),
                                    "tgt_psi_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "tgt_psi_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "tgt_gam_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "tgt_gam_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "distances": gym.spaces.Box(low=0, high=env.WINDOW_SIZE_lat*3,shape=(env.blue_num+env.red_num,)),
                                    "distances": gym.spaces.Box(low=-1, high=1,shape=(env.blue_num+env.red_num,)),
                                    # "phi_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "phi_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "tgt_id": gym.spaces.Discrete(env.red_num),
                                    # "fire": gym.spaces.Discrete(2),
                                    # "tgt_id": gym.spaces.Box(low=-1,high=1,shape=(1,)),
                                    "fire": gym.spaces.Box(low=-1,high=1,shape=(1,)),
                                    # # "role": gym.spaces.Discrete(2),
                                    "vector_psi_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "vector_psi_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "vector_gam_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "vector_gam_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "velocity": gym.spaces.Box(low=-1, high=1, shape=(1,))
                                    })

        # obs_dict = gym.spaces.Dict({"hitpont": gym.spaces.Discrete(2)})

        env.obs_dict_red = gym.spaces.Dict({"hitpoint": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "mrm_num": gym.spaces.Discrete(2),
                                    # "inrange": gym.spaces.Discrete(2),
                                    # "detect": gym.spaces.Discrete(2),
                                    # "tgt_psi_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "tgt_psi_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "tgt_gam_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    # "tgt_gam_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "psi_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "psi_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "gam_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "gam_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "velocity": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "distances": gym.spaces.Box(low=-1, high=1,shape=(1,))})
        
        env.obs_dict_own_mrm = gym.spaces.Dict({"hitpoint": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "parent_id": gym.spaces.Discrete(env.blue_num+1),
                                    "inrange": gym.spaces.Discrete(2),
                                    "tgt_id": gym.spaces.Discrete(env.red_num+1),
                                    # "detect": gym.spaces.Discrete(2),
                                    "self_pos_x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "self_pos_z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "velocity": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                                    "status": gym.spaces.Discrete(2),
                                    # "distances": gym.spaces.Box(low=-1, high=1,shape=(1,)),
                                    })
        obs_blue = {}
        obs_red = {}
        obs_own_mrm = {}
        obs_blue["self"] = env.obs_dict
        for i in range(env.blue_num -1):
            obs_blue["blue_" + str(i)] = env.obs_dict
        for i in range(env.red_num):
            obs_red["red_" + str(i)] = env.obs_dict_red
        for i in range(2*env.blue_num):
            obs_own_mrm["own_mrm_" + str(i)] = env.obs_dict_own_mrm
        observation_space = gym.spaces.Dict({"blues": gym.spaces.Dict(obs_blue),"reds": gym.spaces.Dict(obs_red),"mrms": gym.spaces.Dict(obs_own_mrm)})

        return observation_space