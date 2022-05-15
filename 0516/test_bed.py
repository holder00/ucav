# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 01:40:17 2021

@author: Takumi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 01:39:46 2021

@author: Takumi
"""

import argparse
import gym
import datetime
import os
import random
import tempfile
import numpy as np
import pickle

import ray
from ray import tune
from ray.tune.logger import Logger, UnifiedLogger, pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.examples.models.shared_weights_model import TF2SharedWeightsModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import ppo, PPOTrainer, PPOTFPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from environment_rllib_3d import MyEnv
from settings.initial_settings import *
from settings.reset_conditions import reset_conditions
#from modules.models import MyConv2DModel_v0B_Small_CBAM_1DConv_Share
from modules.models import DenseNetModelLarge
from tensorflow.keras.utils import plot_model
from modules.savers import save_conditions
from utility.result_env import render_env
from utility.terminate_uavsimproc import teminate_proc
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import cv2
import ctypes
import warnings

#UCAV.exeが起動している場合、プロセスキルする。
teminate_proc.UAVsimprockill(proc_name="UCAV.exe")

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
np.set_printoptions(precision=3, suppress=True)
PROJECT = "UCAV"
TRIAL_ID = 2
TRIAL = 'test_' + str(TRIAL_ID)
EVAL_FREQ = 10
CONTINUAL = True

def custom_log_creator(custom_path, custom_str):
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False)

ModelCatalog.register_custom_model('my_model', DenseNetModelLarge)

# config = {"env": MyEnv,
#           "num_workers": NUM_WORKERS,
#           "num_gpus": NUM_GPUS,
#           "num_cpus_per_worker": NUM_CPUS_PER_WORKER,
#           "num_sgd_iter": NUM_SGD_ITER,
#           "lr": LEARNING_RATE,
#           "gamma": GAMMA,  # default=0.99
#           "model": {"custom_model": "my_model"}
#           # "framework": framework
#           }  # use tensorflow 2
eval_env = MyEnv({})
policies = {
    #"blue_1": PolicySpec(config={"gamma": 0.99}),
    #"blue_2": PolicySpec(config={"gamma": 0.95}),
    "blue_0": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space, {}),
    "blue_1": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space, {}),
}
policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, episode, **kwargs):
    #print(agent_id,episode)
    #pol_id = policy_ids[agent_id]

    pol_id = agent_id
    return pol_id

# Instanciate the evaluation env

config = {"env": MyEnv,"num_gpus": 0,"num_workers": 0, "num_cpus_per_worker": 0,"num_gpus_per_worker": 0,
          "create_env_on_driver": True,"train_batch_size": 6000,"batch_mode": "complete_episodes",
          "multiagent": {"policies": policies,  "policy_mapping_fn": policy_mapping_fn}
         }
conditions_dir = os.path.join('./' + PROJECT + '/conditions/')

if not os.path.exists(conditions_dir):
    os.makedirs(conditions_dir)
save_conditions(conditions_dir)

# PPOTrainer()は、try_import_tfを使うと、なぜかTensorflowのeager modeのエラーになる。

trainer = ppo.PPOTrainer(config=config,
                         logger_creator=custom_log_creator(
                             os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))

if CONTINUAL:
    # Continual learning: Need to specify the checkpoint
    model_path = PROJECT + '/checkpoints/' + TRIAL + '/checkpoint_000001/checkpoint-1'
    trainer.restore(checkpoint_path=model_path)

# models_dir = os.path.join('./' + PROJECT + '/models/')
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
# text_name = models_dir + TRIAL + '.txt'
# with open(text_name, "w") as fp:
#     trainer.get_policy().model.base_model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
# png_name = models_dir + TRIAL + '.png'
# plot_model(trainer.get_policy().model.base_model, to_file=png_name, show_shapes=True)



# Define checkpoint dir
check_point_dir = os.path.join('./' + PROJECT + '/checkpoints/', TRIAL)
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
    
    
trainer.compute_single_action()
trainer.compute_action()
trainer.get_policy()