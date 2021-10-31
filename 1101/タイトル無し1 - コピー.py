# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:19:06 2021

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
from ray.rllib.agents.ppo import ppo
from ray.rllib.models import ModelCatalog
from battle_field_strategy_2D_v0 import BattleFieldStrategy
from environment_rllib import MyEnv

from settings.initial_settings import *
from settings.reset_conditions import reset_conditions
#from modules.models import MyConv2DModel_v0B_Small_CBAM_1DConv_Share
from modules.models import DenseNetModelLarge
from tensorflow.keras.utils import plot_model
from modules.savers import save_conditions

import tensorflow as tf

PROJECT = "UCAV"
TRIAL = "00"

def custom_log_creator(custom_path, custom_str):
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator



ray.init(ignore_reinit_error=True, log_to_driver=False)
config = {"env": MyEnv,"num_gpus": 0, "num_cpus_per_worker": 0}
trainer = ppo.PPOTrainer(config=config,logger_creator=custom_log_creator(
                                 os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))