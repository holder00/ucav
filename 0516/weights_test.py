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
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c import a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from environment_rllib_3d1 import MyEnv
#from test_env_for_lstm import MyEnv
from settings.initial_settings import *
from settings.reset_conditions import reset_conditions

from tensorflow.keras.utils import plot_model
from modules.savers import save_conditions
from utility.result_env import render_env
from utility.terminate_uavsimproc import teminate_proc
from utility.latest_learned_file_path import latest_learned_file_path
from utility.read_wright_weights import save_weights
from utility.read_wright_weights import reload_weights
from utility.save_logs import save_logs
from utility.save_logs import save_hists
from utility.save_logs import save_env_info

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
EVAL_FREQ = 1
CONTINUAL = True
NUM_EVAL = 1
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

#ModelCatalog.register_custom_model('my_model', MyRNNUAVClass)

eval_env = MyEnv()
policies_own = {
    "blue_0": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space,
               {"model":{"vf_share_layers": False,"use_lstm": True,"max_seq_len": 200},
               "exploration_config": {"type": "StochasticSampling","random_timesteps":0},"explore":True,}),
    "blue_1": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space,
               {"model":{"vf_share_layers": False,"use_lstm": True,"max_seq_len": 200},
               "exploration_config": {"type": "StochasticSampling","random_timesteps":0},"explore":True,}),
    "red_0": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space,
              {"model":{"vf_share_layers": False,"use_lstm": True,"max_seq_len": 200},"explore":False,}),
    "red_1": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space,
              {"model":{"vf_share_layers": False,"use_lstm": True,"max_seq_len": 200},"explore":False,}),
}
policies_enem = {
    "red_0": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space,
              {"model":{"vf_share_layers": False,"use_lstm": True,"max_seq_len": 200},"explore":False,}),
    "red_1": (PPOTFPolicy, eval_env.observation_space, eval_env.action_space,
              {"model":{"vf_share_layers": False,"use_lstm": True,"max_seq_len": 200},"explore":False,}),
}
# policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, episode, **kwargs):
    print(agent_id,episode)
    #pol_id = policy_ids[agent_id]

    pol_id = agent_id
    return pol_id

# Instanciate the evaluation env
config_own = ppo.DEFAULT_CONFIG.copy()
config_own = {"env": MyEnv,"num_gpus": 0,"num_workers": 0, "num_cpus_per_worker": 0,"num_gpus_per_worker": 0,
          "train_batch_size": 600*5*10,
          "batch_mode": "complete_episodes",
          "gamma":0.995, "lr": 2.5e-4,"shuffle_sequences": True,
          "observation_space":eval_env.observation_space,"action_space":eval_env.action_space,
          "sgd_minibatch_size": 600, "num_sgd_iter":20,
          "multiagent": {"policies": policies_own,  "policy_mapping_fn": policy_mapping_fn}
         }
config_enem = ppo.DEFAULT_CONFIG.copy()
config_enem = {"env": MyEnv,"num_gpus": 0,"num_workers": 0, "num_cpus_per_worker": 0,"num_gpus_per_worker": 0,
          "train_batch_size": 600*5*10,
          "batch_mode": "complete_episodes",
          "gamma":0.995, "lr": 2.5e-4,"shuffle_sequences": True,
          "observation_space":eval_env.observation_space,"action_space":eval_env.action_space,
          "sgd_minibatch_size": 600, "num_sgd_iter":20,
          "multiagent": {"policies": policies_enem,  "policy_mapping_fn": policy_mapping_fn}
         }

res_name = "test"
conditions_dir = os.path.join('./' + PROJECT + '/conditions/')

if not os.path.exists(conditions_dir):
    os.makedirs(conditions_dir)
save_conditions(conditions_dir)

# PPOTrainer()は、try_import_tfを使うと、なぜかTensorflowのeager modeのエラーになる。

trainer = ppo.PPOTrainer(config=config_own,
                         logger_creator=custom_log_creator(
                             os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))

adversary = ppo.PPOTrainer(config=config_enem,
                         logger_creator=custom_log_creator(
                             os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))

if CONTINUAL:
    # Continual learning: Need to specify the checkpoint
    # model_path = PROJECT + '/checkpoints/' + TRIAL + '/checkpoint_000197/checkpoint-197'
    model_path = latest_learned_file_path('./UCAV/checkpoints/test_2/*')
    
    trainer.restore(checkpoint_path=model_path)
    save_weights("blue_0",trainer)
    save_weights("blue_1",trainer)

reload_weights(policy_id="red_0",trainer=trainer,set_policy_id="blue_0")
reload_weights(policy_id="red_1",trainer=trainer,set_policy_id="blue_1")
save_weights("red_0",trainer)
save_weights("red_1",trainer)


models_dir = os.path.join('./' + PROJECT + '/models/')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
for j in range(2):
    text_name = models_dir + TRIAL + "blue_"+str(j) +'.txt'
    with open(text_name, "w") as fp:
        trainer.get_policy("blue_"+str(j)).model.base_model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
    png_name = models_dir + TRIAL + '.png'
    plot_model(trainer.get_policy("blue_"+str(j)).model.base_model, to_file=png_name, show_shapes=True)



# Define checkpoint dir
check_point_dir = os.path.join('./' + PROJECT + '/checkpoints/', TRIAL)
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)