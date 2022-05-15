# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:41:42 2022

@author: OokiT1
"""
"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import argparse
import os
import random

import ray
from ray import tune
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

tf1, tf, tfv = try_import_tf()

# parser = argparse.ArgumentParser()

# parser.add_argument("--num-agents", type=int, default=4)
# parser.add_argument("--num-policies", type=int, default=2)
# parser.add_argument("--num-cpus", type=int, default=0)
# parser.add_argument(
#     "--framework",
#     choices=["tf", "tf2", "tfe", "torch"],
#     default="tf",
#     help="The DL framework specifier.",
# )
# parser.add_argument(
#     "--as-test",
#     action="store_true",
#     help="Whether this script should be run as a test: --stop-reward must "
#     "be achieved within --stop-timesteps AND --stop-iters.",
# )
# parser.add_argument(
#     "--stop-iters", type=int, default=200, help="Number of iterations to train."
# )
# parser.add_argument(
#     "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
# )
# parser.add_argument(
#     "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
# )

# if __name__ == "__main__":
# args = parser.parse_args()
N_cpu = 0
N_policies = 2
N_agents = 2
framework = "tf"
stop_reward = 200
stop_timesteps = 100000
stop_iters = 150.0
as_test = False
ray.init(num_cpus=N_cpu or None)

# Register the models to use.
if framework == "torch":
    mod1 = mod2 = TorchSharedWeightsModel
elif framework in ["tfe", "tf2"]:
    mod1 = mod2 = TF2SharedWeightsModel
else:
    mod1 = SharedWeightsModel1
    mod2 = SharedWeightsModel2
ModelCatalog.register_custom_model("model1", mod1)
ModelCatalog.register_custom_model("model2", mod2)

# Each policy can have a different configuration (including custom model).
def gen_policy(i):
    config = {
        "model": {
            "custom_model": ["model1", "model2"][i % 2],
        },
        "gamma": random.choice([0.95, 0.99]),
    }
    return PolicySpec(config=config)

# Setup PPO with an ensemble of `num_policies` different policies.
policies = {"policy_{}".format(i): gen_policy(i) for i in range(N_policies)}
policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    pol_id = random.choice(policy_ids)
    return pol_id

config = {
    "env": MultiAgentCartPole,
    "env_config": {
        "num_agents": N_agents,
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "num_sgd_iter": 10,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    },
    "framework": framework,
}
stop = {
    "episode_reward_mean": stop_reward,
    "timesteps_total": stop_timesteps,
    "training_iteration": stop_iters,
}

results = tune.run("PPO", stop=stop, config=config, verbose=1)

if as_test:
    check_learning_achieved(results, stop_reward)
ray.shutdown()
