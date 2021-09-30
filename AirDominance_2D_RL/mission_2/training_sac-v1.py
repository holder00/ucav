"""
rllib version of ppo combined with GAN planner
Need 'myenv.py' in the same directory, because rllib handling of custom environment
This 'myenv.py' is different version from stable_baselines version
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from myenv_2D_R_v1 import MyEnv

import ray
from ray.tune.logger import Logger, UnifiedLogger
import ray.rllib.agents.sac as sac
from ray.tune.logger import pretty_print

import tensorflow as tf
import numpy as np
import datetime
import os
import tempfile

PROJECT = 'myenv-v1-sac'
ID = 3
TRIAL = 'trial_' + str(ID)
NUM_WORKERS = 8
MAX_EPISODE = 1000000
N_EVAL_EPISODE = 1000
EVAL_FREQ = 1000


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def main():
    # Initialize ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Define trainer agent
    config = sac.DEFAULT_CONFIG.copy()
    config['env_config'] = {}
    config['num_gpus'] = 0
    config['num_workers'] = NUM_WORKERS
    config['num_cpus_per_worker'] = 1
    print(pretty_print(config))
    trainer = sac.SACTrainer(config=config,
                             env=MyEnv,
                             logger_creator=custom_log_creator(
                                 os.path.expanduser("./" + PROJECT + "/logs"), TRIAL))

    logdir = trainer.logdir
    print(f'\n********************** logdir = {logdir}\n')

    # Check trainer agent
    policy = trainer.get_policy()
    policy.model.base_model.summary()

    # Define evaluator agent
    eval_env = MyEnv({})
    # obs = eval_env.reset()

    # Train agent
    max_episode = MAX_EPISODE
    eval_freq = EVAL_FREQ
    n_eval_episode = N_EVAL_EPISODE
    best_success_count = -100
    best_checkpoint_dir = os.path.join('./' + PROJECT + '/checkpoints/', TRIAL + '_best')
    success_history = []
    f1_history = []
    f2_history = []
    s1_history = []
    s2_history = []
    s3_history = []
    iteration_history = []
    for i in range(max_episode):
        print(f'{i}th iteration is starting.')
        # Training
        result = trainer.train()
        # print(pretty_print(result))

        # Evaluation
        if i % eval_freq == 0:
            print(f'\n--------------- Evaluation results at {i}th iteration ---------------')
            print(pretty_print(result))
            total_return = 0
            success_count = 0
            f1_count = 0
            f2_count = 0
            s1_count = 0
            s2_count = 0
            s3_count = 0
            info = {}
            return_list = []
            for j in range(n_eval_episode):
                # Test the trained agent
                obs = eval_env.reset()
                done = False

                while not done:
                    action = trainer.compute_action(obs)
                    obs, reward, done, info = eval_env.step(action)
                    total_return += reward

                return_list.append(total_return)
                if (info['result'] == 'S1') or (info['result'] == 'S2') or (info['result'] == 'S3'):
                    success_count += 1

                if info['result'] == 'F1':
                    f1_count += 1
                if info['result'] == 'F2':
                    f2_count += 1

                if info['result'] == 'S1':
                    s1_count += 1
                if info['result'] == 'S2':
                    s2_count += 1
                if info['result'] == 'S3':
                    s3_count += 1

            print(f'\niteration {i} success_count: {success_count} / {n_eval_episode}')
            epsilon = 1e-5
            print(f'   f1:s1:s2={f1_count}:{s1_count}:{s2_count}, f2:s3={f2_count}:{s3_count}')
            f1 = f1_count / (f1_count + s1_count + s2_count + epsilon)
            f2 = f2_count / (f2_count + s3_count + epsilon)
            s1 = s1_count / (f1_count + s1_count + s2_count + epsilon)
            s2 = s2_count / (f1_count + s1_count + s2_count + epsilon)
            s3 = s3_count / (f2_count + s3_count + epsilon)

            print(f'   f1/(f1+s1+s2): {f1:.3f}')
            print(f'   f2/(f2+s3): {f2:.3f}')
            print(f'   s1/(f1+s1+s2): {s1:.3f}')
            print(f'   s2/(f1+s1+s2): {s2:.3f}')
            print(f'   s3/(f2+s3): {s3:.3f}')
            # print(f'return list: {return_list}')

            success_history.append(success_count / n_eval_episode)
            f1_history.append(f1)
            s1_history.append(s1)
            s2_history.append(s2)
            f2_history.append(f2)
            s3_history.append(s3)
            iteration_history.append(i)

            success_history_np = np.array(success_history)
            file_name = './' + PROJECT + '/learning_history/trial_' + str(ID)
            # np.savez('./learning_history', iteration_history, success_history)
            np.savez(file=file_name,
                     iteration_history=iteration_history,
                     success_history=success_history,
                     f1_history=f1_history,
                     f2_history=f2_history,
                     s1_history=s1_history,
                     s2_history=s2_history,
                     s3_history=s3_history)

            if success_count >= best_success_count:
                best_checkpoint = trainer.save(checkpoint_dir=best_checkpoint_dir)
                print(f'best checkpoint saved at {best_checkpoint}\n')
                best_success_count = success_count

            print(f'------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
