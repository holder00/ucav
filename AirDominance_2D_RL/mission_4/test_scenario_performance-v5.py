"""
Testing code for env.MyEnv
"""
import pygame
import argparse
import numpy as np
import gym
from myenv_2D_R_v5 import MyEnv
import imageio
import json
import os
import ray
import ray.rllib.agents.sac as sac

NUM_EPISODE = 30
ALGORITHM = 'sac'
ID = 2
MODEL_NAME = 'myenv-v5/checkpoints/trial_' + str(ID) + '_best/checkpoint_35001/checkpoint-35001'

video_dir = '../videos/mission_4/trial_' + str(ID) + '/nominal/'


def pause_for_debug():
    pause = False
    for event in pygame.event.get():
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                pause = True

    while pause:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    pause = False


def status_print(env, reward, done, step):
    # print(f'action_index: {action_index}: '
    #      f'fighter.action: {env.fighter.action},   jammer.action: {env.jammer.action}')
    print(f'\nStep = {step} -------------------------------------------------------------')
    print(f'(fighter_1.x, fighter_1.y)=({env.fighter_1.x:.1f}, {env.fighter_1.y:.1f}),   '
          f'(fighter_2.x, fighter_2.y)=({env.fighter_2.x:.1f}, {env.fighter_2.y:.1f}),   '
          f'(fighter_3.x, fighter_3.y)=({env.fighter_3.x:.1f}, {env.fighter_3.y:.1f}),   '
          f'(jammer_1.x, jammer_1.y)=({env.jammer_1.x:.1f}, {env.jammer_1.y:.1f}),   '
          f'(decoy_1.x, decoy_1.y)=({env.decoy_1.x:.1f}, {env.decoy_1.y:.1f}')
    print(f'fighter_1.alive ={int(env.fighter_1.alive)},   fighter_2.alive ={int(env.fighter_2.alive)},   '
          f'fighter_3.alive ={int(env.fighter_3.alive)}   '
          f'jammer_1.alive ={int(env.jammer_1.alive)},   decoy_1.alive ={int(env.decoy_1.alive)}   '
          f'jammer_1.on={int(env.jammer_1.on)}')
    print(f'\n(sam_1.x, sam_1.y)=({env.sam_1.x:.1f}, {env.sam_1.y:.1f}),   '
          f'sam_1.alive = {int(env.sam_1.alive)},   sam_1.firing_range = {env.sam_1.firing_range},   '
          f'sam_1.cooldown_on = {env.sam_1.cooldown_on},   '
          f' sam_1.cooldown_counter = {env.sam_1.cooldown_counter}')
    print(f'(target_1.x, target_1.y)=({env.target_1.x:.1f}, {env.target_1.y:.1f}),   '
          f'target_1.alive = {int(env.target_1.alive)}')
    print(f'\nreward = {reward}')

    if done:
        print(f'\n*************** {env.mission_id}: {env.mission_condition} ***************')
        print(f'   fighter_1.firing_range: {env.fighter_1.firing_range:.1f} km')
        print(f'   fighter_2.firing_range: {env.fighter_2.firing_range:.1f} km')
        print(f'   fighter_3.firing_range: {env.fighter_3.firing_range:.1f} km')
        print(f'   jammer_1.jam_range: {env.jammer_1.jam_range:.1f} km')
        print(f'   sam_1.firing_range: {env.sam_1.firing_range:.1f} km')
        print(f'   sam_1.jammed_firing_range: {env.sam_1.jammed_firing_range:.1f} km')


def make_video(video_name, frames):
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    filename = video_dir + video_name + '.gif'
    imageio.mimsave(filename, np.array(frames), fps=30)


def make_jason(env, video_name, info):
    filename = video_dir + video_name + '.json'

    results = dict()

    results['mission_id'] = env.mission_id
    results['mission_condition'] = env.mission_condition
    results['result'] = info['result']

    results['fighter_1'] = {'alive': env.fighter_1.alive,
                            'firing_range': env.fighter_1.firing_range}
    results['fighter_2'] = {'alive': env.fighter_2.alive,
                            'firing_range': env.fighter_2.firing_range}
    results['fighter_3'] = {'alive': env.fighter_3.alive,
                            'firing_range': env.fighter_3.firing_range}

    results['sam_1'] = {'alive': env.sam_1.alive,
                        'firing_range': env.sam_1.firing_range,
                        'cooldown_on': env.sam_1.cooldown_on,
                        'cooldown_max_count': env.sam_1.cooldown_max_count,
                        'cooldown_counter': env.sam_1.cooldown_counter}

    # print(json.dumps(results, ensure_ascii=False, indent=2))

    with open(filename, mode='wt', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def main():
    # Initialize ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Generate & Check environment
    env = MyEnv({})

    # Define trainer agent
    model_name = MODEL_NAME

    config = sac.DEFAULT_CONFIG.copy()
    config['env_config'] = {}
    config['num_gpus'] = 0
    config['Q_model']['fcnet_hiddens'] = [512, 512, 256]
    config['policy_model']['fcnet_hiddens'] = [512, 512, 256]

    agent = sac.SACTrainer(config=config, env=MyEnv)
    agent.restore(model_name)

    for idx in range(NUM_EPISODE):
        """ Initialization """
        observation = env.reset()
        done = False
        frames = []
        step = 0
        while True:
            action = agent.compute_action(observation)

            # ????????????step ??????
            observation, reward, done, info = env.step(action)

            # ????????????
            status_print(env, reward, done, step)

            # ?????????????????????????????????
            # shot = env.render(mode=args.render_mode)
            frames.append(env.render(mode=args.render_mode))

            # Space key???pause, ???????????????
            pause_for_debug()

            # Slow down rendering
            pygame.time.wait(10)

            # ??????????????????????????????
            if done:
                status_print(env, reward, done, step)
                video_name = ALGORITHM + '_' + info['result'] + '_' + str(idx)
                make_video(video_name, frames)
                make_jason(env, video_name, info)
                break

            step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
