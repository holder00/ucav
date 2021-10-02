"""
Testing code for env.MyEnv
"""
import pygame
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
from myenv_2D_R_v2 import MyEnv

NUM_EPISODE = 10000


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


def pause_at_done():
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
          f'sam_1.alive = {int(env.sam_1.alive)}')
    print(f'(target_1.x, target_1.y)=({env.target_1.x:.1f}, {env.target_1.y:.1f}),   '
          f'target_1.alive = {int(env.target_1.alive)}')
    print(f'\nreward = {int(reward)}')

    if done:
        print(f'\n*************** {env.mission_id}: {env.mission_condition} ***************')
        print(f'   fighter_1.firing_range: {env.fighter_1.firing_range:.1f} km')
        print(f'   fighter_2.firing_range: {env.fighter_2.firing_range:.1f} km')
        print(f'   fighter_3.firing_range: {env.fighter_3.firing_range:.1f} km')
        print(f'   jammer_1.jam_range: {env.jammer_1.jam_range:.1f} km')
        print(f'   sam_1.firing_range: {env.sam_1.firing_range:.1f} km')
        print(f'   sam_1.jammed_firing_range: {env.sam_1.jammed_firing_range:.1f} km')


def main():
    env = MyEnv({})

    total_reward = 0
    step = 0
    while True:
        # print(f'step {step}')
        # ランダムアクションの選択
        actions = env.action_space.sample()
        # actions = 0 * np.ones(env.action_space.shape)

        # 環境を１step 実行
        observation, reward, done, info = env.step(actions)
        total_reward += reward
        if args.render_mode == 'human':
            # print(f'\naction is selected at {env.steps}')
            status_print(env, reward, done, step)
            pass

        # 環境の描画
        shot = env.render(mode=args.render_mode)

        # エピソードの終了処理
        if done:
            # pause_at_done()
            break

        # Space keyでpause, デバッグ用
        pause_for_debug()

        step += 1

    if env.mission_id == 'mission_2':
        result_type = info['result']
        print(f'\nresult type = {result_type}')
    elif env.mission_id == 'mission_3':
        mission_result = info['result']
        print(f'\n   result={mission_result},   reward={reward}')

    #    pause_at_done()

    return info, reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    total_reward = 0
    for _ in range(NUM_EPISODE):
        info, reward = main()
        total_reward += reward
        pause_at_done()

    print('\n\n------- Summaries of Missions by random actions -------')
    print(f'   total_reward = {total_reward}')
