"""
Testing code for env.MyEnv
"""
import pygame
import os
import argparse
import numpy as np
import tensorflow as tf
import gym
from myenv_2D_R_v1 import MyEnv

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
          f'(jammer_1.x, jammer_1.y)=({env.jammer_1.x:.1f}, {env.jammer_1.y:.1f}),   '
          f'(decoy_1.x, decoy_1.y)=({env.decoy_1.x:.1f}, {env.decoy_1.y:.1f}')
    print(f'fighter_1.alive ={int(env.fighter_1.alive)},   fighter_2.alive ={int(env.fighter_2.alive)},   '
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

    result_type = info['result']
    print(f'\nresult type = {result_type}')
    print(f'reward {reward}')
    # if result_type == 'F1':
    #    pause_at_done()

    return info, reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="rgb", help="rgb, if training")
    args = parser.parse_args()

    success = 0
    fail = 0

    total_F1 = 0
    total_F2 = 0
    total_S1 = 0
    total_S2 = 0
    total_S3 = 0

    for _ in range(NUM_EPISODE):
        info, reward = main()
        # pause_at_done()

        if info['result'] == 'F1':
            total_F1 += 1
            fail += 1
        elif info['result'] == 'F2':
            total_F2 += 1
            fail += 1
        elif info['result'] == 'S1':
            total_S1 += 1
            success += 1
        elif info['result'] == 'S2':
            total_S2 += 1
            success += 1
        elif info['result'] == 'S3':
            total_S3 += 1
            success += 1

    print('\n\n------- Summaries of Missions by random actions -------')
    print(f'   Success: {success} / {NUM_EPISODE},   {success / NUM_EPISODE * 100:.2f} [%]')
    print(f'   F1={total_F1},   F2={total_F2},   S1={total_S1},   S2={total_S2},   S3={total_S3}')
    epsilon = 1e-5
    print(f'   F1/(F1+S1+S2)={total_F1 / (total_F1 + total_S1 + total_S2 + epsilon) * 100:.2f} %,'
          f'   S1/(F1+S1+S2)={total_S1 / (total_F1 + total_S1 + total_S2 + epsilon) * 100:.2f} %,'
          f'   S2/(F1+S1+S2)={total_S2 / (total_F1 + total_S1 + total_S2 + epsilon) * 100:.2f} %')
    print(f'   F2/(F2+S3)={total_F2 / (total_F2 + total_S3 + epsilon) * 100:.2f} %,'
          f'   S3/(F2+S3)={total_S3 / (total_F2 + total_S3 + epsilon) * 100:.2f} %')
