"""
Testing code for env.MyEnv
"""
import pygame
import argparse
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import gym
import myenv_2D
import imageio
import json
from stable_baselines import PPO2, A2C, SAC

ALGORITHM = 'sac'  # 'ppo2' or 'a2c'
ID = 7
MODEL_NAME = 'myenv_2D-v0-sac/sac_32_w2_' + str(ID) + '/best_model.zip'
NUM_TRIALS = 3

video_dir = './videos/myenv_2D-v0-sac/'


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


def status_print(env, observation, reward, done, fighter_x0, fighter_y0, jammer_x0, jammer_y0):
    print(f'\n\n*************** Mission Summaries: {env.mission_condition} ***************')
    if reward > 0:
        print(f'Mission Succeeded for mission condition {env.mission_condition}')
    elif reward < 0:
        print('Mission Failed for mission condition {env.mission_condition}')
    else:
        raise Exception('Something is wrong')

    if env.jammer_1.on > 0.5:
        print('   Jammer_1 is used')
    else:
        print('   Jammer_1 is not used')

    print(f'\n--------------- Mission Conditions: {env.mission_condition} ---------------')
    print(f'   fighter_1.firing_range: {env.fighter_1.firing_range:.1f} km')
    print(f'   sam_1.firing_range: {env.sam_1.firing_range:.1f} km')
    print(f'   sam_1.jammed_firing_range: {env.sam_1.jammed_firing_range:.1f} km')
    print(f'   sam_1.(x,y): ({env.sam_1.x:.1f}, {env.sam_1.y}) km')

    print(f'\n--------------- Initial Conditions: {env.mission_condition} ---------------')
    print(f'fighter_1.initial_(x,y): ({fighter_x0:.1f}, {fighter_y0:.1f}) km,   '
          f'jammer_1.initial_(x,y): ({jammer_x0:.1f}, {jammer_y0:.1f}) km')

    print(f'\n--------------- End status: {env.mission_condition} ---------------')
    print(f'fighter_1.(x,y): ({env.fighter_1.x:.1f}, {env.fighter_1.y:.1f}) km,   '
          f'jammer_1.(x,y): ({env.jammer_1.x:.1f}, {env.jammer_1.y:.1f}) km')
    print(f'fighter_1.alive: {env.fighter_1.alive},   jammer_1.alive: {env.jammer_1.alive},   '
          f'jammer_1.on: {env.jammer_1.on},   sam_1.alive: {env.sam_1.alive}')


def count_w_and_success(env, w_count, success_count, idx):
    w_count[idx] += 1
    if (env.sam.alive < .5) and (env.fighter.alive > .5) and (env.jammer.alive > .5):
        success_count[idx] += 1
    return w_count, success_count


def conunt_results(env, w_count, success_count):
    w_id = env.mission_condition
    if w_id == "w1":
        idx = 0
    elif w_id == "w2":
        idx = 1
    elif w_id == "w3":
        idx = 2
    elif w_id == "l1":
        idx = 3
    elif w_id == "l2":
        idx = 4
    else:
        raise Exception('Error!')

    w_count, success_count = count_w_and_success(env, w_count, success_count, idx)
    return w_count, success_count


def make_video(video_name, frames):
    filename = video_dir + video_name + '.gif'
    # imageio.mimsave(filename, np.array(frames), 'GIF', fps=10)
    imageio.mimsave(filename, np.array(frames), fps=30)


def make_jason(env, model_name, video_name, fighter_x0, fighter_y0, jammer_x0, jammer_y0, reward, idx):
    filename = video_dir + video_name + '.json'

    results = dict()
    results['idx'] = idx
    results['model_name'] = model_name
    if (reward > 0) and (env.jammer_1.on < .5):
        results['Mission'] = 'Success without using Jammer'
    elif (reward > 0) and (env.jammer_1.on > .5):
        results['Mission'] = 'Success with using Jammer'
    else:
        results['Mission'] = 'Failed'

    results['mission_condition'] = env.mission_condition

    results['fighter_1'] = {'alive': env.fighter_1.alive,
                            'initial (x,y)': (fighter_x0, fighter_y0),
                            'final (x,y)': (env.fighter_1.x, env.fighter_1.y),
                            'firing_range': env.fighter_1.firing_range}

    results['jammer_1'] = {'alive': env.jammer_1.alive,
                           'initial (x,y)': (jammer_x0, jammer_y0),
                           'final (x,y)': (env.jammer_1.x, env.jammer_1.y),
                           'jam_range': env.jammer_1.jam_range,
                           'jamming': env.jammer_1.on}

    results['sam_1'] = {'alive': env.sam_1.alive,
                        '(x,y)': (env.sam_1.x, env.sam_1.y),
                        'firing_range': env.sam_1.firing_range,
                        'jammed_firing_range': env.sam_1.jammed_firing_range}

    # print(json.dumps(results, ensure_ascii=False, indent=2))

    with open(filename, mode='wt', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def main():
    model_dir = './models'
    model_name = model_dir + '/' + MODEL_NAME

    """ Generate & Check environment """
    env_name = 'myenv_2D-v0'
    env = gym.make(env_name)
    # env = gym.wrappers.Monitor(env, "./videos", force=True)  # For video making

    """ Vectorize environment """
    # Unnecessary to vectorize environment
    # env = DummyVecEnv([lambda: env])

    """ Load model and set environment """
    if ALGORITHM == 'ppo2':
        model = PPO2.load(model_name)
    elif ALGORITHM == 'a2c':
        model = A2C.load(model_name)
    elif ALGORITHM == 'sac':
        model = SAC.load(model_name)
    else:
        raise Exception('Load error.  Specify proper name')

    for idx in range(NUM_TRIALS):
        """ Initialization """
        observation = env.reset()
        frames = []

        """ Save some initial values """
        fighter_x0 = env.fighter_1.x
        fighter_y0 = env.fighter_1.y
        jammer_x0 = env.jammer_1.x
        jammer_y0 = env.jammer_1.y

        while True:
            action_index, _ = model.predict(observation)

            # 環境を１step 実行
            observation, reward, done, _ = env.step(action_index)

            # 環境の描画とビデオ録画
            # shot = env.render(mode=args.render_mode)
            frames.append(env.render(mode=args.render_mode))

            # Space keyでpause, デバッグ用
            pause_for_debug()

            # Slow down rendering
            pygame.time.wait(10)

            # エピソードの終了処理
            if done:
                status_print(env, observation, reward, done, fighter_x0, fighter_y0, jammer_x0, jammer_y0)
                video_name = ALGORITHM + '_' + env.mission_condition + '-' + str(idx)
                make_video(video_name, frames)
                make_jason(env, model_name, video_name,
                           fighter_x0, fighter_y0, jammer_x0, jammer_y0, reward, idx)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
