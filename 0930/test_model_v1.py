import gym
import numpy as np
from environment_rllib import MyEnv
from modules.models import DenseNetModelLargeShare_3


# def main():
# 環境の生成
env = MyEnv()
observations = env.reset()

# my_model = DenseNetModelLargeShare_3(env.observation_space,
#                                      env.action_space,
#                                      env.action_space.n,
#                                      {}, 'my_model')

while False:
    action_dict = {}
    for i in range(env.blue_num):
        action_dict['blue_' + str(i)] = env.action_space.sample()

    observations, rewards, dones, infos = env.step(action_dict)
    # print(observations)
    # print(f'env.steps: {env.steps}')
    # np.set_printoptions(precision=1)
    # print(f'red_force: {env.red.force}')
    # np.set_printoptions(precision=1)
    # print(f'blue_force: {env.blue.force}')
    print(f'dones: {dones}')
    np.set_printoptions(precision=3)
    print(f'observations:{observations}')
    np.set_printoptions(precision=3)
    print(f'rewards: {rewards}')
    print(f'infos: {infos}')

    # env.render()

    # エピソードの終了処理
    if dones['__all__']:
        # print(f'all done at {env.steps}')
        break


# if __name__ == '__main__':
#     for _ in range(1):
#         main()
