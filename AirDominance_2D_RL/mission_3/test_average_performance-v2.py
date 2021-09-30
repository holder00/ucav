"""
Compute success ratio by using learned model
"""
import argparse
from myenv_2D_R_v2 import MyEnv
import ray
import ray.rllib.agents.sac as sac

N_EVAL_EPISODES = 1000

ALGORITHM = 'sac'
ID = 0
if ID == 0:
    MODEL_NAME = 'myenv-v2-sac/checkpoints/trial_' + str(ID) + '_best/checkpoint_26001/checkpoint-26001'
elif ID == 2:
    MODEL_NAME = 'myenv-v1-sac/checkpoints/trial_' + str(ID) + '_best/checkpoint_7001/checkpoint-7001'
else:
    raise Exception('Specify proper model!')


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

    agent = sac.SACTrainer(config=config, env=MyEnv)
    agent.restore(model_name)

    success_count = 0

    for idx in range(N_EVAL_EPISODES):
        """ Initialization """
        observation = env.reset()
        done = False

        while True:
            action = agent.compute_action(observation)

            # 環境を１step 実行
            observation, reward, done, info = env.step(action)

            # エピソードの終了処理
            if done:
                break

        if info['result'] == 'success':
            success_count += 1

        if idx % 100 == 0:
            print(f'idx = {idx}')

    print(f'mission: {env.mission_id}, condition: {env.mission_condition}')
    print(f' - model: {model_name}')
    print(f' - success_count: {success_count} / {N_EVAL_EPISODES}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
