"""
Compute success ratio by using learned model
"""
import argparse
from myenv_2D_R_v5 import MyEnv
import ray
import ray.rllib.agents.sac as sac

N_EVAL_EPISODES = 1000

ALGORITHM = 'sac'
ID = 2
MODEL_NAME = 'myenv-v5/checkpoints/trial_' + str(ID) + '_best/checkpoint_35001/checkpoint-35001'


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
