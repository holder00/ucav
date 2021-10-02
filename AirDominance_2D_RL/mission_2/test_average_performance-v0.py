"""
Compute success ratio by using learned model
"""
import argparse
from myenv_2D_R_v1 import MyEnv
import ray
import ray.rllib.agents.sac as sac

N_EVAL_EPISODES = 1000

ALGORITHM = 'sac'
ID = 3
MODEL_NAME = 'myenv-v1-sac/checkpoints/trial_' + str(ID) + '_best/checkpoint_39001/checkpoint-39001'



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
    f1_count = 0
    f2_count = 0
    s1_count = 0
    s2_count = 0
    s3_count = 0

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

        if idx % 100 == 0:
            print(f'idx = {idx}')

    print(f'mission: {env.mission_id}, condition: {env.mission_condition}')
    print(f' - model: {model_name}')
    print(f' - success_count: {success_count} / {N_EVAL_EPISODES}')
    epsilon = 1e-5
    print(f'   f1:s1:s2 = {f1_count}:{s1_count}:{s2_count},   f2:s3 = {f2_count}:{s3_count}')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_mode", type=str, default="human", help="rgb, if training")
    args = parser.parse_args()

    main()
