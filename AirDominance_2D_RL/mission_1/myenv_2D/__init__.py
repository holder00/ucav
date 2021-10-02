from gym.envs.registration import register

register(
    id='myenv_2D-v0',
    entry_point='myenv_2D.envs:MyEnv'
)
