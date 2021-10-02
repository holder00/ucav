from gym.envs.registration import register

register(
    id='myenv_2D_R-v1',
    entry_point='myenv_2D_R_v1.envs:MyEnv'
)
