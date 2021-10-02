import math

# For training
PROJECT = 'battle_field_strategy_2D_v0'
ALGORITHM = 'ppo'  # 'ppo'
TRIAL_ID = 2
TRIAL = 'fwd_' + str(TRIAL_ID)
EVAL_FREQ = 25  # Also save checkpoint
NUM_EVAL = 1
CONTINUAL = False  # 継続学習？

# For testing
TEST_ID = 0
NUM_TEST = 1
MAKE_ANIMATION = False
MODE = 'rgb'

# For rllib setting
NUM_GPUS = 0
NUM_SGD_ITER = 10
LEARNING_RATE = 5e-5  # default: 5e-5 (fwd_20501以外)

# For environment
GRID_SIZE = 10
CHANNEL = 7  # number of maps to use
DT = 1
INIT_POS_RANGE_RATIO = 0.3  # agents initial position ratio in grid_size

RED_TOTAL_FORCE = 500
BLUE_TOTAL_FORCE = 490

NUM_BLOCK_MAX = 10
NUM_BLOCK_MIN = 2

# For Agents training
GAMMA = 0.99  # default=0.99
NUM_CPUS_PER_WORKER = 0  # 1 for GCP, 0 for local exec
NUM_WORKERS = 8  # Default=8 fro local exec
MAX_STEPS = 100
NUM_RED_MAX = 8  # <= 8
NUM_RED_MIN = 1  # >=1
NUM_BLUE_MAX = 7  # <= 10
NUM_BLUE_MIN = 1  # >=1
RED_MIN_FORCE = 50  # default
BLUE_MIN_FORCE = RED_MIN_FORCE  # default


# For Agents test
NUM_CPUS_PER_WORKER = 0
NUM_WORKERS = 1
MAX_STEPS = 1
NUM_RED_MAX = 6
NUM_RED_MIN = 6
NUM_BLUE_MAX = 7
NUM_BLUE_MIN = 7
#RED_MIN_FORCE = math.floor(RED_TOTAL_FORCE / NUM_RED_MAX)
#BLUE_MIN_FORCE = math.floor(BLUE_TOTAL_FORCE / NUM_BLUE_MAX)
RED_MIN_FORCE = 50
BLUE_MIN_FORCE = RED_MIN_FORCE


# |For training & test
RED_MAX_FORCE = RED_TOTAL_FORCE  # for observation_normalize
BLUE_MAX_FORCE = BLUE_TOTAL_FORCE  # for observation_normalize

RED_MIN_EFFICIENCY = 0.6
RED_MAX_EFFICIENCY = 0.6
BLUE_MIN_EFFICIENCY = 0.6
BLUE_MAX_EFFICIENCY = 0.6

ALIVE_CRITERIA = RED_MIN_FORCE * 0.1
