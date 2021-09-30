"""
This code may be called from test_env.py
Use relative observation, mission_1, mission_2, & mission_3
Gym environment for 2D reinforcement learning problem inspored by RAND corporations report:
    "AirDominance Through Machine Learning, A Preliminary Exploration of
    Artificial Intelligence-Assisted Mission Planning", RAND Corporation, 2020
        https://www.rand.org/pubs/research_reports/RR4311.html
"""
import gym
import numpy as np
import pygame
import math
import random
import matplotlib.pyplot as plt

"""
Define breakdown of mission conditions
"""
MISSION_IDS = ['mission_1', 'mission_2', 'mission_3', 'mission_4']
MISSION_IDS_RATIO = [0, 0, 0, 1]

MISSION_CONDITIONS_1 = ['w1', 'w2', 'w3', 'l1', 'l2']
MISSION_RATIO_1 = [0, 1, 0, 0, 0]

MISSION_CONDITIONS_2 = ['m1']
MISSION_RATIO_2 = [1]

MISSION_CONDITIONS_3 = ['f1']
MISSION_RATIO_3 = [1]

MISSION_CONDITIONS_4 = ['d1']
MISSION_RATIO_4 = [1]

MISSION_CONDITIONS = [MISSION_CONDITIONS_1, MISSION_CONDITIONS_2, MISSION_CONDITIONS_3, MISSION_CONDITIONS_4]
MISSION_RATIO = [MISSION_RATIO_1, MISSION_RATIO_2, MISSION_RATIO_3, MISSION_RATIO_4]

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
MAGENTA = (255, 0, 255, 255)
AQUA = (0, 255, 255, 255)
RADIUS = 10

RED_ZONE = (255, 0, 0, 64)
JAMMED_RED_ZONE = (255, 0, 0, 96)
GREEN_ZONE = (0, 255, 0, 64)
BLUE_ZONE = (0, 0, 255, 64)

RANGE_MULTIPLIER = 2


class Fighter:
    FIGHTER_MAX_HEADING_CHANGE_SEC = 20 * np.pi / 180  # rad/sec
    FIGHTER_MIN_FIRING_RANGE = 3.5 * RANGE_MULTIPLIER  # km
    FIGHTER_MAX_FIRING_RANGE = 10.5 * RANGE_MULTIPLIER  # km

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.previous_x = None  # km
        self.previous_y = None  # km
        self.firing_range = None  # km
        self.weapon_count = None
        self.alive = None
        self.long_weapon = None  # for mission_3

        # Additional states
        self.heading = None

        # Actions
        self.action = None

        # Specifications
        self.speed = 740.  # km/h
        self.max_heading_change_step = None
        self.max_heading_change_sec = self.FIGHTER_MAX_HEADING_CHANGE_SEC
        self.min_firing_range = self.FIGHTER_MIN_FIRING_RANGE
        self.max_firing_range = self.FIGHTER_MAX_FIRING_RANGE
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program

        # Rendering
        self.color = BLUE
        self.radius = RADIUS
        self.zone_color = BLUE_ZONE

        self.screen_x = None
        self.screen_y = None
        self.screen_firing_range = None
        self.surface = None
        self.surface_range = None


class Jammer:
    JAMMER_MAX_HEADING_CHANGE_SEC = 10 * np.pi / 180  # rad/sec

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.previous_x = None  # km
        self.previous_y = None  # km
        self.on = None
        self.alive = None

        # Additional states
        self.heading = None

        # Actions
        self.action = None

        # Specifications
        self.jam_range = 9. * RANGE_MULTIPLIER  # km
        self.speed = 740.  # km/h
        self.max_heading_change_step = None
        self.max_heading_change_sec = self.JAMMER_MAX_HEADING_CHANGE_SEC
        self.jam_effectiveness = 0.7  # Reduce ratio to the adversarial SAM range

        # Rendering
        self.color = GREEN
        self.radius = RADIUS
        self.zone_color = GREEN_ZONE

        self.screen_x = None
        self.screen_y = None
        self.screen_jam_range = None
        self.surface = None
        self.surface_range = None


class Decoy:
    DECOY_MAX_HEADING_CHANGE_SEC = 10 * np.pi / 180  # rad/sec

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.previous_x = None  # km
        self.previous_y = None  # km
        self.alive = None

        # Additional states
        self.heading = None

        # Actions
        self.action = None

        # Specifications
        self.speed = 740.  # km/h
        self.max_heading_change_step = None
        self.max_heading_change_sec = self.DECOY_MAX_HEADING_CHANGE_SEC

        # Rendering
        self.color = AQUA
        self.radius = RADIUS

        self.screen_x = None
        self.screen_y = None
        self.surface = None


class SAM:
    SAM_MIN_FIRING_RANGE = 5. * RANGE_MULTIPLIER  # km
    SAM_MAX_FIRING_RANGE = 10.5 * RANGE_MULTIPLIER  # km
    COOLDOWN_MAX_COUNT = 10  # Cool down time step, default=40

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.firing_range = None
        self.jammed_firing_range = None
        self.weapon_count = None
        self.alive = None
        self.previous_alive = None
        self.jammed = None

        self.firing_range_backup = self.firing_range  # For mission 4
        self.jammed_firing_range_backup = self.jammed_firing_range  # For mission 4
        self.cooldown_counter = None  # For mission 4
        self.cooldown_on = None

        # Additional states
        self.heading = None

        # Specifications
        self.min_firing_range = self.SAM_MIN_FIRING_RANGE
        self.max_firing_range = self.SAM_MAX_FIRING_RANGE
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program
        self.cooldown_max_count = self.COOLDOWN_MAX_COUNT

        # Rendering
        self.color = RED
        self.radius = RADIUS
        self.zone_color = RED_ZONE
        self.jammed_zone_color = JAMMED_RED_ZONE

        self.screen_x = None
        self.screen_y = None
        self.screen_firing_range = None
        self.screen_jammed_firing_range = None
        self.surface = None
        self.surface_range = None
        self.surface_jammed_range = None


class Target:
    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.alive = None

        # Additional states
        self.heading = None

        # Specifications

        # Rendering
        self.color = MAGENTA
        self.radius = RADIUS

        self.screen_x = None
        self.screen_y = None
        self.surface = None


class MyEnv(gym.Env):
    # For simulation
    SPACE_X = 100.  # km, positive size of battle space
    SPACE_Y = SPACE_X  # km
    # SPACE_OFFSET = 20.  # km, negative size of battle space default
    SPACE_OFFSET = 50.

    # For rendering
    WIDTH = 800
    HEIGHT = WIDTH

    # For screen shot
    SHOT_SHAPE = (80, 80)

    def __init__(self, env_config):
        super(MyEnv, self).__init__()
        self.width = self.WIDTH
        self.height = self.HEIGHT

        self.shot_shape = self.SHOT_SHAPE

        self.space_x = self.SPACE_X
        self.space_y = self.SPACE_Y
        self.space_offset = self.SPACE_OFFSET

        self.to_screen_x = self.width / (self.space_x + 2 * self.space_offset)
        self.to_screen_y = self.height / (self.space_y + 2 * self.space_offset)
        self.to_screen_offset = self.space_offset * self.to_screen_x

        self.blue_team = None
        self.red_team = None

        self.fighter_1 = Fighter()
        self.fighter_2 = Fighter()
        self.fighter_3 = Fighter()
        self.jammer_1 = Jammer()
        self.decoy_1 = Decoy()
        self.sam_1 = SAM()
        self.target_1 = Target()

        self.mission_id = None

        self.mission_ids = MISSION_IDS
        mission_ids_ratio = np.array(MISSION_IDS_RATIO)
        self.mission_ids_probability = mission_ids_ratio / np.sum(mission_ids_ratio)

        self.mission_conditions = MISSION_CONDITIONS
        self.mission_probability = []
        for ratio in MISSION_RATIO:
            self.mission_probability.append(ratio / np.sum(ratio))

        self.dt = 1 / self.fighter_1.speed * .5  # simulation step
        self.max_steps = int(self.space_x // (self.dt * self.fighter_1.speed) * 3)  # 300
        self.action_interval = 1
        self.resolution = self.dt * self.fighter_1.speed * self.action_interval  # resolution of simulations

        self.sam_1.firing_range_lower_bound = math.ceil((self.resolution * 2) / 0.3)  # 4km
        if self.sam_1.firing_range_lower_bound > self.jammer_1.jam_range:
            raise Exception('Error! Resolution is too big! Reduce action interval!')

        self.fighter_1.max_heading_change_step = \
            self.fighter_1.max_heading_change_sec * 3600 * self.dt  # rad/step
        self.fighter_2.max_heading_change_step = \
            self.fighter_2.max_heading_change_sec * 3600 * self.dt  # rad/step
        self.fighter_3.max_heading_change_step = \
            self.fighter_3.max_heading_change_sec * 3600 * self.dt  # rad/step

        self.jammer_1.max_heading_change_step = \
            self.jammer_1.max_heading_change_sec * 3600 * self.dt  # rad/step

        self.decoy_1.max_heading_change_step = \
            self.decoy_1.max_heading_change_sec * 3600 * self.dt  # rad/step

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.reset()

        if self.mission_id == 'mission_1':
            observation_low = np.array([0.] * 20, dtype=np.float32)
            observation_low[0:2] = -1
            observation_low[3:5] = -1
            observation_low[6:10] = -1

            observation_high = np.array([1.] * 20, dtype=np.float32)

        elif self.mission_id == 'mission_2':
            r = (self.space_x ** 2 + self.space_y ** 2) ** .5
            dr = (self.space_offset ** 2 + self.space_offset ** 2) ** .5

            observation_low = np.array([-1.] * 19, dtype=np.float32)
            observation_low[2] = - dr / r
            observation_low[5] = - dr / r
            observation_low[10] = - dr / r
            observation_low[11:] = 0

            observation_high = np.array([1.] * 19, dtype=np.float32)
            observation_high[2] = 1 + dr / r
            observation_high[5] = 1 + dr / r
            observation_high[10] = 1 + dr / r

        elif self.mission_id == 'mission_3':
            r = (self.space_x ** 2 + self.space_y ** 2) ** .5
            dr = (self.space_offset ** 2 + self.space_offset ** 2) ** .5

            observation_low = np.array([-1.] * 37, dtype=np.float32)
            observation_low[2] = - dr / r
            observation_low[5] = - dr / r
            observation_low[8] = - dr / r
            observation_low[21:24] = - dr / r
            observation_low[24:] = 0

            observation_high = np.array([1.] * 37, dtype=np.float32)
            observation_high[2] = 1 + dr / r
            observation_high[5] = 1 + dr / r
            observation_high[8] = 1 + dr / r
            observation_high[21:24] = 1 + dr / r

        elif self.mission_id == 'mission_4':
            r = (self.space_x ** 2 + self.space_y ** 2) ** .5
            dr = (self.space_offset ** 2 + self.space_offset ** 2) ** .5

            observation_low = np.array([-1.] * 58, dtype=np.float32)
            observation_low[2] = 0
            observation_low[5] = 0
            observation_low[8] = 0
            observation_low[21:24] = 0
            observation_low[24:37] = 0
            observation_low[39] = 0
            observation_low[52:55] = 0
            observation_low[55:] = 0

            observation_high = np.array([1.] * 58, dtype=np.float32)
            observation_high[2] = 1 + 2 * dr / r
            observation_high[5] = 1 + 2 * dr / r
            observation_high[8] = 1 + 2 * dr / r
            observation_high[21:24] = 1 + 2 * dr / r
            observation_high[39] = 1 + 2 * dr / r
            observation_high[52:55] = 1 + 2 * dr / r

        else:
            raise Exception('Not yet implemented')

        action_low = np.array([-1.] * len(self.blue_team), dtype=np.float32)
        action_high = np.array([1.] * len(self.blue_team), dtype=np.float32)

        # Define continuous action space and observation space
        self.action_space = gym.spaces.Box(low=action_low, high=action_high)
        self.observation_space = gym.spaces.Box(low=observation_low, high=observation_high)

    def reset_fighter(self, fighter):
        # Battle space coordinate
        fighter.heading = 0
        fighter.heading_sin = math.sin(fighter.heading)
        fighter.heading_cos = math.cos(fighter.heading)
        fighter.x = 0
        fighter.y = 0
        fighter.previous_x = fighter.x
        fighter.previous_y = fighter.y
        fighter.firing_range = 0
        fighter.weapon_count = 1
        fighter.alive = 1
        fighter.long_weapon = 0  # For mission_3

    def reset_render_fighter(self, fighter):
        # Render coordinate
        fighter.screen_x = fighter.x * self.to_screen_x + self.to_screen_offset
        fighter.screen_y = fighter.y * self.to_screen_y + self.to_screen_offset
        fighter.screen_firing_range = fighter.firing_range * self.to_screen_x

        # Fighter circle
        width = fighter.radius * 2
        height = fighter.radius * 2
        fighter.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=fighter.surface,
                           center=(fighter.radius, fighter.radius),
                           color=fighter.color, radius=fighter.radius)

        # Range circle
        width = fighter.screen_firing_range * 2
        height = fighter.screen_firing_range * 2
        fighter.surface_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=fighter.surface_range,
                           center=(fighter.screen_firing_range, fighter.screen_firing_range),
                           color=fighter.zone_color, radius=fighter.screen_firing_range)

    def reset_jammer(self, jammer):
        # Battle space coordinate
        jammer.heading = 0
        jammer.heading_sin = math.sin(jammer.heading)
        jammer.heading_cos = math.cos(jammer.heading)
        jammer.x = 0
        jammer.y = 0
        jammer.previous_x = jammer.x
        jammer.previous_y = jammer.y
        jammer.alive = 1
        jammer.on = 0

    def reset_render_jammer(self, jammer):
        # Render coordinate
        jammer.screen_x = jammer.x * self.to_screen_x + self.to_screen_offset
        jammer.screen_y = jammer.y * self.to_screen_y + self.to_screen_offset
        jammer.screen_jam_range = jammer.jam_range * self.to_screen_x

        # Jammer circle
        width = jammer.radius * 2
        height = jammer.radius * 2
        jammer.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=jammer.surface,
                           center=(jammer.radius, jammer.radius),
                           color=jammer.color, radius=jammer.radius)

        # Range circle
        width = jammer.screen_jam_range * 2
        height = jammer.screen_jam_range * 2
        jammer.surface_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=jammer.surface_range,
                           center=(jammer.screen_jam_range, jammer.screen_jam_range),
                           color=jammer.zone_color, radius=jammer.screen_jam_range)

    def reset_decoy(self, decoy):
        # Battle space coordinate
        decoy.heading = 0
        decoy.heading_sin = math.sin(decoy.heading)
        decoy.heading_cos = math.cos(decoy.heading)
        decoy.x = 0
        decoy.y = 0
        decoy.previous_x = decoy.x
        decoy.previous_y = decoy.y
        decoy.alive = 1

    def reset_render_decoy(self, decoy):
        # Render coordinate
        decoy.screen_x = decoy.x * self.to_screen_x + self.to_screen_offset
        decoy.screen_y = decoy.y * self.to_screen_y + self.to_screen_offset

        # Jammer circle
        width = decoy.radius * 2
        height = decoy.radius * 2
        decoy.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=decoy.surface,
                           center=(decoy.radius, decoy.radius),
                           color=decoy.color, radius=decoy.radius)

    def reset_sam(self, sam):
        # Battle space coordinate
        sam.heading = 0
        sam.heading_sin = math.sin(sam.heading)
        sam.heading_cos = math.cos(sam.heading)
        sam.x = 0
        sam.y = 0
        sam.firing_range = 0
        sam.jammed_firing_range = 0
        sam.weapon_count = 1
        sam.alive = 1
        sam.previous_alive = sam.alive
        sam.jammed = 0
        sam.firing_range_backup = sam.firing_range
        sam.jammed_firing_range_backup = sam.firing_range_backup
        sam.cooldown_counter = 0
        sam.cooldown_on = 0

    def reset_render_sam(self, sam):
        # Render coordinate
        sam.screen_x = sam.x * self.to_screen_x + self.to_screen_offset
        sam.screen_y = sam.y * self.to_screen_y + self.to_screen_offset
        sam.screen_firing_range = sam.firing_range * self.to_screen_x
        sam.screen_jammed_firing_range = sam.jammed_firing_range * self.to_screen_x

        # SAM circle
        width = sam.radius * 2
        height = sam.radius * 2
        sam.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=sam.surface,
                           center=(sam.radius, sam.radius),
                           color=sam.color, radius=sam.radius)

        # Range circle
        width = sam.screen_firing_range * 2
        height = sam.screen_firing_range * 2
        sam.surface_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=sam.surface_range,
                           center=(sam.screen_firing_range, sam.screen_firing_range),
                           color=sam.zone_color, radius=sam.screen_firing_range)

        # Jammed range circle
        width = sam.screen_jammed_firing_range * 2
        height = sam.screen_jammed_firing_range * 2
        sam.surface_jammed_range = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(surface=sam.surface_jammed_range,
                           center=(sam.screen_jammed_firing_range, sam.screen_jammed_firing_range),
                           color=sam.jammed_zone_color, radius=sam.screen_jammed_firing_range)

    def reset_target(self, target):
        # Battle space coordinate
        target.heading = 0
        target.heading_sin = math.sin(target.heading)
        target.heading_cos = math.cos(target.heading)
        target.x = 0
        target.y = 0
        target.alive = 1

    def reset_render_target(self, target):
        # Render coordinate
        target.screen_x = target.x * self.to_screen_x + self.to_screen_offset
        target.screen_y = target.y * self.to_screen_y + self.to_screen_offset

        # SAM circle
        width = target.radius * 2
        height = target.radius * 2
        target.surface = pygame.Surface((width, height))
        pygame.draw.circle(surface=target.surface,
                           center=(target.radius, target.radius),
                           color=target.color, radius=target.radius)

    def condition_w1(self, fighter, jammer, sam):
        sam_firing_range_list = np.arange(jammer.jam_range + self.resolution,
                                          sam.max_firing_range + self.resolution,
                                          self.resolution)
        if sam_firing_range_list[-1] > sam.max_firing_range:
            sam_firing_range_list = sam_firing_range_list[:-1]

        sam.firing_range = np.random.choice(sam_firing_range_list[:-1])
        sam.jammed_firing_range = sam.firing_range * jammer.jam_effectiveness

        fighter_firing_range_index = np.where(sam_firing_range_list > sam.firing_range)[0]
        fighter_firing_range_list = sam_firing_range_list[fighter_firing_range_index]
        fighter.firing_range = np.random.choice(fighter_firing_range_list)

        condition = (sam.max_firing_range >= sam.firing_range > jammer.jam_range) and \
                    (fighter.max_firing_range >= fighter.firing_range > sam.firing_range) and \
                    (fighter.firing_range >= sam.firing_range + self.resolution)

        if not condition:
            raise Exception('Error in condition_w1_generator')

    def condition_w2(self, fighter, jammer, sam):
        sam_firing_range_list = np.arange(sam.firing_range_lower_bound,
                                          jammer.jam_range,
                                          self.resolution)
        sam.firing_range = np.random.choice(sam_firing_range_list[1:])
        sam.jammed_firing_range = sam.firing_range * jammer.jam_effectiveness

        cond = (sam_firing_range_list < sam.firing_range) & \
               (sam_firing_range_list > sam.jammed_firing_range + self.resolution)
        fighter_firing_range_list_index = np.where(cond)[0]
        fighter_firing_range_index = np.random.choice(fighter_firing_range_list_index)
        fighter.firing_range = sam_firing_range_list[fighter_firing_range_index]

        condition = (jammer.jam_range > sam.firing_range >
                     fighter.firing_range > sam.jammed_firing_range) and \
                    (fighter.firing_range >= sam.jammed_firing_range + self.resolution)

        if not condition:
            raise Exception('Error in condition_w2_generator')

    def condition_w3(self, fighter, jammer, sam):
        sam_firing_range_list = np.arange(sam.firing_range_lower_bound,
                                          jammer.jam_range,
                                          self.resolution)
        sam.firing_range = np.random.choice(sam_firing_range_list[:-1])
        sam.jammed_firing_range = sam.firing_range * jammer.jam_effectiveness

        fighter_firing_range_index = np.where(sam_firing_range_list > sam.firing_range)[0]
        fighter_firing_range_list = sam_firing_range_list[fighter_firing_range_index]
        fighter.firing_range = np.random.choice(fighter_firing_range_list)

        condition = (jammer.jam_range > sam.firing_range) and \
                    (fighter.max_firing_range > fighter.firing_range > sam.firing_range) and \
                    (fighter.firing_range >= sam.firing_range + self.resolution)

        if not condition:
            raise Exception('Error in condition_w3_generator')

    def condition_l1(self, fighter, jammer, sam):
        fighter_firing_range_list = np.arange(0, fighter.max_firing_range, self.resolution)
        fighter.firing_range = np.random.choice(fighter_firing_range_list)

        sam_firing_range_list = np.arange(
            np.max([fighter.firing_range, jammer.jam_range]) + self.resolution,
            sam.max_firing_range + self.resolution,
            self.resolution)
        sam.firing_range = np.random.choice(sam_firing_range_list)
        sam.jammed_firing_range = sam.firing_range * jammer.jam_effectiveness

        condition = (sam.firing_range > jammer.jam_range) and \
                    (sam.firing_range > fighter.firing_range)

        if not condition:
            raise Exception('Error in condition_l1_generator')

    def condition_l2(self, fighter, jammer, sam):
        sam_firing_range_list = np.arange(self.resolution * 3, jammer.jam_range, self.resolution)
        sam.firing_range = np.random.choice(sam_firing_range_list)
        sam.jammed_firing_range = sam.firing_range * jammer.jam_effectiveness

        fighter_firing_range_list = np.arange(0, sam.jammed_firing_range, self.resolution)
        fighter.firing_range = np.random.choice(fighter_firing_range_list)

        condition = (jammer.jam_range > sam.firing_range >
                     sam.jammed_firing_range > fighter.firing_range)

        if not condition:
            raise Exception('Error in condition_l2_generator')

    def condition_m1(self, fighter, sam, target):
        sam_firing_range_list = np.arange(sam.firing_range_lower_bound,
                                          sam.max_firing_range + self.resolution, self.resolution)
        if sam_firing_range_list[-1] > sam.max_firing_range:
            sam_firing_range_list = sam_firing_range_list[:-1]

        c = np.random.rand()
        if c > 0.5:  # fighter.firing_range > sam.firing_range
            sam.firing_range = np.random.choice(sam_firing_range_list[:-1])
            fighter_firing_range_index = np.where(sam_firing_range_list > sam.firing_range)[0]
            fighter_firing_range_list = sam_firing_range_list[fighter_firing_range_index]
            fighter.firing_range = np.random.choice(fighter_firing_range_list)
            # print(f'{fighter.firing_range} > {sam.firing_range}')

            condition = (sam.max_firing_range >= sam.firing_range) and \
                        (fighter.max_firing_range >= fighter.firing_range) and \
                        (fighter.firing_range >= sam.firing_range + self.resolution)
        else:  # fighter.firing_range < sam.firing_range
            fighter_firing_range_list = np.arange(fighter.min_firing_range,
                                                  fighter.max_firing_range,
                                                  self.resolution)
            fighter.firing_range = np.random.choice(fighter_firing_range_list)
            sam_firing_range_list = np.arange(fighter.firing_range + self.resolution,
                                              sam.max_firing_range + self.resolution,
                                              self.resolution)
            sam.firing_range = np.random.choice(sam_firing_range_list)
            # print(f'{fighter.firing_range} < {sam.firing_range}')

            condition = (sam.max_firing_range >= sam.firing_range) and \
                        (fighter.max_firing_range >= fighter.firing_range) and \
                        (sam.firing_range > fighter.firing_range)

        if not condition:
            raise Exception('Error in condition mission m1!')

    def condition_f1_fighter(self, fighter, sam):
        if fighter.long_weapon > .5:
            fighter_firing_range_list = np.arange(sam.firing_range + self.resolution,
                                                  fighter.max_firing_range + self.resolution, self.resolution)
        else:
            fighter_firing_range_list = np.arange(fighter.min_firing_range,
                                                  sam.firing_range + self.resolution, self.resolution)

        fighter.firing_range = np.random.choice(fighter_firing_range_list)

        if fighter.long_weapon > .5:
            condition = (fighter.max_firing_range >= fighter.firing_range > sam.firing_range) and \
                        (sam.max_firing_range >= sam.firing_range)
        else:
            condition = (sam.max_firing_range >= sam.firing_range > fighter.firing_range) and \
                        (fighter.max_firing_range >= fighter.firing_range)

        return condition

    def condition_f1(self, fighter_1, fighter_2, fighter_3, sam):
        sam_firing_range_list = np.arange(sam.min_firing_range,
                                          sam.max_firing_range + self.resolution, self.resolution)
        sam_firing_range_list = sam_firing_range_list[1:-1]
        sam.firing_range = np.random.choice(sam_firing_range_list)

        long_weapon_fighter = np.random.choice(['fighter_1', 'fighter_2', 'fighter_3'])
        if long_weapon_fighter == 'fighter_1':
            fighter_1.long_weapon = 1
            fighter_2.long_weapon = 0
            fighter_3.long_weapon = 0
        elif long_weapon_fighter == 'fighter_2':
            fighter_1.long_weapon = 0
            fighter_2.long_weapon = 1
            fighter_3.long_weapon = 0
        elif long_weapon_fighter == 'fighter_3':
            fighter_1.long_weapon = 0
            fighter_2.long_weapon = 0
            fighter_3.long_weapon = 1

        condition_1 = self.condition_f1_fighter(self.fighter_1, sam)
        condition_2 = self.condition_f1_fighter(self.fighter_2, sam)
        condition_3 = self.condition_f1_fighter(self.fighter_3, sam)
        condition = condition_1 or condition_2 or condition_3

        if not condition:
            raise Exception('Error in condition mission f1!')

        # print(f'long_weapon_fighter={long_weapon_fighter}')
        # print(self.fighter_1.long_weapon, self.fighter_2.long_weapon, self.fighter_3.long_weapon)
        # print(self.fighter_1.firing_range, self.fighter_2.firing_range, self.fighter_3.firing_range)
        # print(f'sam.firing_range = {self.sam_1.firing_range}')

    def condition_d1(self, fighter_1, fighter_2, fighter_3, decoy, sam):
        sam_firing_range_list = np.arange(sam.min_firing_range,
                                          sam.max_firing_range + self.resolution, self.resolution)
        sam_firing_range_list = sam_firing_range_list[1:]
        sam.firing_range = np.random.choice(sam_firing_range_list)

        fighter_min_range = \
            np.max([sam.firing_range - sam.cooldown_max_count * self.resolution + self.resolution,
                    fighter_1.min_firing_range])

        fighter_firing_range_list = np.arange(fighter_min_range,
                                              sam.firing_range, self.resolution)
        fighter_1.firing_range = np.random.choice(fighter_firing_range_list)
        fighter_2.firing_range = np.random.choice(fighter_firing_range_list)
        fighter_3.firing_range = np.random.choice(fighter_firing_range_list)

    def reset_mission_condition(self):
        # Select the mission
        if self.mission_id == 'mission_1':
            # Select condition w1, w2, w3, l1, or l2
            selection_list = self.mission_conditions[0]
            p = self.mission_probability[0]
            cond_id = np.random.choice(selection_list, 1, p=p)[0]
            self.mission_condition = cond_id
            # print(f'\n--------- Mission condition: {cond_id}')

            func = 'self.condition_' + cond_id
            eval(func)(self.fighter_1, self.jammer_1, self.sam_1)

        elif self.mission_id == 'mission_2':
            # Select condition m1
            selection_list = self.mission_conditions[1]
            p = self.mission_probability[1]
            cond_id = np.random.choice(selection_list, 1, p=p)[0]
            self.mission_condition = cond_id
            # print(f'\n--------- Mission condition: {cond_id}')

            func = 'self.condition_' + cond_id
            eval(func)(self.fighter_1, self.sam_1, self.target_1)

        elif self.mission_id == 'mission_3':
            # Select condition f1
            selection_list = self.mission_conditions[2]
            p = self.mission_probability[2]
            cond_id = np.random.choice(selection_list, 1, p=p)[0]
            self.mission_condition = cond_id
            # print(f'\n--------- Mission condition: {cond_id}')

        elif self.mission_id == 'mission_4':
            # Select condition f1
            selection_list = self.mission_conditions[3]
            p = self.mission_probability[3]
            cond_id = np.random.choice(selection_list, 1, p=p)[0]
            self.mission_condition = cond_id
            # print(f'\n--------- Mission condition: {cond_id}')

            func = 'self.condition_' + cond_id
            eval(func)(self.fighter_1, self.fighter_2, self.fighter_3, self.decoy_1, self.sam_1)
            self.sam_1.firing_range_backup = self.sam_1.firing_range
            self.sam_1.jammed_firing_range_backup = self.sam_1.jammed_firing_range

        else:
            raise Exception('Mission condition is wrong !')

    def reset_mission_id(self):
        selection_list = self.mission_ids
        p = self.mission_ids_probability
        cond_id = np.random.choice(selection_list, 1, p=p)[0]
        self.mission_id = cond_id

        if cond_id == 'mission_1':
            self.blue_team = ['fighter_1', 'jammer_1']
            self.red_team = ['sam_1']

        elif cond_id == 'mission_2':
            self.blue_team = ['fighter_1']
            self.red_team = ['sam_1', 'target_1']

        elif cond_id == 'mission_3':
            self.blue_team = ['fighter_1', 'fighter_2', 'fighter_3']
            self.red_team = ['sam_1']

        elif cond_id == 'mission_4':
            self.blue_team = ['fighter_1', 'fighter_2', 'fighter_3', 'decoy_1']
            self.red_team = ['sam_1']

        else:
            raise Exception('mission_id error')

    def reset_blue_team(self):
        for id in self.blue_team:
            if id == 'fighter_1':
                self.set_initial_condition_fighter(self.fighter_1)
            elif id == 'fighter_2':
                self.set_initial_condition_fighter(self.fighter_2)
            elif id == 'fighter_3':
                self.set_initial_condition_fighter(self.fighter_3)
            elif id == 'jammer_1':
                self.set_initial_condition_jammer(self.jammer_1)
            elif id == 'decoy_1':
                self.set_initial_condition_decoy(self.decoy_1)
            else:
                raise Exception('Wrong blue team!')

    def set_initial_condition_fighter(self, fighter):
        r = 30 + 20 * np.random.rand()  # default
        # r = 50 + 20 * np.random.rand()

        c = np.random.rand()
        if c < .5:
            theta = - np.pi * np.random.rand()
            fighter.x = 50 + r * np.sin(theta)
            fighter.y = 50 + r * np.cos(theta)
            fighter.heading = - (-np.pi - theta)
        else:
            theta = np.pi * np.random.rand()
            fighter.x = 50 + r * np.sin(theta)
            fighter.y = 50 + r * np.cos(theta)
            fighter.heading = - (np.pi - theta)

        heading_error = 0 * np.pi / 180 * (np.random.rand() - .5) * 2  # default
        # heading_error = 180 * np.pi / 180 * (np.random.rand() - .5) * 2
        fighter.heading += heading_error
        fighter.heading_sin = np.sin(fighter.heading)
        fighter.heading_cos = np.cos(fighter.heading)

    def set_initial_condition_jammer(self, jammer):
        theta = np.pi * (np.random.rand() - 0.5) * 2
        r = 50 + 10 * (np.random.rand() - 0.5) * 2  # default
        # r = 40
        jammer.x = 50 + r * np.sin(theta)
        jammer.y = 50 + r * np.cos(theta)

        heading_error = 20 * np.pi / 180 * (np.random.rand() - 0.5) * 2  # default
        # heading_error = 90 * np.pi / 180
        if theta > 0:
            jammer.heading = -np.pi + theta + heading_error
        else:
            jammer.heading = np.pi + theta + heading_error

        jammer.heading_sin = np.sin(jammer.heading)
        jammer.heading_cos = np.cos(jammer.heading)

    def set_initial_condition_decoy(self, decoy):
        r = 30 + 20 * np.random.rand()  # default
        # r = 50 + 20 * np.random.rand()

        c = np.random.rand()
        if c < .5:
            theta = - np.pi * np.random.rand()
            decoy.x = 50 + r * np.sin(theta)
            decoy.y = 50 + r * np.cos(theta)
            decoy.heading = - (-np.pi - theta)
        else:
            theta = np.pi * np.random.rand()
            decoy.x = 50 + r * np.sin(theta)
            decoy.y = 50 + r * np.cos(theta)
            decoy.heading = - (np.pi - theta)

        heading_error = 0 * np.pi / 180 * (np.random.rand() - .5) * 2  # default
        # heading_error = 180 * np.pi / 180 * (np.random.rand() - .5) * 2
        decoy.heading += heading_error
        decoy.heading_sin = np.sin(decoy.heading)
        decoy.heading_cos = np.cos(decoy.heading)

    def reset_red_team(self):
        for id in self.red_team:
            if id == 'sam_1':
                self.set_initial_condition_sam(self.sam_1)
            elif id == 'target_1':
                self.set_initial_condition_target(self.target_1)
            else:
                raise Exception('Wrong red team!')

    def set_initial_condition_sam(self, sam):
        # r = 15 + 5 * np.random.rand()  # default
        r = 0

        c = np.random.rand()
        if c < .5:
            theta = - np.pi * np.random.rand()
            sam.x = 50 + r * np.sin(theta)
            sam.y = 50 + r * np.cos(theta)
            sam.heading = - (-np.pi - theta)
        else:
            theta = np.pi * np.random.rand()
            sam.x = 50 + r * np.sin(theta)
            sam.y = 50 + r * np.cos(theta)
            sam.heading = - (np.pi - theta)

        sam.heading_sin = np.sin(sam.heading)
        sam.heading_cos = np.cos(sam.heading)

    def set_initial_condition_target(self, target):
        target.x = 50
        target.y = 50

    def reset_render_blue_team(self):
        for id in self.blue_team:
            if id == 'fighter_1':
                self.reset_render_fighter(self.fighter_1)
            elif id == 'fighter_2':
                self.reset_render_fighter(self.fighter_2)
            elif id == 'fighter_3':
                self.reset_render_fighter(self.fighter_3)
            elif id == 'jammer_1':
                self.reset_render_jammer(self.jammer_1)
            elif id == 'decoy_1':
                self.reset_render_decoy(self.decoy_1)

    def reset_render_red_team(self):
        for id in self.red_team:
            if id == 'sam_1':
                self.reset_render_sam(self.sam_1)
            elif id == 'target_1':
                self.reset_render_target(self.target_1)

    def reset(self):
        self.steps = 0
        self.mission_id = None
        self.mission_condition = None

        # Reset fighter
        self.reset_fighter(self.fighter_1)
        self.reset_fighter(self.fighter_2)
        self.reset_fighter(self.fighter_3)

        # Reset jammer
        self.reset_jammer(self.jammer_1)

        # Reset decoy
        self.reset_decoy(self.decoy_1)

        # Reset sam
        self.reset_sam(self.sam_1)

        # Reset target
        self.reset_target(self.target_1)

        # Reset mission id
        self.reset_mission_id()

        # Reset mission condition
        self.reset_mission_condition()

        # Reset blue and red team
        self.reset_blue_team()
        self.reset_red_team()

        # Reset render blue and red team
        self.reset_render_blue_team()
        self.reset_render_red_team()

        observation = self.get_observation()
        return observation

    def check_jammer_on(self, jammer, sam):
        x = sam.x - jammer.x
        y = sam.y - jammer.y
        r = (x * x + y * y) ** .5
        if jammer.jam_range >= r:
            jammer.on = 1
            sam.jammed = 1
        else:
            jammer.on = 0
            sam.jammed = 0

    def check_sam_cooldown_start(self, decoy, sam):
        r = ((sam.x - decoy.x) ** 2 + (sam.y - decoy.y) ** 2) ** .5
        if r <= sam.firing_range:
            sam.cooldown_on = 1
            # decoy.speed = 1

    def sam_cooldown(self, sam):
        if sam.cooldown_counter <= sam.cooldown_max_count:
            sam.firing_range = 0
            sam.jammed_firing_range = 0
            sam.cooldown_counter += 1
        else:
            sam.cooldown_on = 0
            sam.firing_range = sam.firing_range_backup
            sam.jammed_firing_range = sam.jammed_firing_range_backup
            sam.cooldown_counter = 0

        # pygame.time.wait(100)

    def check_fighter_win(self, fighter, entity):
        x = entity.x - fighter.x
        y = entity.y - fighter.y
        r = (x * x + y * y) ** .5
        if fighter.firing_range >= r:
            entity.alive = 0

    def check_sam_win_fighter(self, fighter, sam):
        x = fighter.x - sam.x
        y = fighter.y - sam.y
        r = (x * x + y * y) ** .5
        if (r < sam.firing_range) and (sam.jammed < 0.5):
            fighter.alive = 0

    def check_sam_win_jammer(self, jammer, sam):
        x = jammer.x - sam.x
        y = jammer.y - sam.y
        r = (x * x + y * y) ** .5
        if (r < sam.firing_range) and (jammer.on < 0.5):
            jammer.alive = 0

    def check_sam_win_decoy(self, decoy, sam):
        if sam.cooldown_on > .5:
            decoy.alive = 0

        x = decoy.x - sam.x
        y = decoy.y - sam.y
        r = (x * x + y * y) ** .5
        if (r < sam.firing_range) and (sam.jammed < 0.5):
            decoy.alive = 0

    def check_jammed_sam_win_fighter(self, fighter, sam):
        x = fighter.x - sam.x
        y = fighter.y - sam.y
        r = (x * x + y * y) ** .5

        if (r < sam.jammed_firing_range) and (sam.jammed > 0.5):
            fighter.alive = 0

    def check_jammed_sam_win_jammer(self, jammer, sam):
        x = jammer.x - sam.x
        y = jammer.y - sam.y
        r = (x * x + y * y) ** .5

        if (r < sam.jammed_firing_range) and (sam.jammed > 0.5):
            jammer.alive = 0

    def check_jammed_sam_win_decoy(self, decoy, sam):
        x = decoy.x - sam.x
        y = decoy.y - sam.y
        r = (x * x + y * y) ** .5

        if (r < sam.jammed_firing_range) and (sam.jammed > 0.5):
            decoy.alive = 0

    def transient(self, entity):
        entity.heading += entity.action
        entity.heading_sin = np.sin(entity.heading)
        entity.heading_cos = np.cos(entity.heading)

        entity.x += entity.speed * entity.heading_sin * self.dt
        entity.y += entity.speed * entity.heading_cos * self.dt

    def step(self, actions):
        n = 0  # ローカル・ステップ
        done = False
        info = {}

        ''' 前のタイムステップの状態更新 '''
        if self.mission_id == 'mission_1':
            self.fighter_1.previous_x = self.fighter_1.x
            self.fighter_1.previous_y = self.fighter_1.y
            self.jammer_1.previous_x = self.jammer_1.x
            self.jammer_1.previous_y = self.jammer_1.y
        elif self.mission_id == 'mission_2':
            self.fighter_1.previous_x = self.fighter_1.x
            self.fighter_1.previous_y = self.fighter_1.y
            self.sam_1.previous_alive = self.sam_1.alive
        elif self.mission_id == 'mission_3':
            self.fighter_1.previous_x = self.fighter_1.x
            self.fighter_1.previous_y = self.fighter_1.y
            self.fighter_2.previous_x = self.fighter_2.x
            self.fighter_2.previous_y = self.fighter_2.y
            self.fighter_3.previous_x = self.fighter_3.x
            self.fighter_3.previous_y = self.fighter_3.y
        elif self.mission_id == 'mission_4':
            self.fighter_1.previous_x = self.fighter_1.x
            self.fighter_1.previous_y = self.fighter_1.y
            self.fighter_2.previous_x = self.fighter_2.x
            self.fighter_2.previous_y = self.fighter_2.y
            self.fighter_3.previous_x = self.fighter_3.x
            self.fighter_3.previous_y = self.fighter_3.y
            self.decoy_1.previous_x = self.decoy_1.x
            self.decoy_1.previous_y = self.decoy_1.y
        else:
            raise Exception('Not yet implemented')

        ''' アクション取得 '''
        if self.mission_id == 'mission_1':
            self.fighter_1.action = actions[0] * self.fighter_1.max_heading_change_step
            self.jammer_1.action = actions[1] * self.jammer_1.max_heading_change_step
        elif self.mission_id == 'mission_2':
            self.fighter_1.action = actions[0] * self.fighter_1.max_heading_change_step
        elif self.mission_id == 'mission_3':
            self.fighter_1.action = actions[0] * self.fighter_1.max_heading_change_step
            self.fighter_2.action = actions[1] * self.fighter_2.max_heading_change_step
            self.fighter_3.action = actions[2] * self.fighter_3.max_heading_change_step
        elif self.mission_id == 'mission_4':
            self.fighter_1.action = actions[0] * self.fighter_1.max_heading_change_step
            self.fighter_2.action = actions[1] * self.fighter_2.max_heading_change_step
            self.fighter_3.action = actions[2] * self.fighter_3.max_heading_change_step
            self.decoy_1.action = actions[3] * self.decoy_1.max_heading_change_step

        while (done != True) and (n < self.action_interval):

            ''' 状態遷移(state transition)計算 '''
            for id in self.blue_team:
                if id == 'fighter_1':
                    self.transient(self.fighter_1)
                if id == 'fighter_2':
                    self.transient(self.fighter_2)
                if id == 'fighter_3':
                    self.transient(self.fighter_3)
                if id == 'jammer_1':
                    self.transient(self.jammer_1)
                if id == 'decoy_1':
                    self.transient(self.decoy_1)

            # For the future application
            # rgb_shot = self.get_snapshot()

            # Jammerのjam_range内にSAMがいれば、Jammerが有効
            if ('jammer_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_jammer_on(self.jammer_1, self.sam_1)

            # SAMのcooldown time内であれば、SAMの射程＝０
            if ('decoy_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_cooldown_start(self.decoy_1, self.sam_1)

            if self.sam_1.cooldown_on > .5:
                self.sam_cooldown(self.sam_1)

            # fighter win
            if ('fighter_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_fighter_win(self.fighter_1, self.sam_1)

            if ('fighter_2' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_fighter_win(self.fighter_2, self.sam_1)

            if ('fighter_3' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_fighter_win(self.fighter_3, self.sam_1)

            if ('fighter_1' in self.blue_team) and ('target_1' in self.red_team):
                self.check_fighter_win(self.fighter_1, self.target_1)

            # clean sam win to fighter
            if ('fighter_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_fighter(self.fighter_1, self.sam_1)

            if ('fighter_2' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_fighter(self.fighter_2, self.sam_1)

            if ('fighter_3' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_fighter(self.fighter_3, self.sam_1)

            # clean sam win to jammer & decoy
            if ('jammer_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_jammer(self.jammer_1, self.sam_1)

            if ('decoy_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_decoy(self.decoy_1, self.sam_1)

            # jammed sam win to fighter
            if ('fighter_1' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_jammed_sam_win_fighter(self.fighter_1, self.sam_1)
            if ('fighter_2' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_jammed_sam_win_fighter(self.fighter_2, self.sam_1)
            if ('fighter_3' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_jammed_sam_win_fighter(self.fighter_3, self.sam_1)

            # jammed sam win to jammer & decoy
            if ('jammer_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_jammed_sam_win_jammer(self.jammer_1, self.sam_1)

            if ('decoy_1' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_jammed_sam_win_decoy(self.decoy_1, self.sam_1)

            # Update blue_team
            if (self.fighter_1.alive < 0.5) and ('fighter_1' in self.blue_team):
                self.blue_team.remove('fighter_1')

            if (self.fighter_2.alive < 0.5) and ('fighter_2' in self.blue_team):
                self.blue_team.remove('fighter_2')

            if (self.fighter_3.alive < 0.5) and ('fighter_3' in self.blue_team):
                self.blue_team.remove('fighter_3')

            if (self.jammer_1.alive < 0.5) and ('jammer_1' in self.blue_team):
                self.blue_team.remove('jammer_1')

            if (self.decoy_1.alive < 0.5) and ('decoy_1' in self.blue_team):
                self.blue_team.remove('decoy_1')

            # Update red_team
            if (self.sam_1.alive < 0.5) and ('sam_1' in self.red_team):
                self.red_team.remove('sam_1')

            if (self.target_1.alive < 0.5) and ('target_1' in self.red_team):
                self.red_team.remove('target_1')

            ''' 終了(done)判定 '''
            done = self.is_done()

            ''' 報酬(reward)計算 '''
            reward = self.get_reward(done)

            ''' ローカル・ステップのカウント・アップ '''
            n += 1

        ''' 観測(observation)の計算 '''
        observation = self.get_observation()

        ''' Set information, if any '''
        if done:
            if self.mission_id == 'mission_2':
                if self.fighter_1.firing_range > self.sam_1.firing_range:
                    if self.target_1.alive > .5:
                        info['result'] = 'F1'
                    else:
                        if self.sam_1.alive < .5:
                            info['result'] = 'S1'
                        else:
                            info['result'] = 'S2'
                else:
                    if self.target_1.alive > .5:
                        info['result'] = 'F2'
                    else:
                        info['result'] = 'S3'
            elif (self.mission_id == 'mission_3') or (self.mission_id == 'mission_4'):
                if (self.fighter_1.alive > .5) and (self.fighter_2.alive > .5) and \
                        (self.fighter_3.alive > .5) and (self.sam_1.alive < .5):
                    info['result'] = 'success'
                else:
                    info['result'] = 'fail'

        ''' for debug '''
        # pygame.time.wait(100)

        self.steps += 1

        return observation, reward, done, info

    def render_fighter(self, fighter):
        # transform coordinate from battle space to screen
        fighter.screen_x = fighter.x * self.to_screen_x + self.to_screen_offset
        fighter.screen_y = fighter.y * self.to_screen_y + self.to_screen_offset

        # draw fighter
        screen_x = fighter.screen_x - fighter.radius
        screen_y = fighter.screen_y - fighter.radius
        self.screen.blit(fighter.surface, (screen_x, screen_y))

        # draw fighter's firing range
        range_screen_x = fighter.screen_x - fighter.screen_firing_range
        range_screen_y = fighter.screen_y - fighter.screen_firing_range
        self.screen.blit(fighter.surface_range, (range_screen_x, range_screen_y))

    def render_jammer(self, jammer):
        # transform coordinate from battle space to screen
        jammer.screen_x = jammer.x * self.to_screen_x + self.to_screen_offset
        jammer.screen_y = jammer.y * self.to_screen_y + self.to_screen_offset

        # draw jammer
        screen_x = jammer.screen_x - jammer.radius
        screen_y = jammer.screen_y - jammer.radius
        self.screen.blit(jammer.surface, (screen_x, screen_y))

        # draw jammer's firing range
        range_screen_x = jammer.screen_x - jammer.screen_jam_range
        range_screen_y = jammer.screen_y - jammer.screen_jam_range
        self.screen.blit(jammer.surface_range, (range_screen_x, range_screen_y))

    def render_decoy(self, decoy):
        # transform coordinate from battle space to screen
        decoy.screen_x = decoy.x * self.to_screen_x + self.to_screen_offset
        decoy.screen_y = decoy.y * self.to_screen_y + self.to_screen_offset

        # draw decoy
        screen_x = decoy.screen_x - decoy.radius
        screen_y = decoy.screen_y - decoy.radius
        self.screen.blit(decoy.surface, (screen_x, screen_y))

    def render_sam(self, sam):
        # transform coordinate from battle space to screen
        sam.screen_x = sam.x * self.to_screen_x + self.to_screen_offset
        sam.screen_y = sam.y * self.to_screen_y + self.to_screen_offset

        # draw sam
        screen_x = sam.screen_x - sam.radius
        screen_y = sam.screen_y - sam.radius
        self.screen.blit(sam.surface, (screen_x, screen_y))

        if sam.cooldown_on < .5:
            if sam.jammed < 0.5:
                # draw sam's firing range
                range_screen_x = sam.screen_x - sam.screen_firing_range
                range_screen_y = sam.screen_y - sam.screen_firing_range
                self.screen.blit(sam.surface_range, (range_screen_x, range_screen_y))
            else:
                # draw sam's jammed firing range
                range_screen_x = sam.screen_x - sam.screen_jammed_firing_range
                range_screen_y = sam.screen_y - sam.screen_jammed_firing_range
                self.screen.blit(sam.surface_jammed_range, (range_screen_x, range_screen_y))

    def render_target(self, target):
        # transform coordinate from battle space to screen
        target.screen_x = target.x * self.to_screen_x + self.to_screen_offset
        target.screen_y = target.y * self.to_screen_y + self.to_screen_offset

        # draw target
        screen_x = target.screen_x - target.radius
        screen_y = target.screen_y - target.radius
        self.screen.blit(target.surface, (screen_x, screen_y))

    def render(self, mode='human'):
        if mode == 'human':
            self.screen.fill((0, 0, 0))

            for id in self.red_team:
                if (id == 'sam_1') and (self.sam_1.alive > .5):
                    self.render_sam(self.sam_1)
                if (id == 'target_1') and (self.target_1.alive > .5):
                    self.render_target(self.target_1)

            for id in self.blue_team:
                if (id == 'fighter_1') and (self.fighter_1.alive > .5):
                    self.render_fighter(self.fighter_1)
                if (id == 'fighter_2') and (self.fighter_2.alive > .5):
                    self.render_fighter(self.fighter_2)
                if (id == 'fighter_3') and (self.fighter_3.alive > .5):
                    self.render_fighter(self.fighter_3)
                if (id == 'jammer_1') and (self.jammer_1.alive > .5):
                    self.render_jammer(self.jammer_1)
                if (id == 'decoy_1') and (self.decoy_1.alive > .5) and (self.sam_1.cooldown_on < .5):
                    self.render_decoy(self.decoy_1)

            pygame.display.update()

            shot = pygame.surfarray.array3d(self.screen)
            shot = np.array(shot, dtype=np.uint8)
        else:
            shot = None
        return shot

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_observation(self):
        if self.mission_id == 'mission_1':
            obs = self.get_relative_observation_mission_1()
        elif self.mission_id == 'mission_2':
            obs = self.get_relative_observation_mission_2()
        elif self.mission_id == 'mission_3':
            obs = self.get_relative_observation_mission_3()
        elif self.mission_id == 'mission_4':
            obs = self.get_relative_observation_mission_4()
        else:
            raise Exception('Not yet implemented')

        observation = np.array(obs, dtype=np.float32)  # (38,)
        return observation

    def make_v(self, entity):
        v = [entity.x - entity.previous_x, entity.y - entity.previous_y]
        return v

    def make_r(self, entity_1, entity_2):
        r = [entity_1.x - entity_2.x, entity_1.y - entity_2.y]
        return r

    def compute_heading_error(self, v, r):
        epsiron = 1e-5
        e = v / (np.linalg.norm(v, ord=2) + epsiron)
        u = r / (np.linalg.norm(r, ord=2) + epsiron)

        cos_heading = e[0] * u[0] + e[1] * u[1]
        sin_heading = e[0] * u[1] - e[1] * u[0]

        epsilon = 1e-3
        if (np.abs(cos_heading) > 1 + epsilon) or (np.abs(sin_heading) > 1 + epsilon):
            print(sin_heading, cos_heading)
            raise Exception('Wrong heading error')

        return sin_heading, cos_heading

    def get_relative_observation_mission_1(self):
        obs = []
        r = (self.space_x ** 2 + self.space_y ** 2) ** 0.5

        v_f = self.make_v(self.fighter_1)
        r_fs = self.make_r(self.sam_1, self.fighter_1)
        sin_f, cos_f = self.compute_heading_error(v_f, r_fs)
        obs.append(sin_f)
        obs.append(cos_f)
        obs.append(np.linalg.norm(r_fs, ord=2) / r)

        v_j = self.make_v(self.jammer_1)
        r_js = self.make_r(self.sam_1, self.jammer_1)
        sin_j, cos_j = self.compute_heading_error(v_j, r_js)
        obs.append(sin_j)
        obs.append(cos_j)
        obs.append(np.linalg.norm(r_js, ord=2) / r)

        r_fj = self.make_r(self.jammer_1, self.fighter_1)
        sin_phai_f, cos_phai_f = self.compute_heading_error(v_f, r_fj)
        sin_phai_j, cos_phai_j = self.compute_heading_error(v_j, r_fj)
        obs.append(sin_phai_f)
        obs.append(cos_phai_f)
        obs.append(sin_phai_j)
        obs.append(cos_phai_j)
        obs.append(np.linalg.norm(r_fj, ord=2) / r)

        obs.append(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.weapon_count)
        obs.append(self.fighter_1.alive)

        obs.append(self.jammer_1.alive)
        obs.append(self.jammer_1.on)

        obs.append(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(
            self.sam_1.jammed_firing_range / (self.sam_1.max_firing_range * self.jammer_1.jam_effectiveness))
        obs.append(self.sam_1.weapon_count)
        obs.append(self.sam_1.alive)

        return obs

    def get_relative_observation_mission_2(self):
        obs = []
        r = (self.space_x ** 2 + self.space_y ** 2) ** 0.5

        v_f = self.make_v(self.fighter_1)
        r_fs = self.make_r(self.sam_1, self.fighter_1)
        sin_fs, cos_fs = self.compute_heading_error(v_f, r_fs)

        r_ft = self.make_r(self.target_1, self.fighter_1)
        sin_ft, cos_ft = self.compute_heading_error(v_f, r_ft)

        r_st = self.make_r(self.target_1, self.sam_1)
        sin_s, cos_s = self.compute_heading_error(r_fs, r_st)
        sin_t, cos_t = self.compute_heading_error(r_ft, r_st)

        obs.append(sin_fs)
        obs.append(cos_fs)
        obs.append(np.linalg.norm(r_fs, ord=2) / r)
        obs.append(sin_ft)
        obs.append(cos_ft)
        obs.append(np.linalg.norm(r_ft, ord=2) / r)
        obs.append(sin_s)
        obs.append(cos_s)
        obs.append(sin_t)
        obs.append(cos_t)
        obs.append(np.linalg.norm(r_st, ord=2) / r)

        # print(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.weapon_count)
        obs.append(self.fighter_1.alive)

        # print(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(
            self.sam_1.jammed_firing_range / (self.sam_1.max_firing_range * self.jammer_1.jam_effectiveness))
        obs.append(self.sam_1.weapon_count)
        obs.append(self.sam_1.alive)

        obs.append(self.target_1.alive)

        return obs

    def get_relative_observation_mission_3(self):
        obs = []
        r = (self.space_x ** 2 + self.space_y ** 2) ** 0.5

        v_1 = self.make_v(self.fighter_1)
        r_1s = self.make_r(self.sam_1, self.fighter_1)
        sin_1, cos_1 = self.compute_heading_error(v_1, r_1s)
        obs.append(sin_1)
        obs.append(cos_1)
        obs.append(np.linalg.norm(r_1s, ord=2) / r)

        v_2 = self.make_v(self.fighter_2)
        r_2s = self.make_r(self.sam_1, self.fighter_2)
        sin_2, cos_2 = self.compute_heading_error(v_2, r_2s)
        obs.append(sin_2)
        obs.append(cos_2)
        obs.append(np.linalg.norm(r_2s, ord=2) / r)

        v_3 = self.make_v(self.fighter_3)
        r_3s = self.make_r(self.sam_1, self.fighter_3)
        sin_3, cos_3 = self.compute_heading_error(v_3, r_3s)
        obs.append(sin_3)
        obs.append(cos_3)
        obs.append(np.linalg.norm(r_3s, ord=2) / r)

        r_12 = self.make_r(self.fighter_2, self.fighter_1)
        r_13 = self.make_r(self.fighter_3, self.fighter_1)
        r_23 = self.make_r(self.fighter_3, self.fighter_2)

        sin_phai_12, cos_phai_12 = self.compute_heading_error(v_1, r_12)
        sin_phai_13, cos_phai_13 = self.compute_heading_error(v_1, r_13)
        sin_phai_21, cos_phai_21 = self.compute_heading_error(v_2, r_12)
        sin_phai_23, cos_phai_23 = self.compute_heading_error(v_2, r_23)
        sin_phai_31, cos_phai_31 = self.compute_heading_error(v_3, r_13)
        sin_phai_32, cos_phai_32 = self.compute_heading_error(v_3, r_23)
        obs.append(sin_phai_12)
        obs.append(cos_phai_12)
        obs.append(sin_phai_13)
        obs.append(cos_phai_13)
        obs.append(sin_phai_21)
        obs.append(cos_phai_21)
        obs.append(sin_phai_23)
        obs.append(cos_phai_23)
        obs.append(sin_phai_31)
        obs.append(cos_phai_31)
        obs.append(sin_phai_32)
        obs.append(cos_phai_32)
        obs.append(np.linalg.norm(r_12, ord=2) / r)
        obs.append(np.linalg.norm(r_13, ord=2) / r)
        obs.append(np.linalg.norm(r_23, ord=2) / r)

        obs.append(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.weapon_count)
        obs.append(self.fighter_1.alive)

        obs.append(self.fighter_2.firing_range / self.fighter_2.max_firing_range)
        obs.append(self.fighter_2.weapon_count)
        obs.append(self.fighter_2.alive)

        obs.append(self.fighter_3.firing_range / self.fighter_3.max_firing_range)
        obs.append(self.fighter_3.weapon_count)
        obs.append(self.fighter_3.alive)

        obs.append(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(self.sam_1.jammed_firing_range /
                   (self.sam_1.max_firing_range * self.jammer_1.jam_effectiveness))
        obs.append(self.sam_1.weapon_count)
        obs.append(self.sam_1.alive)

        return obs

    def get_relative_observation_mission_4(self):
        obs = []
        r = (self.space_x ** 2 + self.space_y ** 2) ** 0.5

        v_1 = self.make_v(self.fighter_1)
        r_1s = self.make_r(self.sam_1, self.fighter_1)
        sin_1, cos_1 = self.compute_heading_error(v_1, r_1s)
        obs.append(sin_1)
        obs.append(cos_1)
        obs.append(np.linalg.norm(r_1s, ord=2) / r)

        v_2 = self.make_v(self.fighter_2)
        r_2s = self.make_r(self.sam_1, self.fighter_2)
        sin_2, cos_2 = self.compute_heading_error(v_2, r_2s)
        obs.append(sin_2)
        obs.append(cos_2)
        obs.append(np.linalg.norm(r_2s, ord=2) / r)

        v_3 = self.make_v(self.fighter_3)
        r_3s = self.make_r(self.sam_1, self.fighter_3)
        sin_3, cos_3 = self.compute_heading_error(v_3, r_3s)
        obs.append(sin_3)
        obs.append(cos_3)
        obs.append(np.linalg.norm(r_3s, ord=2) / r)

        r_12 = self.make_r(self.fighter_2, self.fighter_1)
        r_13 = self.make_r(self.fighter_3, self.fighter_1)
        r_23 = self.make_r(self.fighter_3, self.fighter_2)

        sin_phai_12, cos_phai_12 = self.compute_heading_error(v_1, r_12)
        sin_phai_13, cos_phai_13 = self.compute_heading_error(v_1, r_13)
        sin_phai_21, cos_phai_21 = self.compute_heading_error(v_2, r_12)
        sin_phai_23, cos_phai_23 = self.compute_heading_error(v_2, r_23)
        sin_phai_31, cos_phai_31 = self.compute_heading_error(v_3, r_13)
        sin_phai_32, cos_phai_32 = self.compute_heading_error(v_3, r_23)
        obs.append(sin_phai_12)
        obs.append(cos_phai_12)
        obs.append(sin_phai_13)
        obs.append(cos_phai_13)
        obs.append(sin_phai_21)
        obs.append(cos_phai_21)
        obs.append(sin_phai_23)
        obs.append(cos_phai_23)
        obs.append(sin_phai_31)
        obs.append(cos_phai_31)
        obs.append(sin_phai_32)
        obs.append(cos_phai_32)
        obs.append(np.linalg.norm(r_12, ord=2) / r)
        obs.append(np.linalg.norm(r_13, ord=2) / r)
        obs.append(np.linalg.norm(r_23, ord=2) / r)

        obs.append(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.weapon_count)
        obs.append(self.fighter_1.alive)

        obs.append(self.fighter_2.firing_range / self.fighter_2.max_firing_range)
        obs.append(self.fighter_2.weapon_count)
        obs.append(self.fighter_2.alive)

        obs.append(self.fighter_3.firing_range / self.fighter_3.max_firing_range)
        obs.append(self.fighter_3.weapon_count)
        obs.append(self.fighter_3.alive)

        obs.append(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(self.sam_1.jammed_firing_range /
                   (self.sam_1.max_firing_range * self.jammer_1.jam_effectiveness))
        obs.append(self.sam_1.weapon_count)
        obs.append(self.sam_1.alive)

        if self.decoy_1.alive:
            v_d = self.make_v(self.decoy_1)
            r_ds = self.make_r(self.sam_1, self.decoy_1)
            sin_d1, cos_d1 = self.compute_heading_error(v_d, r_ds)
            obs.append(sin_d1)
            obs.append(cos_d1)
            obs.append(np.linalg.norm(r_ds, ord=2) / r)

            r_d1 = self.make_r(self.fighter_1, self.decoy_1)
            r_d2 = self.make_r(self.fighter_2, self.decoy_1)
            r_d3 = self.make_r(self.fighter_3, self.decoy_1)

            sin_phai_d1, cos_phai_d1 = self.compute_heading_error(v_d, r_d1)
            sin_phai_d2, cos_phai_d2 = self.compute_heading_error(v_d, r_d2)
            sin_phai_d3, cos_phai_d3 = self.compute_heading_error(v_d, r_d3)
            sin_phai_1d, cos_phai_1d = self.compute_heading_error(v_1, r_d1)
            sin_phai_2d, cos_phai_2d = self.compute_heading_error(v_2, r_d2)
            sin_phai_3d, cos_phai_3d = self.compute_heading_error(v_3, r_d3)
            obs.append(sin_phai_d1)
            obs.append(cos_phai_d1)
            obs.append(sin_phai_d2)
            obs.append(cos_phai_d2)
            obs.append(sin_phai_d3)
            obs.append(cos_phai_d3)
            obs.append(sin_phai_1d)
            obs.append(cos_phai_1d)
            obs.append(sin_phai_2d)
            obs.append(cos_phai_2d)
            obs.append(sin_phai_3d)
            obs.append(cos_phai_3d)
            obs.append(np.linalg.norm(r_d1, ord=2) / r)
            obs.append(np.linalg.norm(r_d2, ord=2) / r)
            obs.append(np.linalg.norm(r_d3, ord=2) / r)
        else:
            for _ in range(18):
                obs.append(0)

        obs.append(self.decoy_1.alive)
        obs.append(self.sam_1.cooldown_counter / (self.sam_1.cooldown_max_count + 1))
        obs.append(self.sam_1.cooldown_on)

        return obs

    def get_reward_mission_1(self, done, fighter, jammer, sam):
        reward = 0

        # For done
        if done:
            if (self.mission_condition == 'w1') and (fighter.alive > .5) and \
                    (jammer.alive > .5) and (jammer.on < .5) and (sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'w2') and (fighter.alive > .5) and \
                    (jammer.alive > .5) and (jammer.on > .5) and (sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'w3') and (fighter.alive > .5) and \
                    (jammer.alive > .5) and (jammer.on < .5) and (sam.alive < .5):
                reward = 1

            elif (self.mission_condition == 'l1') and (fighter.alive > .5) and (jammer.alive > .5):
                reward = 1

            elif (self.mission_condition == 'l2') and (fighter.alive > .5) and (jammer.alive > .5):
                reward = 1

            else:
                reward = -1

        return reward

    def get_reward_mission_2(self, done, fighter, sam, target):
        reward = 0

        # For destroy sam
        if (sam.alive < .5) and (sam.previous_alive > .5):
            reward += 1

        # For done
        if done:
            """
            if (self.fighter_1.alive > .5) and (self.target_1.alive < .5):
                if (self.fighter_1.firing_range > self.sam_1.firing_range) and (self.sam_1.alive < .5):
                    reward = 1
                elif self.fighter_1.firing_range < self.sam_1.firing_range:
                    reward = 1
            """
            if self.target_1.alive < .5:
                reward += 1
            else:
                reward = -1

        return reward

    def get_reward_mission_3(self, done, fighter_1, fighter_2, fighter_3, sam):
        reward = 0

        if done:
            if (fighter_1.alive > .5) and (fighter_2.alive > .5) and (fighter_3.alive > .5) and \
                    (sam.alive < .5):
                reward = 1
            else:
                reward = -1

        return reward

    def get_reward_mission_4(self, done, fighter_1, fighter_2, fighter_3, decoy, sam):
        reward = -0.1

        # cond_1
        cond_1 = (decoy.alive > .5)

        # cond_2
        v_d = self.make_v(decoy)
        r_ds = self.make_r(sam, decoy)
        _, cos_d = self.compute_heading_error(v_d, r_ds)
        cond_2 = (cos_d > np.cos(10 * np.pi / 180))

        # cond_3
        fighters = ['fighter_1', 'fighter_2', 'fighter_3']

        cos_f_list = []
        r_f_list = []
        for f in fighters:
            fighter = eval(f)
            v_f = self.make_v(fighter)
            r_fs = self.make_r(sam, fighter)
            _, cos_f = self.compute_heading_error(v_f, r_fs)
            cos_f_list.append(cos_f)
            r_f_list.append(np.linalg.norm(r_fs, ord=2) - np.linalg.norm(r_ds, ord=2))

        cond_3_list = []
        for i in range(len(fighters)):
            con1 = (0 < r_f_list[i] <= sam.cooldown_max_count * self.resolution)
            con2 = (cos_f_list[i] > np.cos(10 * np.pi / 180))
            cond_3_list.append(con1 and con2)

        cond_3 = (True in cond_3_list)

        cond_4_list = []
        for i in range(len(fighters)):
            con1 = (0 < r_f_list[i])
            cond_4_list.append(con1)

        cond_4 = not (False in cond_4_list)

        cond = (cond_1 and cond_2 and cond_3 and cond_4)
        if cond:
            alpha = 0.1
            reward += alpha * cos_d

        if done:
            if (fighter_1.alive > .5) and (fighter_2.alive > .5) and (fighter_3.alive > .5) and \
                    (sam.alive < .5):
                reward = 1
            else:
                reward = -1

        return reward

    def get_reward(self, done):
        if self.mission_id == 'mission_1':
            reward = self.get_reward_mission_1(done, self.fighter_1, self.jammer_1, self.sam_1)
        elif self.mission_id == 'mission_2':
            reward = self.get_reward_mission_2(done, self.fighter_1, self.sam_1, self.target_1)
        elif self.mission_id == 'mission_3':
            reward = self.get_reward_mission_3(done,
                                               self.fighter_1, self.fighter_2, self.fighter_3, self.sam_1)
        elif self.mission_id == 'mission_4':
            reward = self.get_reward_mission_4(
                done, self.fighter_1, self.fighter_2, self.fighter_3, self.decoy_1, self.sam_1)
        else:
            raise Exception('Wrong reward!')

        return reward

    def get_snapshot(self):
        shot = pygame.surfarray.array3d(pygame.transform.scale(self.screen, self.shot_shape))
        return np.array(shot, dtype=np.uint8)

    def is_done_mission_1(self, fighter, jammer, sam):
        done = False

        if (sam.alive < .5) or (jammer.alive < .5) or (fighter.alive < .5):
            done = True

        if (fighter.x < - self.space_offset) or (jammer.x < - self.space_offset):
            done = True
        if (fighter.x > self.space_x + self.space_offset) or (jammer.x > self.space_x + self.space_offset):
            done = True
        if (fighter.y < - self.space_offset) or (jammer.y < - self.space_offset):
            done = True
        if (fighter.y > self.space_y + self.space_offset) or (jammer.y > self.space_y + self.space_offset):
            done = True

        if self.steps > self.max_steps:
            done = True

        return done

    def is_done_mission_2(self, fighter, sam, target):
        done = False
        if (fighter.alive < .5) or (target.alive < .5):
            done = True

        if fighter.x < - self.space_offset:
            done = True
        if fighter.x > self.space_x + self.space_offset:
            done = True
        if fighter.y < - self.space_offset:
            done = True
        if fighter.y > self.space_y + self.space_offset:
            done = True

        if self.steps > self.max_steps:
            done = True

        return done

    def check_boundary(self, fighter):
        local_done = False

        if fighter.x < - self.space_offset:
            local_done = True
        if fighter.x > self.space_x + self.space_offset:
            local_done = True
        if fighter.y < - self.space_offset:
            local_done = True
        if fighter.y > self.space_y + self.space_offset:
            local_done = True

        return local_done

    def is_done_mission_3(self, fighter_1, fighter_2, fighter_3, sam):
        done = False

        if (fighter_1.alive < .5) or (fighter_2.alive < .5) or (fighter_3.alive < .5) or (sam.alive < .5):
            done = True

        done_1 = self.check_boundary(fighter_1)
        done_2 = self.check_boundary(fighter_2)
        done_3 = self.check_boundary(fighter_3)
        if done_1 or done_2 or done_3:
            done = True

        if self.steps > self.max_steps:
            done = True

        return done

    def is_done_mission_4(self, fighter_1, fighter_2, fighter_3, decoy, sam):
        done = False

        if (fighter_1.alive < .5) or (fighter_2.alive < .5) or (fighter_3.alive < .5) or (sam.alive < .5):
            done = True

        if (decoy.alive < .5) and (sam.cooldown_on < 0.5):
            done = True

        done_1 = self.check_boundary(fighter_1)
        done_2 = self.check_boundary(fighter_2)
        done_3 = self.check_boundary(fighter_3)
        done_4 = self.check_boundary(decoy)
        if done_1 or done_2 or done_3 or done_4:
            done = True

        if self.steps > self.max_steps:
            done = True

        return done

    def is_done(self):
        done = False

        if self.mission_id == 'mission_1':
            done = self.is_done_mission_1(self.fighter_1, self.jammer_1, self.sam_1)

        if self.mission_id == 'mission_2':
            done = self.is_done_mission_2(self.fighter_1, self.sam_1, self.target_1)

        if self.mission_id == 'mission_3':
            done = self.is_done_mission_3(self.fighter_1, self.fighter_2, self.fighter_3, self.sam_1)

        if self.mission_id == 'mission_4':
            done = self.is_done_mission_4(self.fighter_1, self.fighter_2, self.fighter_3,
                                          self.decoy_1, self.sam_1)

        if (not self.blue_team) or (not self.red_team):
            done = True

        return done
