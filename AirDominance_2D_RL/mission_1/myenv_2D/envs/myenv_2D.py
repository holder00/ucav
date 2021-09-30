"""
This code may be called from test_env.py
Gym environment for 2D reinforcement learning problem inspored by RAND corporations report:
    "AirDominance Through Machine Learning, A Preliminary Exploration of
    Artificial Intelligence-Assisted Mission Planning", RAND Corporation, 2020
        https://www.rand.org/pubs/research_reports/RR4311.html
"""
import gym
import numpy as np
import pygame
import math
import matplotlib.pyplot as plt

"""
Define breakdown of mission conditions
"""
MISSION_IDS = ['mission_1', 'mission_2', 'mission_3']
MISSION_IDS_RATIO = [1, 0, 0]

MISSION_CONDITIONS_1 = ['w1', 'w2', 'w3', 'l1', 'l2']
MISSION_RATIO_1 = [0, 1, 0, 0, 0]

MISSION_CONDITIONS_2 = []
MISSION_RATIO_2 = []

MISSION_CONDITIONS_3 = []
MISSION_RATIO_3 = []

MISSION_CONDITIONS = [MISSION_CONDITIONS_1, MISSION_CONDITIONS_2, MISSION_CONDITIONS_3]
MISSION_RATIO = [MISSION_RATIO_1, MISSION_RATIO_2, MISSION_RATIO_3]

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
RADIUS = 10

RED_ZONE = (255, 0, 0, 64)
JAMMED_RED_ZONE = (255, 0, 0, 96)
GREEN_ZONE = (0, 255, 0, 64)
BLUE_ZONE = (0, 0, 255, 64)

RANGE_MULTIPLIER = 2


class Fighter:
    FIGHTER_MAX_HEADING_CHANGE_SEC = 3 * np.pi / 180  # rad/sec
    FIGHTER_MIN_FIRING_RANGE = 3.5 * RANGE_MULTIPLIER  # km
    FIGHTER_MAX_FIRING_RANGE = 10.5 * RANGE_MULTIPLIER  # km

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.firing_range = None  # km
        self.weapon_count = None
        self.alive = None

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
    JAMMER_MAX_HEADING_CHANGE_SEC = 3 * np.pi / 180  # rad/sec

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
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
    DECOY_MAX_HEADING_CHANGE_SEC = 5 * np.pi / 180  # rad/sec

    def __init__(self):
        # Observations
        self.heading_sin = None
        self.heading_cos = None
        self.x = None  # km
        self.y = None  # km
        self.alive = None

        # Additional states
        self.heading = None

        # Actions
        self.action = None

        # Specifications
        self.max_heading_change_step = None
        self.max_heading_change_sec = self.DECOY_MAX_HEADING_CHANGE_SEC

        # Rendering


class SAM:
    SAM_MIN_FIRING_RANGE = 5. * RANGE_MULTIPLIER  # km
    SAM_MAX_FIRING_RANGE = 10.5 * RANGE_MULTIPLIER  # km

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
        self.jammed = None

        # Additional states
        self.heading = None

        # Specifications
        self.min_firing_range = self.SAM_MIN_FIRING_RANGE
        self.max_firing_range = self.SAM_MAX_FIRING_RANGE
        self.sskp = 1.0  # Assume perfect, but explicitly not used in the program

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


class MyEnv(gym.Env):
    # For simulation
    SPACE_X = 100.  # km, positive size of battle space
    SPACE_Y = SPACE_X  # km
    SPACE_OFFSET = 20.  # km, negative size of battle space

    # For rendering
    WIDTH = 800
    HEIGHT = WIDTH

    # For screen shot
    SHOT_SHAPE = (80, 80)

    def __init__(self):
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
        self.jammer_1.max_heading_change_step = \
            self.jammer_1.max_heading_change_sec * 3600 * self.dt  # rad/step

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.reset()

        # Define continuous observation space
        min_pos = -self.space_offset / self.space_x
        max_pos = (self.space_x + self.space_offset) / self.space_x

        ## Define fighter low and high
        fighter_low = np.array([0.] * 7, dtype=np.float32)
        fighter_low[0:2] = -1.
        fighter_low[2:4] = min_pos
        fighter_high = np.array([1.] * 7, dtype=np.float32)
        fighter_high[2:4] = max_pos

        ## Define jammer low and high
        jammer_low = np.array([0.] * 6, dtype=np.float32)
        jammer_low[0:2] = -1.
        jammer_low[2:4] = min_pos
        jammer_high = np.array([1.] * 6, dtype=np.float32)
        jammer_high[2:4] = max_pos

        ## Define decoy low and high
        decoy_low = np.array([0.] * 5, dtype=np.float32)
        decoy_low[0:2] = -1.
        decoy_low[2:4] = min_pos
        decoy_high = np.array([1.] * 5, dtype=np.float32)
        decoy_high[2:4] = max_pos

        ## Define sam low and high
        sam_low = np.array([0.] * 8, dtype=np.float32)
        sam_low[0:2] = -1.
        sam_high = np.array([1.] * 8, dtype=np.float32)

        ## Define target low and high
        target_low = np.array([0.] * 5, dtype=np.float32)
        target_low[0:2] = -1.
        target_high = np.array([1.] * 5, dtype=np.float32)

        ## Combine low and high
        low_list = []
        high_list = []

        for idx in self.blue_team:
            if (idx == 'fighter_1') or (idx == 'fighter_2'):
                low_list.append(fighter_low)
                high_list.append(fighter_high)
            if idx == 'jammer_1':
                low_list.append(jammer_low)
                high_list.append(jammer_high)
            if idx == 'decoy_1':
                low_list.append(decoy_low)
                high_list.append(decoy_high)

        for idx in self.red_team:
            if idx == 'sam_1':
                low_list.append(sam_low)
                high_list.append(sam_high)
            if idx == 'target_1':
                low_list.append(target_low)
                high_list.append(target_high)

        low_tuple = tuple(low_list)
        high_tuple = tuple(high_list)

        observation_low = np.hstack(low_tuple)
        observation_high = np.hstack(high_tuple)

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
        fighter.firing_range = 0
        fighter.weapon_count = 1
        fighter.alive = 1

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
        decoy.alive = 1

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
        sam.jammed = 0

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

    def reset_blue_team(self):
        for id in self.blue_team:
            if id == 'fighter_1':
                self.set_initial_condition_fighter(self.fighter_1)
            elif id == 'fighter_2':
                self.set_initial_condition_fighter(self.fighter_2)
            elif id == 'jammer_1':
                self.set_initial_condition_jammer(self.jammer_1)
            else:
                raise Exception('Wrong blue team!')

    def set_initial_condition_fighter(self, fighter):
        """
        c = np.random.rand()
        if c < .5:
            fighter.heading = -3 / 4 * np.pi + 5 * np.pi / 180 * (np.random.rand() - 0.5) * 2
            fighter.x = 50 + 40 / np.sqrt(2) + 0 * np.random.rand()
        else:
            fighter.heading = 3 / 4 * np.pi + 5 * np.pi / 180 * (np.random.rand() - 0.5) * 2
            fighter.x = 50 - 40 / np.sqrt(2) + 0 * np.random.rand()

        fighter.heading_sin = np.sin(fighter.heading)
        fighter.heading_cos = np.cos(fighter.heading)
        fighter.y = 50 + 40 / np.sqrt(2) + 0 * np.random.rand()
        """
        theta = np.pi * (np.random.rand() - 0.5) * 2
        r = 40 + 0 * (np.random.rand() - 0.5) * 2
        fighter.x = 50 + r * np.sin(theta)
        fighter.y = 50 + r * np.cos(theta)

        heading_error = 2 * np.pi / 180 * (np.random.rand() - 0.5) * 2
        if theta > 0:
            fighter.heading = -np.pi + theta + heading_error
        else:
            fighter.heading = np.pi + theta + heading_error

        fighter.heading_sin = np.sin(fighter.heading)
        fighter.heading_cos = np.cos(fighter.heading)

    def set_initial_condition_jammer(self, jammer):
        """
        c = np.random.rand()
        heading_error = 5 * np.pi / 180 * (np.random.rand() - 0.5) * 2
        if c < .5:
            jammer.heading = np.pi + heading_error
            jammer.x = 50 + 0 * np.random.rand()
        else:
            jammer.heading = - np.pi + heading_error
            jammer.x = 50 + 0 * np.random.rand()

        jammer.heading_sin = np.sin(jammer.heading)
        jammer.heading_cos = np.cos(jammer.heading)
        jammer.y = 85 + 10 * np.random.rand()
        """
        theta = np.pi * (np.random.rand() - 0.5) * 2
        r = 40 + 5 * (np.random.rand() - 0.5) * 2
        jammer.x = 50 + r * np.sin(theta)
        jammer.y = 50 + r * np.cos(theta)

        heading_error = 5 * np.pi / 180 * (np.random.rand() - 0.5) * 2
        if theta > 0:
            jammer.heading = -np.pi + theta + heading_error
        else:
            jammer.heading = np.pi + theta + heading_error

        jammer.heading_sin = np.sin(jammer.heading)
        jammer.heading_cos = np.cos(jammer.heading)

    def reset_red_team(self):
        for id in self.red_team:
            if id == 'sam_1':
                self.set_initial_condition_sam(self.sam_1)
            else:
                raise Exception('Wrong red team!')

    def set_initial_condition_sam(self, sam):
        sam.x = 50 + 0 * np.random.rand()
        sam.y = 50 + 0 * np.random.rand()

    def reset_render_blue_team(self):
        for id in self.blue_team:
            if id == 'fighter_1':
                self.reset_render_fighter(self.fighter_1)
            elif id == 'fighter_2':
                self.reset_render_fighter(self.fighter_2)
            elif id == 'jammer_1':
                self.reset_render_jammer(self.jammer_1)

    def reset_render_red_team(self):
        for id in self.red_team:
            if id == 'sam_1':
                self.reset_render_sam(self.sam_1)

    def reset(self):
        self.steps = 0
        self.mission_id = None
        self.mission_condition = None

        # Reset fighter
        self.reset_fighter(self.fighter_1)
        self.reset_fighter(self.fighter_2)

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

    def check_fighter_win(self, fighter, sam):
        x = sam.x - fighter.x
        y = sam.y - fighter.y
        r = (x * x + y * y) ** .5
        if fighter.firing_range >= r:
            sam.alive = 0

    """
    def check_fighter_win_with_jammer(self, fighter, jammer, sam):
        x = sam.x - fighter.x
        y = sam.y - fighter.y
        r = (x * x + y * y) ** .5
        if (jammer.on > 0.5) and (fighter.firing_range >= r):
            sam.alive = 0
    """

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

    def transient(self, entity):
        entity.heading += entity.action
        entity.heading_sin = np.sin(entity.heading)
        entity.heading_cos = np.cos(entity.heading)

        entity.x += entity.speed * entity.heading_sin * self.dt
        entity.y += entity.speed * entity.heading_cos * self.dt

    def step(self, actions):
        n = 0  # ローカル・ステップ
        done = False

        ''' アクション取得 '''
        if self.mission_id == 'mission_1':
            self.fighter_1.action = actions[0] * self.fighter_1.max_heading_change_step
            self.jammer_1.action = actions[1] * self.jammer_1.max_heading_change_step
        else:
            self.fighter_1.action = actions[0] * self.fighter_1.max_heading_change_step
            self.fighter_2.action = actions[1] * self.fighter_2.max_heading_change_step
            self.jammer_1.action = actions[2] * self.jammer_1.max_heading_change_step
            self.decoy_1.action = actions[3] * self.decoy_1.max_heading_change_step

        while (done != True) and (n < self.action_interval):

            ''' 状態遷移(state transition)計算 '''
            for id in self.blue_team:
                if id == 'fighter_1':
                    self.transient(self.fighter_1)
                if id == 'fighter_2':
                    self.transient(self.fighter_2)
                if id == 'jammer_1':
                    self.transient(self.jammer_1)

            # For the future application
            # rgb_shot = self.get_snapshot()

            # Jammerのjam_range内にSAMがいれば、Jammerが有効
            if ('jammer_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_jammer_on(self.jammer_1, self.sam_1)

            # fighter win
            if ('fighter_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_fighter_win(self.fighter_1, self.sam_1)

            if ('fighter_2' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_fighter_win(self.fighter_2, self.sam_1)

            # fighter win with jammer
            """
            if ('fighter_1' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_fighter_win_with_jammer(self.fighter_1, self.jammer_1, self.sam_1)

            if ('fighter_2' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_fighter_win_with_jammer(self.fighter_2, self.jammer_1, self.sam_1)
            """

            # clean sam win to fighter
            if ('fighter_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_fighter(self.fighter_1, self.sam_1)

            if ('fighter_2' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_fighter(self.fighter_2, self.sam_1)

            # clean sam win to jammer
            if ('jammer_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_sam_win_jammer(self.jammer_1, self.sam_1)

            # jammed sam win to fighter
            if ('fighter_1' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_jammed_sam_win_fighter(self.fighter_1, self.sam_1)
            if ('fighter_2' in self.blue_team) and ('jammer_1' in self.blue_team) and \
                    ('sam_1' in self.red_team):
                self.check_jammed_sam_win_fighter(self.fighter_2, self.sam_1)

            # jammed sam win to jammer
            if ('jammer_1' in self.blue_team) and ('sam_1' in self.red_team):
                self.check_jammed_sam_win_jammer(self.jammer_1, self.sam_1)

            # Update blue_team
            if (self.fighter_1.alive < 0.5) and ('fighter_1' in self.blue_team):
                self.blue_team.remove('fighter_1')

            if (self.fighter_2.alive < 0.5) and ('fighter_2' in self.blue_team):
                self.blue_team.remove('fighter_2')

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
        info = {}

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

    def render_sam(self, sam):
        # transform coordinate from battle space to screen
        sam.screen_x = sam.x * self.to_screen_x + self.to_screen_offset
        sam.screen_y = sam.y * self.to_screen_y + self.to_screen_offset

        # draw sam
        screen_x = sam.screen_x - sam.radius
        screen_y = sam.screen_y - sam.radius
        self.screen.blit(sam.surface, (screen_x, screen_y))

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

    def render(self, mode='human'):
        if mode == 'human':
            self.screen.fill((0, 0, 0))

            for id in self.red_team:
                if (id == 'sam_1') and (self.sam_1.alive > .5):
                    self.render_sam(self.sam_1)

            for id in self.blue_team:
                if (id == 'fighter_1') and (self.fighter_1.alive > .5):
                    self.render_fighter(self.fighter_1)
                if (id == 'fighter_2') and (self.fighter_2.alive > .5):
                    self.render_fighter(self.fighter_2)
                if (id == 'jammer_1') and (self.jammer_1.alive > .5):
                    self.render_jammer(self.jammer_1)

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
            obs = self.get_observation_mission_1()
        else:
            obs = self.get_observation_mission_2()

        observation = np.array(obs)  # (38,)
        return observation

    def get_observation_mission_1(self):
        obs = []

        obs.append(self.fighter_1.heading_sin)
        obs.append(self.fighter_1.heading_cos)
        obs.append(self.fighter_1.x / self.space_x)
        obs.append(self.fighter_1.y / self.space_y)
        obs.append(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.weapon_count)
        obs.append(self.fighter_1.alive)

        obs.append(self.jammer_1.heading_sin)
        obs.append(self.jammer_1.heading_cos)
        obs.append(self.jammer_1.x / self.space_x)
        obs.append(self.jammer_1.y / self.space_y)
        obs.append(self.jammer_1.alive)
        obs.append(self.jammer_1.on)

        obs.append(self.sam_1.heading_sin)
        obs.append(self.sam_1.heading_cos)
        obs.append(self.sam_1.x / self.space_x)
        obs.append(self.sam_1.y / self.space_y)
        obs.append(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(
            self.sam_1.jammed_firing_range / (self.sam_1.max_firing_range * self.jammer_1.jam_effectiveness))
        obs.append(self.sam_1.weapon_count)
        obs.append(self.sam_1.alive)

        return obs

    def get_observation_mission_2(self):
        obs = []

        obs.append(self.fighter_1.heading_sin)
        obs.append(self.fighter_1.heading_cos)
        obs.append(self.fighter_1.x / self.space_x)
        obs.append(self.fighter_1.y / self.space_y)
        obs.append(self.fighter_1.firing_range / self.fighter_1.max_firing_range)
        obs.append(self.fighter_1.weapon_count)
        obs.append(self.fighter_1.alive)

        obs.append(self.fighter_2.heading_sin)
        obs.append(self.fighter_2.heading_cos)
        obs.append(self.fighter_2.x / self.space_x)
        obs.append(self.fighter_2.y / self.space_y)
        obs.append(self.fighter_2.firing_range / self.fighter_2.max_firing_range)
        obs.append(self.fighter_2.weapon_count)
        obs.append(self.fighter_2.alive)

        obs.append(self.jammer_1.heading_sin)
        obs.append(self.jammer_1.heading_cos)
        obs.append(self.jammer_1.x / self.space_x)
        obs.append(self.jammer_1.y / self.space_y)
        obs.append(self.jammer_1.alive)
        obs.append(self.jammer_1.on)

        obs.append(self.decoy_1.heading_sin)
        obs.append(self.decoy_1.heading_cos)
        obs.append(self.decoy_1.x / self.space_x)
        obs.append(self.decoy_1.y / self.space_y)
        obs.append(self.decoy_1.alive)

        obs.append(self.sam_1.heading_sin)
        obs.append(self.sam_1.heading_cos)
        obs.append(self.sam_1.x / self.space_x)
        obs.append(self.sam_1.y / self.space_y)
        obs.append(self.sam_1.firing_range / self.sam_1.max_firing_range)
        obs.append(
            self.sam_1.jammed_firing_range / (self.sam_1.max_firing_range * self.jammer_1.jam_effectiveness))
        obs.append(self.sam_1.weapon_count)
        obs.append(self.sam_1.alive)

        obs.append(self.target_1.heading_sin)
        obs.append(self.target_1.heading_cos)
        obs.append(self.target_1.x / self.space_x)
        obs.append(self.target_1.y / self.space_y)
        obs.append(self.target_1.alive)

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

    def get_reward(self, done):
        if self.mission_id == 'mission_1':
            reward = self.get_reward_mission_1(done, self.fighter_1, self.jammer_1, self.sam_1)

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

    def is_done(self):
        done = False

        if self.mission_id == 'mission_1':
            done = self.is_done_mission_1(self.fighter_1, self.jammer_1, self.sam_1)

        if (not self.blue_team) or (not self.red_team):
            done = True

        return done
