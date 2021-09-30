"""
Read the learning results and make some plots.
"""
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from settings.initial_settings import *


def main():
    TRIALS = ["fwd_10059", "fwd_10060"]

    results_dir = os.path.join('./' + PROJECT + '/results/')
    file_dir = results_dir + TRIALS[0] + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    steps_history = []
    success_history = []
    fail_history = []
    draw_history = []
    alive_red_history = []
    alive_blue_history = []
    red_force_history = []
    blue_force_history = []
    episode_length_history = []

    for TRIAL in TRIALS:
        results_file = results_dir + TRIAL + '.pkl'
        print(results_file)

        file = open(results_file, 'rb')
        if len(steps_history) > 0:
            new_step = pickle.load(file)
            new_step = [x + steps_history[-1] for x in new_step]
            steps_history.extend(new_step)
        else:
            steps_history.extend(pickle.load(file))
        success_history.extend(pickle.load(file))
        fail_history.extend(pickle.load(file))
        draw_history.extend(pickle.load(file))
        alive_red_history.extend(pickle.load(file))
        alive_blue_history.extend(pickle.load(file))
        red_force_history.extend(pickle.load(file))
        blue_force_history.extend(pickle.load(file))
        episode_length_history.extend(pickle.load(file))

        file.close()

    """
    print(steps_history)
    print(success_history)
    print(fail_history)
    print(draw_history)
    print(alive_red_history)
    print(alive_blue_history)
    print(red_force_history)
    print(blue_force_history)
    """

    plt.plot(steps_history, success_history, label='success', color='r')
    plt.plot(steps_history, fail_history, label='fail', color='b')
    plt.plot(steps_history, draw_history, label='draw', color='g')
    plt.title(f'Number of success, fail, and draw during the training: {TRIAL}')
    plt.xlabel('steps')
    plt.ylabel('success, fail, draw episodes number')
    plt.ylim(0, 100)
    plt.grid()
    plt.legend()
    # plt.show()
    file_name = file_dir + 'success_ratio.png'
    plt.savefig(file_name)
    plt.close()

    plt.plot(steps_history, alive_red_history, label='survived_red', color='r')
    plt.plot(steps_history, alive_blue_history, label='survived_blue', color='b')
    plt.title(f'Number of survived agents during the training: {TRIAL}')
    plt.xlabel('steps')
    plt.ylabel('mean survived agents number')
    plt.ylim(0, np.max([NUM_RED_MAX, NUM_BLUE_MAX]))
    plt.grid()
    plt.legend()
    # plt.show()
    file_name = file_dir + 'survived_agents.png'
    plt.savefig(file_name)
    plt.close()

    plt.plot(steps_history, red_force_history, label='red_force', color='r')
    plt.plot(steps_history, blue_force_history, label='blue_force', color='b')
    plt.title(f'Survived total force of agents during the training: {TRIAL}')
    plt.xlabel('steps')
    plt.ylabel('mean total force of survived agents')
    plt.ylim(0, np.max([RED_TOTAL_FORCE, BLUE_TOTAL_FORCE]) / 2)
    plt.grid()
    plt.legend()
    # plt.show()
    file_name = file_dir + 'survived_force.png'
    plt.savefig(file_name)
    plt.close()

    plt.plot(steps_history, episode_length_history)
    plt.title(f'Episode length during the training: {TRIAL}')
    plt.xlabel('steps')
    plt.ylabel('mean episode_length')
    plt.ylim(0, MAX_STEPS * 1.1)
    plt.grid()
    file_name = file_dir + 'episode_length.png'
    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':
    main()
