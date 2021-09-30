import numpy as np
import matplotlib.pyplot as plt


def main():
    trial = np.load('./myenv-v5/learning_history/trial_2.npz')
    episode = trial['arr_0']
    success = trial['arr_1']
    plt.plot(episode, success * 100, label='No reward shaping, N=40', linewidth=2, color='red')

    trial = np.load('./myenv-v5/learning_history/trial_12.npz')
    episode = trial['arr_0']
    success = trial['arr_1']
    plt.plot(episode, success * 100, label='Reward shaping, N=40', linewidth=2, color='blue')

    trial = np.load('./myenv-v5/learning_history/trial_100.npz')
    episode = trial['arr_0']
    success = trial['arr_1']
    plt.plot(episode, success * 100, label='Reward shaping, N=20', linewidth=2, color='green')

    trial = np.load('./myenv-v5/learning_history/trial_101.npz')
    episode = trial['arr_0']
    success = trial['arr_1']
    plt.plot(episode, success * 100, label='Reward shaping, N=10', linewidth=2, color='black')

    plt.legend()
    plt.title(f'Training history mission_4: success ratio')
    plt.xlabel('training iteration')
    plt.ylabel('average ratio [%] over 100 episodes')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
