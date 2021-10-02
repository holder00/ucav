import numpy as np
import matplotlib.pyplot as plt


def main():
    trial = np.load('./myenv-v2-sac/learning_history/learning_history.npz')
    episode = trial['arr_0']
    success = trial['arr_1']

    plt.plot(episode, success * 100, label='success', linewidth=2, color='red')
    plt.legend()
    plt.title(f'Training history mission_3: success ratio')
    plt.xlabel('training iteration')
    plt.ylabel('average ratio [%] over 1000 episodes')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
