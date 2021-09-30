import numpy as np
import matplotlib.pyplot as plt


def main():
    ID = 3
    #LEN = 12

    file_name = 'trial_' + str(ID)
    trial = np.load('./myenv-v1-sac/learning_history/' + file_name + '.npz')
    """
    episode = trial['iteration_history'][:LEN]
    success = trial['success_history'][:LEN]
    f1 = trial['f1_history'][:LEN]
    f2 = trial['f2_history'][:LEN]
    s1 = trial['s1_history'][:LEN]
    s2 = trial['s2_history'][:LEN]
    s3 = trial['s3_history'][:LEN]
    """
    episode = trial['iteration_history']
    success = trial['success_history']
    f1 = trial['f1_history']
    f2 = trial['f2_history']
    s1 = trial['s1_history']
    s2 = trial['s2_history']
    s3 = trial['s3_history']

    plt.plot(episode, success * 100, label='success', linewidth=3, color='red')
    plt.plot(episode, f1 * 100, label='F1', linewidth=2, linestyle="--")
    plt.plot(episode, s1 * 100, label='S1', linewidth=2, linestyle=":")
    plt.plot(episode, s2 * 100, label='S2', linewidth=2, linestyle=":")
    plt.plot(episode, f2 * 100, label='F2', linewidth=2, linestyle="--")
    plt.plot(episode, s3 * 100, label='S3', linewidth=2, linestyle=":")

    plt.legend()
    plt.title(f'Training history mission_2: {file_name}')
    plt.xlabel('training iteration')
    plt.ylabel('average ratio [%] over 1000 episodes')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
