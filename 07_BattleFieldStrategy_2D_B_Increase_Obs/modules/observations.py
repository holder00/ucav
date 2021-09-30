import numpy as np


def get_observation(env):
    # observations = get_observation_1(env)
    # observations = get_observation_2(env)
    # observations = get_observation_3(env)
    # observations = get_observation_4(env)
    observations = get_observation_5(env)

    return observations


"""
def get_observation_1(env):
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            obs = np.array([env.red.pos[i] / (env.grid_size - 1), env.red.force[i] / env.max_force])

            teammate_id = [j for j in range(env.num_red) if j != i]
            teammate_pos = env.red.pos[teammate_id] / (env.grid_size - 1)
            teammate_force = env.red.force[teammate_id] / env.max_force
            teammate = np.vstack([teammate_pos, teammate_force]).T.flatten()
            obs = np.append(obs, teammate)

            enemy = np.vstack([env.blue.pos / (env.grid_size - 1), env.blue.force / env.max_force])
            obs = np.append(obs, enemy)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation
"""


def get_observation_2(env):
    """
    observation maps: 3ch [agent, team_mates, adversaries]
    """
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            my_matrix = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix = np.zeros((env.grid_size, env.grid_size))

            # my position & force map
            my_matrix[env.red.pos[i][0], env.red.pos[i][1]] += env.red.force[i]

            # teammate position & force map
            teammate_id = [j for j in range(env.num_red) if j != i]
            # teammate_pos = env.red.pos[teammate_id]
            # teammate_force = env.red.force[teammate_id]
            for j in teammate_id:
                teammate_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]

            # adversarial position & force map
            for j in range(env.num_blue):
                adversarial_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]

            # stack the maps
            my_matrix = np.expand_dims(my_matrix, axis=2)
            teammate_matrix = np.expand_dims(teammate_matrix, axis=2)
            adversarial_matrix = np.expand_dims(adversarial_matrix, axis=2)

            # normalize the maps
            my_matrix = my_matrix / env.max_force
            teammate_matrix = teammate_matrix / env.max_force
            adversarial_matrix = adversarial_matrix / env.max_force

            obs = np.concatenate([my_matrix, teammate_matrix, adversarial_matrix], axis=-1)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation


def get_observation_3(env):
    """
    Observation maps : 3ch → 6ch
    Maps of the position and force are divided.
    """
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            my_matrix = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix = np.zeros((env.grid_size, env.grid_size))

            my_matrix_pos = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix_pos = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix_pos = np.zeros((env.grid_size, env.grid_size))

            # my position & force map
            my_matrix_pos[env.red.pos[i][0], env.red.pos[i][1]] += 1
            my_matrix[env.red.pos[i][0], env.red.pos[i][1]] += env.red.force[i]

            # teammate position & force map
            teammate_id = [j for j in range(env.num_red) if j != i]
            # teammate_pos = env.red.pos[teammate_id]
            # teammate_force = env.red.force[teammate_id]
            for j in teammate_id:
                if env.red.alive[j]:
                    teammate_matrix_pos[env.red.pos[j][0], env.red.pos[j][1]] += 1
                teammate_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]

            # adversarial position & force map
            for j in range(env.num_blue):
                if env.blue.alive[j]:
                    adversarial_matrix_pos[env.blue.pos[j][0], env.blue.pos[j][1]] += 1
                adversarial_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]

            # stack the maps
            my_matrix_pos = np.expand_dims(my_matrix_pos, axis=2)
            my_matrix = np.expand_dims(my_matrix, axis=2)
            teammate_matrix_pos = np.expand_dims(teammate_matrix_pos, axis=2)
            teammate_matrix = np.expand_dims(teammate_matrix, axis=2)
            adversarial_matrix_pos = np.expand_dims(adversarial_matrix_pos, axis=2)
            adversarial_matrix = np.expand_dims(adversarial_matrix, axis=2)

            # normalize the maps
            my_matrix = my_matrix / env.max_force
            teammate_matrix = teammate_matrix / env.max_force
            adversarial_matrix = adversarial_matrix / env.max_force

            # Normalize teammate_matrix_pos & adversarial_matrix_pos by number of agents
            teammate_matrix_pos = teammate_matrix_pos / env.num_red
            adversarial_matrix_pos = adversarial_matrix_pos / env.num_blue

            obs = np.concatenate([my_matrix_pos, my_matrix,
                                  teammate_matrix_pos, teammate_matrix,
                                  adversarial_matrix_pos, adversarial_matrix], axis=-1)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation


def get_observation_4(env):
    """
    Observation maps : 3ch → 6ch
    Maps of the position and force are diveided.
    """
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            my_matrix = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix = np.zeros((env.grid_size, env.grid_size))

            my_matrix_pos = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix_pos = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix_pos = np.zeros((env.grid_size, env.grid_size))

            # my position & force map
            my_matrix_pos[env.red.pos[i][0], env.red.pos[i][1]] = 1
            my_matrix[env.red.pos[i][0], env.red.pos[i][1]] += env.red.force[i]

            # teammate position & force map
            teammate_id = [j for j in range(env.num_red) if j != i]
            # teammate_pos = env.red.pos[teammate_id]
            # teammate_force = env.red.force[teammate_id]
            for j in teammate_id:
                if env.red.alive[j]:
                    teammate_matrix_pos[env.red.pos[j][0], env.red.pos[j][1]] = 1
                teammate_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]

            # adversarial position & force map
            for j in range(env.num_blue):
                if env.blue.alive[j]:
                    adversarial_matrix_pos[env.blue.pos[j][0], env.blue.pos[j][1]] = 1
                adversarial_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]

            # stack the maps
            my_matrix_pos = np.expand_dims(my_matrix_pos, axis=2)
            my_matrix = np.expand_dims(my_matrix, axis=2)
            teammate_matrix_pos = np.expand_dims(teammate_matrix_pos, axis=2)
            teammate_matrix = np.expand_dims(teammate_matrix, axis=2)
            adversarial_matrix_pos = np.expand_dims(adversarial_matrix_pos, axis=2)
            adversarial_matrix = np.expand_dims(adversarial_matrix, axis=2)

            # normalize the maps
            my_matrix = my_matrix / env.max_force
            teammate_matrix = teammate_matrix / env.max_force
            adversarial_matrix = adversarial_matrix / env.max_force

            obs = np.concatenate([my_matrix_pos, my_matrix,
                                  teammate_matrix_pos, teammate_matrix,
                                  adversarial_matrix_pos, adversarial_matrix], axis=-1)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation


def get_observation_5(env):
    """
    Observation maps : 3ch → 6ch
    Only normalization is changed from get_observation_3
    Normalized by current number of team mates
    """
    observation = {}

    for i in range(env.num_red):
        if env.red.alive[i]:
            my_matrix = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix = np.zeros((env.grid_size, env.grid_size))

            my_matrix_pos = np.zeros((env.grid_size, env.grid_size))
            teammate_matrix_pos = np.zeros((env.grid_size, env.grid_size))
            adversarial_matrix_pos = np.zeros((env.grid_size, env.grid_size))

            # my position & force map
            my_matrix_pos[env.red.pos[i][0], env.red.pos[i][1]] += 1
            my_matrix[env.red.pos[i][0], env.red.pos[i][1]] += env.red.force[i]

            # teammate position & force map
            teammate_id = [j for j in range(env.num_red) if j != i]
            # teammate_pos = env.red.pos[teammate_id]
            # teammate_force = env.red.force[teammate_id]
            for j in teammate_id:
                if env.red.alive[j]:
                    teammate_matrix_pos[env.red.pos[j][0], env.red.pos[j][1]] += 1
                teammate_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]
                # Don't care because env.red.force[j]=0 if not env.red.alive[j]

            # adversarial position & force map
            for j in range(env.num_blue):
                if env.blue.alive[j]:
                    adversarial_matrix_pos[env.blue.pos[j][0], env.blue.pos[j][1]] += 1
                adversarial_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]
                # Don't care because env.blue.force[j]=0 if not env.blue.alive[j]

            # stack the maps
            my_matrix_pos = np.expand_dims(my_matrix_pos, axis=2)
            my_matrix = np.expand_dims(my_matrix, axis=2)
            teammate_matrix_pos = np.expand_dims(teammate_matrix_pos, axis=2)
            teammate_matrix = np.expand_dims(teammate_matrix, axis=2)
            adversarial_matrix_pos = np.expand_dims(adversarial_matrix_pos, axis=2)
            adversarial_matrix = np.expand_dims(adversarial_matrix, axis=2)

            # normalize the maps
            my_matrix = my_matrix / env.max_force
            teammate_matrix = teammate_matrix / env.max_force
            adversarial_matrix = adversarial_matrix / env.max_force

            # Normalize teammate_matrix_pos & adversarial_matrix_pos by current number of agents
            teammate_matrix_pos = teammate_matrix_pos / np.sum(env.red.alive)
            adversarial_matrix_pos = adversarial_matrix_pos / np.sum(env.blue.alive)

            obs = np.concatenate([my_matrix_pos, my_matrix,
                                  teammate_matrix_pos, teammate_matrix,
                                  adversarial_matrix_pos, adversarial_matrix], axis=-1)

            observation['red_' + str(i)] = obs.astype(np.float32)
    return observation
