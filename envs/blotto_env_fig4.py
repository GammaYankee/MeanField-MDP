from envs.mean_field_env import MeanFieldEnv
import numpy as np

mu_0 = np.array([0.3, 0.2, 0.1, 0.4]).T

connectivity = np.array([[1, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 1]])


def generate_transitions(connectivity):
    n_states = connectivity.shape[0]
    n_actions = []
    for s in range(n_states):
        n_actions.append(int(sum(connectivity[s, :])))

    max_n_actions = max(n_actions)

    T = [np.zeros((n_states, n_states)) for a in range(max_n_actions)]

    for s in range(n_states):
        action_index = 0
        for s_prime in range(n_states):
            if connectivity[s, s_prime] == 1:
                T[action_index][s, s_prime] = 1
                action_index += 1

    return n_states, n_actions, T