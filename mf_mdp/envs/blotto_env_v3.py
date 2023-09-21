from envs.mean_field_env import MeanFieldEnv
import numpy as np

# Graph information
mu_0 = np.array([1, 0, 0, 0, 0]).T

# Connectivity
connectivity = np.array([[1, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 1]])
# Terminal Time
Tf = 5

prior_1_t = [[0.01, 0.98, 0.01], [0.01, 0.01, 0.98], [0.01, 0.99], [0.01, 0.98, 0.01], [0.99, 0.01]]
prior_1 = [prior_1_t for _ in range(Tf + 1)]


def generate_actions(connectivity):
    n_states = connectivity.shape[0]
    n_actions = [sum(connectivity[i, :]) for i in range(n_states)]
    max_n_actions = max(n_actions)

    T = [[] for a in range(max_n_actions)]

    for state in range(n_states):
        action = 0
        for state_g in range(n_states):
            if connectivity[state, state_g] > 0:
                action_vec = np.zeros(n_states)
                action_vec[state_g] = 1
                T[action].append(action_vec)
                action += 1

        while action < max_n_actions:
            T[action].append([])
            action += 1

    return T, n_states, n_actions


class BlottoEnv3(MeanFieldEnv):
    def __init__(self, mu_0):
        T, n_states, n_actions = generate_actions(connectivity)
        super(BlottoEnv3, self).__init__(n_states, n_actions, Tf)
        self.set_init_mu(mu_0)
        assert len(self.T) == max(self.n_actions)
        self.prior = prior_1

    def _init_transitions(self):
        T, n_states, n_actions = generate_actions(connectivity)
        return T

    def individual_reward(self, s_t, a_t, nu_t, t):
        if t < Tf:
            return 0
        elif t == Tf:
            mu_t = self.nu2mu(nu_t)
            reward = 0
            reward -= mu_t[s_t]
            if s_t == 2:
                reward += 1.5
            elif s_t == 3:
                reward += 1
            return reward

        else:
            raise Exception('time step error!')

    def nu2mu(self, nu_vec):
        mu = np.zeros(self.n_states)
        pointer = 0
        for s in range(self.n_states):
            for action in range(self.n_actions[s]):
                mu[s] += nu_vec[pointer]
                pointer += 1

        assert pointer == sum(self.n_actions)
        return mu


if __name__ == "__main__":
    prior_t, n_states, n_actions = generate_actions(connectivity)

    print(prior_t, n_actions)
