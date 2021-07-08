from envs.mean_field_env import MeanFieldEnv
import numpy as np
import math

# Graph Information
N_STATES = 4
N_ACTIONS = [2, 2, 2, 2]

# Transitions
np.random.seed(0)
T1 = [np.random.random(N_STATES) for _ in range(N_STATES)]
T1 = [T1[k] / sum(T1[k]) for k in range(N_STATES)]

T2 = [np.random.random(N_STATES) for _ in range(N_STATES)]
T2 = [T2[k] / sum(T2[k]) for k in range(N_STATES)]

# Terminal Time
Tf = 5


class TestEnv(MeanFieldEnv):
    def __init__(self, mu0):
        super(TestEnv, self).__init__(N_STATES, N_ACTIONS, Tf, gamma=1)
        self.set_init_mu(mu0)

    def _init_transitions(self):
        return [T1, T2]

    def pairwise_reward(self, s, a, s_prime, t):
        return s ** 2 + s_prime ** 2

    def theta(self, x, t):
        if t == self.Tf:
            return x
        else:
            return 0

    def state_action2index(self, s, a):
        index = sum(self.n_actions[k] for k in range(s)) + a
        assert (0 <= index < sum(self.n_actions))
        return index

    def index2state_action(self, index):
        for k in range(self.n_states):
            if index >= sum(self.n_actions[i] for i in range(k)):
                s = k - 1
                a = index - sum(self.n_actions[i] for i in range(k - 1))
                return s, a
        raise Exception('index issues!')

    def nu2mu(self, nu_vec):
        mu = np.zeros(self.n_states)
        pointer = 0
        for s in range(self.n_states):
            for action in range(self.n_actions[s]):
                mu[s] += nu_vec[pointer]
                pointer += 1

        assert pointer == sum(self.n_actions)
        return mu

    # def individual_reward(self, s_t, nu_t, t, a_t=None):
    #     if t < self.Tf:
    #         return 0  # no running reward
    #     elif t == Tf:
    #         mu_t = self.nu2mu(nu_t)
    #         r = 0
    #         for s_prime in range(self.n_states):
    #             r += self.pairwise_reward(s_t, a_t, s_prime, t) * mu_t[s_prime]
    #         return r
    #     else:
    #         raise Exception('time step error!')