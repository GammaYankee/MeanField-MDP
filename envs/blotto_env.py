from envs.mean_field_env import MeanFieldEnv
import numpy as np

# Graph information
N_STATES = 4
N_ACTIONS = 3
mu_0 = np.array([0, 0, 0, 1]).T

# Transitions
T1 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])  # Go CCW
T2 = np.array([[0, 0, 0, 1]], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])  # Go CW
T3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # Stay

# Terminal Time
Tf = 10


class BlottoEnv(MeanFieldEnv):
    def __init__(self, mu_0):
        super(BlottoEnv, self).__init__(N_STATES, N_ACTIONS, Tf)
        self.set_init_mu(mu_0)

    def _init_transitions(self):
        self.T = [T1, T2, T3]
        assert len(self.T) == self.n_actions

    def individual_reward(self, s, a, nu_t, t):
        if t < Tf:
            return 0
        elif t == Tf:
            mu_t = self.nu2mu(nu_t)
            return mu_t[s]

        else:
            raise Exception('time step error!')




