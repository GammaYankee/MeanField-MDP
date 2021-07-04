from envs.base_env import MeanFieldEnv
import numpy as np

# Graph Information
N_STATES = 5
N_ACTIONS = [1, 2, 2, 1, 1]

# Transitions
T1 = [np.array([0, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 1]),
      np.array([0, 0, 1, 0, 0])]
T2 = [[], np.array([0, 0, 1, 0, 0]), np.array([0, 0, 0, 1, 0]), [], [], []]

# Terminal Time
Tf = 5


class RotateConfEnv(MeanFieldEnv):
    def __init__(self, mu0):
        super(RotateConfEnv, self).__init__(N_STATES, N_ACTIONS, Tf, gamma=1)
        self.set_init_mu(mu0)

    def _init_transitions(self):
        return [T1, T2]

    def individual_reward(self, s_t, nu_t, t, a_t=None):
        if t < self.Tf:
            return 0  # no running reward
        elif t == Tf:
            mu_t = self.nu2mu(nu_t)
            return mu_t[s_t]

        else:
            raise Exception('time step error!')

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


