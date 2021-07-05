import numpy as np


class MeanFieldEnv():
    def __init__(self, n_states, n_actions, Tf, gamma=0.9):
        '''
        Initialize number of states, number of actions
        :param n_states (Int): number of states
        :param n_actions (List of Int): List of number of actions available at each state
        :param Tf (Int): terminal time
        '''
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = self._init_transitions()
        self.Tf = Tf
        self.mu0 = None
        self.gamma = gamma
        self.dim_nu = sum(self.n_actions)           # dimension of the state-action mean field

    def set_init_mu(self, mu0):
        self.mu0 = mu0

    def _init_transitions(self):
        raise NotImplemented

    def individual_reward(self, s_t, a_t, nu_t, t):
        r = 0
        mu_t = self.nu2mu(nu_t)
        for s_prime in range(self.n_states):
            r += self.pairwise_reward(s_t, a_t, s_prime, t) * mu_t[s_prime]

        return self.theta(r, t)

    def theta(self, x, t):
        raise NotImplemented

    def pairwise_reward(self, s, a, s_prime, t):
        raise NotImplemented

    def state_action2index(self, s, a):
        index = s * self.n_actions + a
        assert (0 <= index < self.n_actions * self.n_states)
        return index

    def index2state_action(self, index):
        a = index % self.n_actions
        s = (index - a) / self.n_states
        return s, a

    def nu_vec2nu_matrix(self, nu_vec: np.ndarray):
        assert nu_vec.shape[1] == 1
        nu = nu_vec.reshape((self.n_states, self.n_actions))
        return nu

    def nu2mu(self, nu_vec):
        nu = self.nu_vec2nu_matrix(nu_vec)
        mu = np.sum(nu, axis=1)
        assert len(mu) == self.n_states
        return mu

    def nu2alpha(self, nu_vec):
        nu = self.nu_vec2nu_matrix(nu_vec)
        alpha = np.sum(nu, axis=0)
        assert len(alpha) == self.n_actions
        return nu
