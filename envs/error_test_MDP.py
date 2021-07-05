import numpy as np
from envs.MDP_env import MDPEnv
from envs.mean_field_env import MeanFieldEnv
from utils import empirical_dist

class FinitePopMDPEnv(MDPEnv):
    def __init__(self, n_agents, mean_field_env: MeanFieldEnv, mean_field_policy):
        self.mean_field_env = mean_field_env
        self.mu_0 = self.mean_field_env.mu0
        self.n_agents = n_agents
        self.mean_field_policy = mean_field_policy
        super(FinitePopMDPEnv, self).__init__(self.mean_field_env.n_states, self.mean_field_env.n_actions,
                                              self.mean_field_env.Tf, self.mean_field_env.gamma)
        self.emp_dist_flow = self.compute_emp_dist_flow()


    def _init_transitions(self):
        self.T = self.mean_field_env.T



    def reward(self, s, a, t):
        emp_dist = self.emp_dist_flow[t]
        r = 0
        for case in emp_dist:
            n_list = case.n_list


    def compute_emp_dist_flow(self):
        emp_dist_flow = []
        mean_field_flow = self.compute_mean_field_flow()
        for t in range(len(mean_field_flow)):
            mu_t = mean_field_flow[t]
            emp_dist_flow.append(empirical_dist(N=self.n_agents, p=mu_t))

        return emp_dist_flow

    def compute_mean_field_flow(self):
        mu = [self.mu_0]
        for t in range(self.Tf + 1):
            pi_t = self.mean_field_policy[t - 1]
            controlled_trans = self.mean_field_controlled_trans(pi_t)
            mu.append(np.matmul(mu[-1], controlled_trans))
        return mu

    def mean_field_controlled_trans(self, pi_t):
        T_c = np.zeros((self.n_states, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions[s]):
                T_c[s, :] += self.T[a][s] * pi_t[s][a]

        assert all(abs(np.sum(T_c, axis=1) - 1) < 1e-5)

        return T_c
