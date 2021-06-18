import numpy as np
from solvers.MDP_solver import MDP_Solver
from env.base_env import MeanFieldEnv
import copy


class MFSolver:
    def __init__(self, env: MeanFieldEnv, n_ittr=10, eps=0.01):
        self.env = env
        self.n_ittr = n_ittr
        self.eps = eps
        self.mdp_solver = MDP_Solver(env)
        self.soln = None
        np.random.seed(0)

    def solve(self):
        diff1, diff2 = 1, 1

        # randomly initialize policy
        policy = self._init_policy()

        # intialize mean field
        nu = [np.zeros(self.env.dim_nu) for _ in range(self.env.Tf + 1)]
        mu = [np.zeros(self.env.n_states) for _ in range(self.env.Tf + 1)]

        for _ in range(self.n_ittr):
            if diff1 < self.eps or diff2 < self.eps:
                break
            nu_, mu_ = self.propogate_mean_field(policy)  # propagate mean field

            policy = self.mdp_solver.solve(nu_)

            diff1 = sum(sum(abs(nu_[t] - nu[t])) for t in range(self.env.Tf + 1))
            diff2 = sum(sum(abs(mu_[t] - mu[t])) for t in range(self.env.Tf + 1))
            nu = copy.deepcopy(nu_)
            mu = copy.deepcopy(mu_)

        self.soln = {"policy": policy, "mu": mu, "nu": nu}

    def propogate_mean_field(self, policy):
        mu = [np.zeros(self.env.n_states) for _ in range(self.env.Tf + 1)]
        mu[0] = self.env.mu0

        nu = [np.zeros(self.env.dim_nu) for _ in range(self.env.Tf + 1)]
        nu[0] = self.mu2nu(mu_t=mu[0], pi_t=policy[0])

        for t in range(0, self.env.Tf):
            mu_t = mu[t]
            T_c = self.controlled_trans(pi_t=policy[t])
            mu_next = np.matmul(mu_t, T_c)
            mu[t + 1] = mu_next

            nu_next = self.mu2nu(mu_t=mu_next, pi_t=policy[t + 1])
            nu[t + 1] = nu_next

        return nu, mu

    def mu2nu(self, mu_t, pi_t):
        '''
        Compute state action mean field from state mean field given a policy at time t
        :param mu_t: state mean field at t
        :param pi_t: policy at time t
        :return:
        '''
        nu_t = np.zeros(self.env.dim_nu)
        pointer = 0
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions[s]):
                nu_t[pointer] = mu_t[s] * pi_t[s][a]
                pointer += 1

        assert pointer == self.env.dim_nu
        return nu_t

    def controlled_trans(self, pi_t):
        T_c = np.zeros(self.env.n_states, self.env.n_states)
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions[s]):
                T_c[s, :] += self.env.T[a][s] * pi_t[s][a]

        assert all(np.sum(T_c, axis=0) == 1)

        return T_c

    def _init_policy(self):
        policy = []
        for t in range(self.env.Tf):
            policy_t = []
            for s in range(self.env.n_states):
                tmp = np.random.random(self.env.n_states[s])
                tmp = tmp / sum(tmp)
                policy_t.append(tmp)
            policy.append(policy_t)
        return policy
