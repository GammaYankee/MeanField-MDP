import numpy as np
from solvers.MDP_solver import MDP_Solver
from envs.mean_field_env import MeanFieldEnv
from envs.MDP_env import MDPEnv
import copy


class MFSolver:
    def __init__(self, env: MeanFieldEnv, n_ittr=10, eps=0.0001):
        self.mean_field_env = env
        self.n_ittr = n_ittr
        self.eps = eps
        self.soln = None
        np.random.seed(0)

    def solve(self, entropy_regularized=False, beta=1):
        diff1, diff2 = 1, 1

        # randomly initialize policy
        policy = self._init_policy()
        value = None

        # intialize mean field
        nu = [np.zeros(self.mean_field_env.dim_nu) for _ in range(self.mean_field_env.Tf + 1)]
        mu = [np.zeros(self.mean_field_env.n_states) for _ in range(self.mean_field_env.Tf + 1)]
        mdp_env = None

        for ittr in range(self.n_ittr):
            if diff1 < self.eps or diff2 < self.eps:
                break
            nu_, mu_ = self.propagate_mean_field(policy)  # propagate mean field

            mdp_env = self._generate_MDP_env(nu_)

            mdp_solver = MDP_Solver(env=mdp_env)

            if entropy_regularized:
                policy, value = mdp_solver.solve_entropy(prior=None, beta=beta)
            else:
                policy, value = mdp_solver.solve()

            diff1 = sum(sum(abs(nu_[t] - nu[t])) for t in range(self.mean_field_env.Tf + 1))
            diff2 = sum(sum(abs(mu_[t] - mu[t])) for t in range(self.mean_field_env.Tf + 1))
            nu = copy.deepcopy(nu_)
            mu = copy.deepcopy(mu_)

            print("Differene 1 is {}, difference 2 is {}.".format(diff1, diff2))

        self.soln = {"policy": policy, "value": value, "mu": mu, "nu": nu, "MDP_induced": mdp_env}

    def propagate_mean_field(self, policy):
        mu = [np.zeros(self.mean_field_env.n_states) for _ in range(self.mean_field_env.Tf + 1)]
        mu[0] = self.mean_field_env.mu0

        nu = [np.zeros(self.mean_field_env.dim_nu) for _ in range(self.mean_field_env.Tf + 1)]
        nu[0] = self.mu2nu(mu_t=mu[0], pi_t=policy[0])

        for t in range(0, self.mean_field_env.Tf):
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
        nu_t = np.zeros(self.mean_field_env.dim_nu)
        pointer = 0
        for s in range(self.mean_field_env.n_states):
            for a in range(self.mean_field_env.n_actions[s]):
                nu_t[pointer] = mu_t[s] * pi_t[s][a]
                pointer += 1

        assert pointer == self.mean_field_env.dim_nu
        return nu_t

    def controlled_trans(self, pi_t):
        T_c = np.zeros((self.mean_field_env.n_states, self.mean_field_env.n_states))
        for s in range(self.mean_field_env.n_states):
            for a in range(self.mean_field_env.n_actions[s]):
                T_c[s, :] += self.mean_field_env.T[a][s] * pi_t[s][a]

        assert all(abs(np.sum(T_c, axis=1) - 1) < 1e-5)

        return T_c

    def _init_policy(self):
        policy = []
        for t in range(self.mean_field_env.Tf + 1):
            policy_t = []
            for s in range(self.mean_field_env.n_states):
                tmp = np.random.random(self.mean_field_env.n_actions[s])
                tmp = tmp / sum(tmp)
                policy_t.append(tmp)
            policy.append(policy_t)
        return policy

    def _generate_MDP_env(self, nu):
        n_actions = max(self.mean_field_env.n_actions)

        reward_vec = [[] for t in range(self.mean_field_env.Tf + 1)]

        for t in range(self.mean_field_env.Tf + 1):
            for s in range(self.mean_field_env.n_states):
                reward_s = []
                for a in range(self.mean_field_env.n_actions[s]):
                    reward_s.append(self.mean_field_env.individual_reward(s_t=s, a_t=a, nu_t=nu[t], t=t))
                reward_vec[t].append(reward_s)

        mdp_env = MDPEnv(n_states=self.mean_field_env.n_states, n_actions=self.mean_field_env.n_actions,
                         Tf=self.mean_field_env.Tf, gamma=self.mean_field_env.gamma)

        mdp_env.set_reward_vec(reward_vec)
        mdp_env.set_transitions(self.mean_field_env.T)

        return mdp_env
