import numpy as np
from mf_mdp.envs.MDP_env import MDPEnv
from copy import deepcopy
from math import exp, log


class MDP_Solver:
    def __init__(self, env: MDPEnv, n_ittr=20, eps=0.01):
        self.env = env
        self.n_ittr = n_ittr
        self.eps = eps
        self.Tf = self.env.Tf
        self.V = [np.zeros(self.env.n_states) for _ in range(self.Tf + 1)]  # Value Function
        self.V_ = [np.zeros(self.env.n_states) for _ in range(self.Tf + 1)]  # Temp storage
        self.policy = [[np.zeros(self.env.n_actions[s]) for s in range(self.env.n_states)] for _ in range(self.Tf + 1)]

    def reset_V_storage(self):
        self.V = [np.zeros(self.env.n_states) for _ in range(self.Tf + 1)]  # Value Function
        self.V_ = [np.zeros(self.env.n_states) for _ in range(self.Tf + 1)]  # Temp storage

    def solve(self):
        self._resetTemp()
        for t in reversed(range(self.env.Tf + 1)):
            for s in range(self.env.n_states):
                if t == self.env.Tf:  # terminal time step
                    Q_s = np.zeros(self.env.n_actions[s])
                    for a in range(self.env.n_actions[s]):
                        Q_s[a] = self.env.reward(s=s, a=a, t=t)
                else:
                    Q_s = self.compute_Q_s(s, t)

                self.V_[t][s] = max(Q_s)
                a_max = np.argmax(Q_s)
                self.policy[t][s][a_max] = 1
        self._updateNew()

        return self.policy, self.V

    def solve_entropy(self, prior=None, beta=None):
        self._resetTemp()
        for t in reversed(range(self.env.Tf + 1)):
            for s in range(self.env.n_states):
                if prior is not None:
                    prior_s = prior[t][s]
                else:
                    prior_s = [1 / self.env.n_actions[s] for _ in range(self.env.n_actions[s])]
                if t == self.env.Tf:  # terminal time step
                    Q_s = np.zeros(self.env.n_actions[s])
                    for a in range(self.env.n_actions[s]):
                        Q_s[a] = self.env.reward(s=s, a=a, t=t)
                else:
                    Q_s = self.compute_Q_s(s, t)

                # compute entropy regularized value function
                self.V_[t][s] = self._compute_V_KL(Q_s, prior_s, beta)

                # compute soft-optimal policies
                self.policy[t][s] = self._compute_soft_policy(Q_s, prior_s, beta)
        self._updateNew()

        return self.policy, self.V

    def evaluate(self, policy):
        self.reset_V_storage()
        for t in reversed(range(self.env.Tf + 1)):
            for s in range(self.env.n_states):
                if t == self.env.Tf:
                    Q_s = np.zeros(self.env.n_actions[s])
                    for a in range(self.env.n_actions[s]):
                        Q_s[a] = self.env.reward(s=s, a=a, t=t)
                else:
                    Q_s = self.compute_Q_s(s, t)
                self.V_[t][s] = sum(Q_s[a] * policy[t][s][a] for a in range(self.env.n_actions[s]))
        return self.V_

    def evaluate_entropy(self, policy, prior=None, beta=None):
        self.reset_V_storage()
        for t in reversed(range(self.env.Tf + 1)):
            for s in range(self.env.n_states):
                if t == self.env.Tf:
                    Q_s = np.zeros(self.env.n_actions[s])
                    for a in range(self.env.n_actions[s]):
                        Q_s[a] = self.env.reward(s=s, a=a, t=t)
                else:
                    Q_s = self.compute_Q_s(s, t)
                self.V_[t][s] = sum(Q_s[a] * policy[t][s][a] for a in range(self.env.n_actions[s])) \
                                - 1/beta * self._KL_divergence(policy[t][s], prior[t][s])
        return self.V_

    def compute_Q_s(self, s, t):
        n_actions = self.env.n_actions[s]
        Q_s = np.zeros(n_actions)
        V_next = self.V_[t + 1]
        for a in range(n_actions):
            Q_s[a] += self.env.reward(s=s, a=a, t=t)
            T_sa = self.env.get_transition_sa(s, a, t)
            assert len(T_sa) > 0
            for s_prime in range(self.env.n_states):
                Q_s[a] += self.env.gamma * T_sa[s_prime] * V_next[s_prime]
        return Q_s

    def _compute_V_KL(self, Q_s, prior_s, beta):
        temp = 0
        for a in range(len(Q_s)):
            temp += prior_s[a] * exp(beta * Q_s[a])
        V_KL = 1 / beta * log(temp)
        return V_KL

    def _KL_divergence(self, dist1, dist2):
        KL = 0
        for a in range(len(dist1)):
            KL += dist1[a] * np.log(dist1[a] / dist2[a])
        return KL

    def _compute_soft_policy(self, Q_s, prior_s, beta):
        policy_KL = []
        for a in range(len(Q_s)):
            prob_a = prior_s[a] * exp(beta * Q_s[a])
            policy_KL.append(prob_a)
        Z = sum(policy_KL)
        for a in range(len(Q_s)):
            policy_KL[a] /= Z
        assert (abs(sum(policy_KL) - 1) < 1e-5)
        return policy_KL

    def _resetTemp(self):
        self.V_ = [np.zeros(self.env.n_states) for _ in range(self.Tf + 1)]
        self.policy = [[np.zeros(self.env.n_actions[s]) for s in range(self.env.n_states)] for _ in range(self.Tf + 1)]

    def _updateNew(self):
        self.V = deepcopy(self.V_)
