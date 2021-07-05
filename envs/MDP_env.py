import numpy as np

class MDPEnv:
    def __init__(self,n_states, n_actions, Tf, gamma=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.T = None
        self.Tf = Tf
        self.gamma = gamma
        self.reward_vec = None

    def set_transitions(self, transitions):
        self.T = transitions

    def reward(self, s, a, t):
        return self.reward_vec[t][s][a]

    def set_reward_vec(self, reward_vec):
        self.reward_vec = reward_vec
