import numpy as np
from envs.mean_field_env import MeanFieldEnv
from envs.error_test_env import TestEnv


class ErrorVerifier:
    def __init__(self, mean_field_env: MeanFieldEnv, mean_field_policy, n_ittr=50, eps=0.0001):
        self.mean_field_env = mean_field_env
        self.n_ittr = n_ittr
        self.eps = eps
        self.MDP_env = self.__init_MDP_env()



    def __init_MDP_env(self):


