import numpy as np
from typing import List

from mf_mdp.envs.mean_field_env import MeanFieldEnv


class TrafficEnv(MeanFieldEnv):
    def __init__(self, n_lanes: int, n_blocks: int, cells_per_block: int,
                 n_actions: int, action_vec_list: List,
                 obstacle_list: List,
                 Tf: int, gamma=1.0):
        self.n_lanes, self.n_blocks, self.cells_per_block = n_lanes, n_blocks, cells_per_block
        self.action_vector_list = action_vec_list

        super(TrafficEnv, self).__init__(n_states=self.n_lanes * self.n_blocks,
                                         n_actions=[n_actions for _ in range(n_lanes * n_blocks)], Tf=Tf, gamma=gamma)

        self.obstacle_state_list = [self.status2state(lane=o[0], block=o[1]) for o in obstacle_list]

    def _init_transitions(self):
        pass

    def get_transition(self, s, a, mu=None):
        result = np.zeros(self.n_states)
        if self.isTerminal(s):
            result[s] = 1
            return result

        s_list, p_list = self.simulate_action(s, a, mu)
        for s, p in zip(s_list, p_list):
            result[s] = p

        return result

    def individual_reward(self, s_t, a_t, nu_t, t):
        r = 0
        if self.action_vector_list[a_t][0] != 0:
            r -= 1
        if self.isTerminal(s_t):
            r += 5
        else:
            r -= 1
        return r

    def simulate_action(self, s, a, mu):
        action_vector = self.action_vector_list[a]
        lane, block = self.state2status(s)
        lane_new = self.proj(lane + action_vector[0], 0, self.n_lanes - 1)
        block_new = self.proj(block + action_vector[1], 0, self.n_blocks - 1)
        new_state = self.status2state(lane_new, block_new)

        if new_state == s or new_state in self.obstacle_state_list:
            s_list = [s]
            p_list = [1]
        elif self.isTerminal(new_state):
            s_list = [new_state]
            p_list = [1]
        else:
            s_list = [s, new_state]
            p_list = [mu[new_state], 1 - mu[new_state]]

        return s_list, p_list

    def state2status(self, s):
        block = int(s % self.n_blocks)
        lane = int((s - block) / self.n_blocks)
        return lane, block

    def isTerminal(self, s):
        _, block = self.state2status(s)

        if block == self.n_blocks - 1:
            return True
        else:
            return False

    def status2state(self, lane, block):
        state = lane * self.n_blocks + block
        return state

    def proj(self, value, min_v, max_v):
        if value < min_v:
            return min_v
        elif value > max_v:
            return max_v
        else:
            return value
