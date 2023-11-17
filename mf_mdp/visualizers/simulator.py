import copy

import numpy as np
from mf_mdp.envs.traffic import TrafficEnv
from mf_mdp.envs.mean_field_env import MeanFieldEnv
from mf_mdp.visualizers.renderers import TrafficRenderer
from typing import List


class BaseAgent:
    def __init__(self, index, policy, s0, color):
        self.index = index
        self.policy = policy
        self.state = s0
        self.color = color

    def act(self, t):
        policy_s = self.policy[t][self.state]
        action = np.random.choice(a=len(policy_s), p=policy_s)
        return action

    def step(self, new_state):
        self.state = copy.copy(new_state)


class MFSimulator:
    def __init__(self, env: MeanFieldEnv, agent_list: List[BaseAgent], renderer: TrafficRenderer, visualize=True):
        self.env = env
        self.renderer = renderer
        self.agent_list, self.n_agents = agent_list, len(agent_list)
        self.visualize = visualize
        self.t = 0

        self.update_mu()

        if self.visualize:
            self.renderer.create_figure()
            self.renderer.render_agents(agent_list=self.agent_list)
            self.renderer.render_obstacles()
            self.renderer.show()
            self.renderer.hold()
            self.renderer.clear()

    def step(self):
        for agent in self.agent_list:
            s = agent.state
            a = agent.act(self.t)
            p_s = self.env.get_transition(s=s, a=a, mu=self.mu_t)
            s_ = np.random.choice(a=self.env.n_states, p=p_s)
            agent.step(new_state=s_)
        self.update_mu()
        self.t += 1

        if self.visualize:
            self.renderer.render_agents(agent_list=self.agent_list)
            self.renderer.render_obstacles()
            self.renderer.show()
            self.renderer.hold()
            self.renderer.clear()

    def update_mu(self):
        self.mu_t = np.zeros(self.env.n_states)
        for agent in self.agent_list:
            self.mu_t[agent.state] = 1
        self.mu_t /= self.n_agents

