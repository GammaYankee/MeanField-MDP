import numpy as np
from envs.rotate_env import RotateConfEnv
from envs.error_test_env import TestEnv
from solvers.MDP_solver import MDP_Solver
from solvers.MF_solver import MFSolver
from visualizers.visualize_mf import visualize_mf
from envs.MDP_env import MDPEnv

# mu0 = np.array([0.3, 0.4, 0.2, 0.1])
# mean_filed_env = TestEnv(mu0=mu0)

mu0 = np.array([0.1, 0.4, 0.2, 0.1, 0.2])
mean_filed_env = RotateConfEnv(mu0=mu0)

# MDP Solver test
# nu = [np.array([0.1, 0.3, 0, 0.2, 0, 0.1, 0.3]), np.array([0.3, 0.1, 0.1, 0.15, 0.25, 0, 0.1])]
# mdp_solver = MDP_Solver(envs=envs)
# mdp_solver.solve(nu=nu)

# MF Solver test
mf_solver = MFSolver(env=mean_filed_env)
mf_solver.solve()
print(mf_solver.soln["mu"][-1])

visualize_mf(mf_solver.soln["mu"])

# # Error test
# finite_population_MDP = MDPEnv(n_states=mean_filed_env.n_states, n_actions=mean_filed_env.n_actions,
#                                Tf=mean_filed_env.Tf, gamma=mean_filed_env.gamma)
#
# finite_population_MDP.set_transitions(mean_filed_env.T)
