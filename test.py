import numpy as np
from envs.rotate_env import RotateConfEnv
from envs.error_test_env import TestEnv
from solvers.MDP_solver import MDP_Solver
from solvers.MF_solver import MFSolver
from solvers.MDP_solver import MDP_Solver
from visualizers.visualize_mf import visualize_mf
from envs.error_test_MDP import FinitePopMDPEnv

mu0 = np.array([0.3, 0.4, 0.2, 0.1])
mean_filed_env = TestEnv(mu0=mu0)

# mu0 = np.array([0.1, 0.4, 0.2, 0.1, 0.2])
# mean_filed_env = RotateConfEnv(mu0=mu0)

# MDP Solver test
# nu = [np.array([0.1, 0.3, 0, 0.2, 0, 0.1, 0.3]), np.array([0.3, 0.1, 0.1, 0.15, 0.25, 0, 0.1])]
# mdp_solver = MDP_Solver(envs=envs)
# mdp_solver.solve(nu=nu)

# MF Solver test
mf_solver = MFSolver(env=mean_filed_env)
mf_solver.solve()
print(mf_solver.soln["mu"][-1])

visualize_mf(mf_solver.soln["mu"])

# Error test
mean_field_policy_flow = mf_solver.soln["policy"]
mean_field_value_flow = mf_solver.soln["value"]
mean_field_state_flow = mf_solver.soln["mu"]
induced_mdp = mf_solver.soln["MDP_induced"]

mean_field_value = np.matmul(mean_field_value_flow[0], mu0.T)

finite_population_MDP = FinitePopMDPEnv(n_agents=1, mean_field_env=mean_filed_env,
                                        mean_field_policy=mean_field_policy_flow)

error_verification_mdp_solver = MDP_Solver(env=finite_population_MDP)
_, value_flow = error_verification_mdp_solver.solve()
deviated_value = np.matmul(value_flow[0], mu0.T)

print(deviated_value - mean_field_value)
