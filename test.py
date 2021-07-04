import numpy as np
from envs.rotate_env import RotateConfEnv
from solvers.MDP_solver import MDP_Solver
from solvers.MF_solver import MFSolver
from visualizers.visualize_mf import visualize_mf

mu0 = np.array([0.1, 0.4, 0.2, 0.1, 0.2])
env = RotateConfEnv(mu0=mu0)

# MDP Solver test
# nu = [np.array([0.1, 0.3, 0, 0.2, 0, 0.1, 0.3]), np.array([0.3, 0.1, 0.1, 0.15, 0.25, 0, 0.1])]
# mdp_solver = MDP_Solver(envs=envs)
# mdp_solver.solve(nu=nu)

# MF Solver test
mf_solver = MFSolver(env=env)
mf_solver.solve()
print(mf_solver.soln["mu"][-1])

visualize_mf(mf_solver.soln["mu"])

print(mf_solver.soln["policy"])

print(mf_solver.soln["mu"])
