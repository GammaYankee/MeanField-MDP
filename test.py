import numpy as np
from env.rotate_env import RotateConfEnv
from solvers.MDP_solver import MDP_Solver

mu0 = [0.1, 0.3, 0.2, 0.1, 0.3]
env = RotateConfEnv(mu0=mu0)

nu = [np.array([0.1, 0.3, 0, 0.1,0, 0.1, 0.3]), np.array([0.3, 0.3, 0, 0.3,0, 0, 0.1])]

mdp_solver = MDP_Solver(env=env)

mdp_solver.solve(nu=nu)