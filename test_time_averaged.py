import numpy as np
from envs.blotto_env_v2 import BlottoEnv2
from solvers.MF_solver import MF_Solver
from visualizers.visualize_mf import visualize_mf

mu0 = np.array([0.1, 0.2, 0.5, 0.0, 0.2])
mean_field_env = BlottoEnv2(mu_0=mu0)

# time averaged MF Solver test
mf_solver = MF_Solver(env=mean_field_env, n_ittr=5000)
mf_solver.solve(time_averaged=True, eta=0.1, entropy_regularized=False)
print(mf_solver.soln["mu"][-1])

visualize_mf(mf_solver.soln["mu"])
