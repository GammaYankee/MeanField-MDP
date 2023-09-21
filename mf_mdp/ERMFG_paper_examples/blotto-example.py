import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mf_mdp.solvers.MF_solver import MF_Solver
from mf_mdp.visualizers.visualize_mf import visualize_mf
from mf_mdp.ERMFG_paper_examples.blotto_env import BlottoEnv
from utils import ROOT_PATH
import pickle

# Set up blotto env
mu0 = np.array([0.3, 0.2, 0.3, 0.1, 0.1])
mean_field_env = BlottoEnv(mu_0=mu0)
mean_field_env.set_beta(2)

# Set up MF Solver
mf_solver = MF_Solver(env=mean_field_env)
mf_solver.solve(entropy_regularized=True, beta=mean_field_env.beta, prior=mean_field_env.prior)

# visualize mean field
visualize_mf(mf_solver.soln["mu"])

# save results
data = mf_solver.soln
pickle.dump(data, open(ROOT_PATH / "test_data/mf_soln_beta_{}.pkl".format(mean_field_env.beta), "wb"))
