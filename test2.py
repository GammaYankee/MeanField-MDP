from envs.blotto_env_v2 import BlottoEnv2
import numpy as np
from solvers.MF_solver import MF_Solver
from solvers.MDP_solver import MDP_Solver

mu0 = np.array([0.3, 0.2, 0.3, 0.1, 0.1])
mean_field_env = BlottoEnv2(mu_0=mu0)
mf_solver = MF_Solver(env=mean_field_env)
mf_solver.solve()
mf_solver = MF_Solver(env=mean_field_env)
mf_solver.solve(entropy_regularized=True, beta=mean_field_env.beta, prior=mean_field_env.prior)

induced_mdp = mf_solver.soln["MDP_induced"]

mdp_solver = MDP_Solver(env=induced_mdp)
policy, value_flow_solved = mdp_solver.solve_entropy(prior=mean_field_env.prior, beta=mean_field_env.beta)

mdp_evaluator = MDP_Solver(env=induced_mdp)
value_flow_evaluated = mdp_evaluator.evaluate_entropy(policy=policy, prior=mean_field_env.prior,
                                                      beta=mean_field_env.beta)


print("done!")
