import numpy as np
from envs.blotto_env_v2 import BlottoEnv2
from solvers.MF_solver import MF_Solver
from solvers.MDP_solver import MDP_Solver
from visualizers.visualize_mf import visualize_mf
from envs.error_test_MDP import FinitePopMDPEnv
import pickle

# mu0 = np.array([0.3, 0.4, 0.2, 0.1])
# mean_filed_env = TestEnv(mu0=mu0)

# mu0 = np.array([0.1, 0.4, 0.2, 0.1, 0.2])
# mean_filed_env = RotateConfEnv(mu0=mu0)

# MDP Solver test
# nu = [np.array([0.1, 0.3, 0, 0.2, 0, 0.1, 0.3]), np.array([0.3, 0.1, 0.1, 0.15, 0.25, 0, 0.1])]
# mdp_solver = MDP_Solver(envs=envs)
# mdp_solver.solve(nu=nu)

# Blotto Game Env V2
mu0 = np.array([0.3, 0.2, 0.3, 0.1, 0.1])
# mu0 = np.array([0.2, 0.3, 0.1, 0.1, 0.3])
mean_field_env = BlottoEnv2(mu_0=mu0)

# MF Solver test
mf_solver = MF_Solver(env=mean_field_env)
mf_solver.solve(entropy_regularized=True, beta=mean_field_env.beta, prior=mean_field_env.prior)
print(mf_solver.soln["mu"][-1])

visualize_mf(mf_solver.soln["mu"])

# Error test
mean_field_policy_flow = mf_solver.soln["policy"]
mean_field_value_flow = mf_solver.soln["value"]
mean_field_state_flow = mf_solver.soln["mu"]
induced_mdp = mf_solver.soln["MDP_induced"]

# test entropy-regularized evaluator
new_mdp_solver = MDP_Solver(induced_mdp)
verified_value_flow = new_mdp_solver.evaluate_entropy(policy=mean_field_policy_flow, prior=mean_field_env.prior,
                                                      beta=mean_field_env.beta)
verified_value_flow_2 = new_mdp_solver.solve_entropy(prior=mean_field_env.prior,
                                                     beta=mean_field_env.beta)

mean_field_value = np.matmul(mean_field_value_flow[0], mu0.T)

N_agent_test_list = [2, 5, 10, 20]  # , 50, 75, 100]
performance_improve = []
errors = []
reward_errors = []
for N in N_agent_test_list:
    print("Testing finite population with N={}".format(N))

    # Evaluate performance of mean field policy with N agents
    finite_population_MDP = FinitePopMDPEnv(n_agents=N, mean_field_env=mean_field_env,
                                            mean_field_policy=mean_field_policy_flow)
    performance_evaluator = MDP_Solver(env=finite_population_MDP)
    induced_value_flow = performance_evaluator.evaluate(mean_field_policy_flow)
    original_performance = np.matmul(induced_value_flow[0], mu0.T)

    # Let agent i optimize with N-1 agents fixed to mean field policy
    error_verification_mdp_solver = MDP_Solver(env=finite_population_MDP)
    reward_error = abs(induced_mdp.reward_vec[-1][0][0] - finite_population_MDP.reward_vec[-1][0][0])
    deviated_policy, value_flow = error_verification_mdp_solver.solve()

    deviated_value = np.matmul(value_flow[0], mu0.T)

    performance_improve.append(deviated_value - original_performance)
    errors.append(abs(deviated_value - mean_field_value))
    reward_errors.append(reward_error)

# plot_log_log_error(N_agent_test_list, performance_improve)

# save test data
data = {"N_agent_test_list": N_agent_test_list, "error": errors, "reward_error": reward_errors}
pickle.dump(data, open("./test_data/data_test.pkl", "wb"))
