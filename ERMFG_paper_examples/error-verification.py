import pickle
import numpy as np
from envs.error_test_MDP import FinitePopMDPEnv
from solvers.MDP_solver import MDP_Solver
from visualizers.plot_error import plot_log_log_error

BETA = 1

# load solution results
mf_soln = pickle.load(open("./data/mf_soln_beta_{}.pkl".format(BETA), "rb"))

# compute original value
mean_field_env = mf_soln["mfg_env"]
mean_field_policy_flow = mf_soln["policy"]
mean_field_value_flow = mf_soln["value"]
mean_field_state_flow = mf_soln["mu"]
induced_mdp = mf_soln["MDP_induced"]
mu0 = mean_field_env.mu0

mean_field_value = np.matmul(mean_field_value_flow[0], mu0.T)

# test evaluate_entropy
evaluator = MDP_Solver(env=induced_mdp)
mfg_induced_value_flow = evaluator.evaluate_entropy(policy=mean_field_policy_flow, prior=mean_field_env.prior,
                                                    beta=mean_field_env.beta)
mean_field_value_2 = np.matmul(mfg_induced_value_flow[0], mu0.T)

# test cases for N agents
N_agent_test_list = [2, 5, 10, 20, 50, 75, 100]
performance_improve = []
errors = []
reward_errors = []

# conduct error verification
for N in N_agent_test_list:
    print("Testing finite population with N={}".format(N))

    # Generate finite population MDP
    finite_population_MDP = FinitePopMDPEnv(n_agents=N, mean_field_env=mean_field_env,
                                            mean_field_policy=mean_field_policy_flow)

    # Let agent i play mean field policy in the N-agent environment
    performance_evaluator = MDP_Solver(env=finite_population_MDP)
    induced_value_flow = performance_evaluator.evaluate_entropy(policy=mean_field_policy_flow,
                                                                prior=mean_field_env.prior, beta=mean_field_env.beta)
    original_performance = np.matmul(induced_value_flow[0], mu0.T)

    # Let agent i optimize with N-1 agents fixed to mean field policy
    error_verification_mdp_solver = MDP_Solver(env=finite_population_MDP)
    reward_error = abs(induced_mdp.reward_vec[-1][0][0] - finite_population_MDP.reward_vec[-1][0][0])
    deviated_policy, value_flow = error_verification_mdp_solver.solve_entropy(prior=mean_field_env.prior,
                                                                              beta=mean_field_env.beta)
    deviated_value = np.matmul(value_flow[0], mu0.T)

    performance_improve.append(deviated_value - original_performance)
    errors.append(abs(deviated_value - mean_field_value))
    reward_errors.append(reward_error)

    np.set_printoptions(precision=3)
    print("Performance improvement is {}".format(performance_improve[-1]))

plot_log_log_error(N_agent_test_list, performance_improve)

# save test data
data = {"N_agent_test_list": N_agent_test_list, "error": errors, "reward_error": reward_errors,
        "performance_improve": performance_improve}
pickle.dump(data, open("./data/error_beta_{}.pkl".format(mean_field_env.beta), "wb"))
