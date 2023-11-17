import numpy as np

from mf_mdp.solvers.MF_solver import MF_Solver
from mf_mdp.envs.traffic import TrafficEnv
from mf_mdp.visualizers.renderers import TrafficRenderer
from mf_mdp.visualizers.simulator import BaseAgent, MFSimulator

if __name__ == "__main__":
    from utils import ROOT_PATH

    file_path = ROOT_PATH / "data/mf_soln/traffic_soln.pkl"

    action_vector_list = [np.array([-1, 0]), np.array([-1, 1]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]
    obstacle_list = [[0, 5], [2, 10]]

    mf_env = TrafficEnv(n_lanes=3, n_blocks=20, cells_per_block=1, n_actions=5, action_vec_list=action_vector_list,
                        obstacle_list=obstacle_list, Tf=40)
    mu0 = np.zeros(mf_env.n_states)
    n = 0
    for l in range(2):
        for b in range(2):
            s = mf_env.status2state(lane=l, block=b)
            mu0[s] = 1
            n += 1
    mu0 /= n

    mf_env.set_init_mu(mu0)
    mf_env.set_beta(1)

    mf_solver = MF_Solver(env=mf_env)
    # mf_solver.solve(entropy_regularized=True, beta=mf_env.beta, prior=None)
    # mf_solver.save(file_path)

    mf_solver.load(file_path)

    agent_list = [
        BaseAgent(index=0, policy=mf_solver.soln["policy"], s0=mf_env.status2state(lane=0, block=0), color='b'),
        BaseAgent(index=1, policy=mf_solver.soln["policy"], s0=mf_env.status2state(lane=0, block=1), color='r'),
        BaseAgent(index=2, policy=mf_solver.soln["policy"], s0=mf_env.status2state(lane=1, block=0), color='y'),
        BaseAgent(index=3, policy=mf_solver.soln["policy"], s0=mf_env.status2state(lane=1, block=1), color='g')
    ]

    renderer = TrafficRenderer(env=mf_env, save_gif=True, save_dir=ROOT_PATH/"figures/gif")

    simulator = MFSimulator(env=mf_env, agent_list=agent_list, renderer=renderer)

    for t in range(30):
        simulator.step()

    simulator.renderer.render_gif()
