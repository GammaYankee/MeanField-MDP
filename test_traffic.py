import numpy as np

from mf_mdp.solvers.MF_solver import MF_Solver
from mf_mdp.envs.traffic import TrafficEnv
from mf_mdp.visualizers.renderers import TrafficRenderer

if __name__ == "__main__":
    from utils import ROOT_PATH

    file_path = ROOT_PATH / "data/mf_soln/traffic_soln.pkl"

    action_vector_list = [np.array([-1, 0]), np.array([-1, 1]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]
    obstacle_list = [[0, 5], [2, 10]]
    # obstacle_list = []


    mf_env = TrafficEnv(n_lanes=3, n_blocks=30, cells_per_block=1, n_actions=5, action_vec_list=action_vector_list,
                        obstacle_list=obstacle_list, Tf=40)
    mu0 = np.zeros(mf_env.n_states)
    n = 0
    for l in range(1):
        for b in range(4):
            s = mf_env.status2state(lane=l, block=b)
            mu0[s] = 1
            n += 1
    mu0 /= n

    mf_env.set_init_mu(mu0)
    mf_env.set_beta(2)

    mf_solver = MF_Solver(env=mf_env)
    mf_solver.solve(entropy_regularized=True, beta=mf_env.beta, prior=None)
    mf_solver.save(file_path)

    mf_solver.load(file_path)

    renderer = TrafficRenderer(env=mf_env, cmap_type="jet", save_dir=ROOT_PATH / "figures/gif", save_gif=True)

    renderer.create_figure()
    for t in range(40):
        renderer.render_lanes()
        renderer.render_mf(mf=mf_solver.soln["mu"][t])
        renderer.render_obstacles()
        renderer.show()
        renderer.hold(t=0.2)
        renderer.clear()

    renderer.render_gif(duration=0.5)
