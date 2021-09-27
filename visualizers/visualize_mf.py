import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_mf(mu):
    # info of the solution
    T_f = len(mu) - 1
    n_states = len(mu[0])

    # create figure
    fig = plt.figure(dpi=120)
    ax = Axes3D(fig)
    column_names = [str(s+1) for s in range(n_states)]  # naming of states
    column_names.insert(0, "")
    column_names.append("")
    xpos = np.arange(0, n_states, 1)  # generate the mesh
    ypos = np.arange(0, T_f + 1, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos - 0.75)
    xpos, ypos = xpos.flatten(), ypos.flatten()
    zpos = np.zeros(n_states * (T_f + 1))

    # set bar dimensions
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = np.array(mu).flatten()

    # plot pars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.7)
    ax.set_xticks(np.arange(0, n_states+2, 1)-0.5)

    ax.xaxis.set_ticklabels(column_names)  # set axis name
    ax.set_xlabel('State', fontsize=15)
    ax.set_ylabel('Time', fontsize=15, labelpad=10)
    ax.set_zlabel('Population', fontsize=15)
    ax.set_zlim3d(bottom=0, top=1.0)

    # show plot
    plt.show()

