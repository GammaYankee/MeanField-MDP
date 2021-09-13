import numpy as np

# Graph information
mu_0 = np.array([6, 0, 0, 0, 0])

# Connectivity
connectivity = np.array([[1, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 1]])

T = np.zeros(connectivity.shape)
for state in range(connectivity.shape[0]):
    T_s = connectivity[state, :]
    n_actions = sum(T_s)
    T_s_ = T_s / n_actions
    T[state, :] = T_s_

# connectivity = np.array([[0.5,  0.2,    0,      0,      0.3],
#                          [0,    0.8,    0.1,    0.1,    0],
#                          [0,    0,      0.6,    0.4,    0],
#                          [0,    0,      0.3,    0.5,    0.2],
#                          [0.5,  0,      0,      0,      0.3]])

mu = mu_0
for t in range(100):
    mu = np.matmul(mu, T)

print(mu)
