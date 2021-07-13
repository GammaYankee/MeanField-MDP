import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from envs.mean_field_env import MeanFieldEnv


def empirical_dist(N, p):
    '''
    Compute the distribution of empirical distribution
    :param N: Number of i.i.d. random variables
    :param p: Probability mass function
    :return:
    '''

    class Node:
        def __init__(self, n_list, prob):
            self.n_list = n_list
            self.prob = prob

        def __len__(self):
            return len(self.n_list)

    def expand(queue, N):
        if len(queue[0]) == len(p):
            return queue
        for _ in range(len(queue)):
            node = queue.pop(0)
            state = len(node)
            N_left = N - sum(node.n_list)
            if state == len(p) - 1:
                n_k = N_left
                n_list = deepcopy(node.n_list)
                n_list.append(n_k)
                prob = node.prob * math.comb(N_left, n_k) * p[state] ** n_k
                node_ = Node(n_list, prob)
                queue.append(node_)
            else:
                for n_k in range(N_left + 1):
                    n_list = deepcopy(node.n_list)
                    n_list.append(n_k)
                    prob = node.prob * math.comb(N_left, n_k) * p[state] ** n_k
                    node_ = Node(n_list, prob)
                    queue.append(node_)
        # print(len(queue))
        queue = expand(queue, N)
        return queue

    queue = [Node([], 1)]

    return expand(queue, N)


def test_emp_dist(emp_dist, N, p):
    n_list = np.array([0, 0, 0, 0]).astype(float)
    for case in emp_dist:
        n_list += np.array(case.n_list) * case.prob
    print((n_list / N, p))




if __name__ == "__main__":
    from plot_error import plot_log_log_error
    N = [1, 5, 10, 20, 50]
    p = [0.3, 0.2, 0.1, 0.4]
    errors = []
    for n in N:
        error = 0
        dist = empirical_dist(n, p)
        probs = [dist[k].prob for k in range(len(dist))]
        cases = [dist[k].n_list for k in range(len(dist))]

        for prob, case in zip(probs, cases):
            emp_dist = [case[k]/n for k in range(len(case))]
            error += prob * sum([abs(emp_dist[k] - p[k]) for k in range(len(p))])

        errors.append(error)
        print("done with N={}".format(n))

    plot_log_log_error(N, errors)








