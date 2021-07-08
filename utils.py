import math
from copy import deepcopy
import numpy as np


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
        print(len(queue))
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
    N = 100
    p = [0.3, 0.2, 0.1, 0.4]
    dist = empirical_dist(N, p)

    prob = [dist[k].prob for k in range(len(dist))]

    print(sum(prob))

    print("done!")

    test_emp_dist(dist, N, p)
