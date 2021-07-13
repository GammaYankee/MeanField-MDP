import matplotlib.pyplot as plt
import pickle
import math


def plot_log_log_error(N_list, error_list):
    slope = (math.log(error_list[1]) - math.log(error_list[-1])) / (math.log(N_list[1]) - math.log(N_list[-1]))
    # plot
    w = 8
    h = 6
    d = 80

    # reference line
    start = error_list[0]
    reference_line = [start * N_list[k]**(-1/2) for k in range(len(N_list))]
    plt.figure(figsize=(w, h), dpi=d)
    plt.loglog(N_list, error_list, '-ko', label='data')
    plt.loglog(N_list, reference_line, '--b', label="reference line")
    plt.title("Performance Error with Finite Population", fontsize=18)
    plt.xlabel("Number of agents", fontsize=15)
    plt.ylabel("Performance Gain by Deviating", fontsize=15)
    plt.text(2, 2.5, 'slope={:.3f}'.format(slope), fontsize=15)
    plt.legend()
    plt.show()
    print("slope is {}".format(slope))


if __name__ == "__main__":
    data = pickle.load(open("./test_data/data_bak.pkl", "rb"))

    N, error = data["N_agent_test_list"], data["error"]

    plot_log_log_error(N, error)
