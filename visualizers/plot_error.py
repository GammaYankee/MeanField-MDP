import matplotlib.pyplot as plt
import pickle
import math


def plot_log_log_error(N_list, error_list):
    # error_list[0] += 0.005
    # error_list[2] += 0.007
    # error_list[4] -= 0.0005

    slope = (math.log(error_list[0]) - math.log(error_list[-1])) / (math.log(N_list[0]) - math.log(N_list[-1]))
    # plot
    w = 8
    h = 6
    d = 120

    # reference line
    start = error_list[0]
    reference_line = [start * N_list[0] ** (1 / 2) * N_list[k] ** (-1 / 2) for k in range(len(N_list))]
    fig = plt.figure(figsize=(w, h), dpi=d)
    plt.loglog(N_list, error_list, '-ko', label='data', linewidth=2.5)
    plt.loglog(N_list, reference_line, '--b', label="reference line", linewidth=2.5)
    # plt.title("Performance Gain with Finite Population", fontsize=18)
    plt.xlabel("Number of agents", fontsize=18)
    plt.ylabel("Performance Gain by Deviating", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.text(2, 0.03, 'slope={:.3f}'.format(slope), fontsize=15)
    plt.text(35, 0.025, 'slope={}'.format(-0.5), fontsize=18, color='b')
    plt.legend(fontsize=15, loc="lower left")
    plt.show()
    print("slope is {}".format(slope))

    file_name = 'deviate.pdf'
    fig.savefig(file_name, dpi=120)


if __name__ == "__main__":
    data = pickle.load(open("../ERMFG_paper_examples/data/error_beta_1.pkl", "rb"))

    N, error = data["N_agent_test_list"], data["error"]

    plot_log_log_error(N, error)


