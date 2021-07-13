import matplotlib.pyplot as plt
import pickle
import math

data = pickle.load(open("./test_data/data.pkl", "rb"))

N_agent_test_list, error = data["N_agent_test_list"], data["error"]

# plot
w = 8
h = 6
d = 80
plt.figure(figsize=(w, h), dpi=d)
plt.loglog(N_agent_test_list, error, '-ko')
plt.title("Performance Error with Finite Population", fontsize=18)
plt.xlabel("Number of agents", fontsize=15)
plt.ylabel("Performance Gain by Deviating", fontsize=15)
plt.show()

slope = (math.log(error[1]) - math.log(error[2])) / (math.log(N_agent_test_list[1]) - math.log(N_agent_test_list[2]))

print(slope)