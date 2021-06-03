import matplotlib.pyplot as plt
import numpy as np

FILENAME_BASE = "velocity"
FILENAME_EXT = ["_ddpg.csv", "_sac.csv"]
LEGEND = ["DDPG", "SAC"]
COLORS = ["C1", "C2"]

for i in range(2):
    data = np.genfromtxt(FILENAME_BASE + FILENAME_EXT[i], delimiter=',')
    plt.plot(data[:,0], data[:,1], color=COLORS[i])

plt.legend(LEGEND)
plt.xlabel("Training Step")
plt.ylabel("Trial Reward")

plt.savefig("train_" + FILENAME_BASE + ".png")
plt.show()
