import matplotlib.pyplot as plt
import numpy as np

FILENAME = "lqr_noise_rejection.csv"
data = np.genfromtxt(FILENAME, delimiter=',')
plt.plot(data[:,0], data[:,2])

FILENAME = "ddpg_noise_rejection.csv"
data = np.genfromtxt(FILENAME, delimiter=',')
plt.plot(data[:,0], data[:,1])

FILENAME = "sac_noise_rejection.csv"
data = np.genfromtxt(FILENAME, delimiter=',')
plt.plot(data[:,0], data[:,1])

plt.legend(["LQR", "DDPG", "SAC"])
plt.xlabel("Time (s)")
plt.ylabel("Body Angle (rad)")

plt.savefig("noise_rejection.png")