import matplotlib.pyplot as plt
import numpy as np

FILENAMES = ["lqr_velocity_control.csv", "ddpg_velocity_control.csv", "sac_velocity_control.csv"]
LEGEND = ["LQR", "DDPG", "SAC", "Target"]

for fname in FILENAMES:
    data = np.genfromtxt(fname, delimiter=',')
    plt.plot(data[:,0], data[:,1])


target = np.zeros((500,))
for i in range(1,500):
    if i < 200:
        target[i] = 0.32
    elif i < 250:
        target[i] = 0
    elif i < 450:
        target[i] = -0.32
    else:
        target[i] = 0
plt.plot(data[:,0], target, "r--")


plt.legend(LEGEND)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.savefig("velocity_control_velocity.png")

plt.cla()

for fname in FILENAMES:
    data = np.genfromtxt(fname, delimiter=',')
    plt.plot(data[:,0], data[:,2])

plt.legend(LEGEND)
plt.xlabel("Time (s)")
plt.ylabel("Body Angle (rad)")

plt.savefig("velocity_control_angle.png")