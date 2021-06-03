import numpy as np
from scipy.linalg import solve_continuous_are
from wheeled_inverted_pendulum_env import WIPEnv
import time

x0 = [0.0, 0.0, 0.0, 0.0]

Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
R = 5000*np.eye(1)

T = 500

env = WIPEnv()
goal_state = np.array([0.0, 0.0, 0.0, 0.0])
f, A, B = env.linearized_dynamics(goal_state, 0.0)

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P

state = env.reset(state=x0)
env.render()
time.sleep(1)

data = np.zeros((T, 3))
dt = env.dt

start = time.monotonic()
for t in range(T):
    u = - L @ (state - goal_state)
    state, reward, done, _ = env.step(u)
    data[t] = [t * dt, state[3]*env.RADIUS_WHEEL, state[0]]
    env.render()
end = time.monotonic()
print(end-start)
env.close()

np.savetxt("lqr_data.csv", data, delimiter=",")

