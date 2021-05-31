import numpy as np
from scipy.linalg import solve_continuous_are
from wheeled_inverted_pendulum_env import WIPEnv
import time

x0 = [0.0, 0.0, 0.0, 0.0]

Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
R = 2000*np.eye(1)

env = WIPEnv()
goal_state = np.array([0.0, 80.0, 0.0, 0.0])
f, A, B = env.linearized_dynamics(goal_state, 0.0)

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P

print(L)

state = env.reset(state=x0)
env.render()
time.sleep(1)

start = time.monotonic()
for i in range(1000):
    u = - L @ (state - goal_state)
    state = env.step(u)
    # print(state)
    # print(u)
    env.render()
    if i == 500:
        goal_state = np.array([0.0, 0.0, 0.0, 0.0])
end = time.monotonic()
print(end-start)
env.close()

