"""Wheeled Inverted Pendulum by Aaron Schultz for optimal control"""
import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import jacfwd, jit
from gym import spaces

class WIPEnv():

    dt = 0.02

   
    LENGTH_PENDULUM = 0.15 # (m)
    MASS_PENDULUM = 0.5 # (kg) mass of pendulum
    MOI_PENDULUM = MASS_PENDULUM * LENGTH_PENDULUM**2 # (Nm) moment of intertia with respect to the pivot
    RADIUS_WHEEL = 0.02 # (m)
    MASS_WHEEL =  0.02# (kg)
    MOI_WHEEL = 0.5 * MASS_WHEEL * RADIUS_WHEEL**2 # (Nm)

    g = 9.81

    MAX_TORQUE = 1.0 # (Nm)

    def __init__(self):
        self.viewer = None
        self.state = None
        self.action_space = spaces.Box(-self.MAX_TORQUE, self.MAX_TORQUE, shape=(1,), dtype=np.float32)
        high = np.array([np.pi/2, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, state=None):
        if state == None:
            self.state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.state = state
        return np.array(self.state)

    def dynamics(self, x, u):
        Jp = self.MOI_PENDULUM
        mp = self.MASS_PENDULUM
        l = self.LENGTH_PENDULUM / 2
        r = self.RADIUS_WHEEL
        mw = self.MASS_WHEEL
        Jw = self.MOI_WHEEL
        g = self.g
        theta = x[0]
        phi = x[1]
        dtheta = x[2]
        dphi = x[3]
        den = Jp * (Jw + mp * r**2 + mw * r**2) - l**2 * mp**2 * r**2 * jnp.cos(theta)**2
        ddtheta_num = (Jw + mp * r**2 + mw * r**2) * (-u + mp * g * l * jnp.sin(theta)) - mp * l * r * jnp.cos(theta) * (u + mp * l * r * dtheta**2 * jnp.sin(theta))
        ddphi_num = - mp * l * r * jnp.cos(theta) * (-u + mp * g * l * jnp.sin(theta)) + Jp * (u + mp * l * r * dtheta**2 * jnp.sin(theta))
        ddtheta = ddtheta_num / den
        ddphi = ddphi_num / den

        return jnp.array([dtheta, dphi, ddtheta, ddphi])
    
    def linearized_dynamics(self, x_bar, u_bar):
        f = dynamics(x_bar, u_bar)
        A = jacfwd(dynamics, 0)(x_bar, u_bar)
        B = jacfwd(dynamics, 1)(x_bar, u_bar)[np.newaxis].T
        return f, A, B

    def f(self, x, u):
        s_augmented = np.append(x, u)

        solve = solve_ivp(self._dsdt, [0, self.dt], s_augmented, t_eval=[self.dt])
        xn = solve.y[:4,0]

        return xn


    def step(self, u):
        x = self.state

        # Add noise to the force action
        # if self.torque_noise_max > 0:
        #     u += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        self.state = self.f(x, u)

        #self.state += np.random.multivariate_normal([0, 0, 0, 0], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.01, 0], [0, 0, 0, 0.01]])

        done = bool(np.abs(x[0]) > np.pi/2)
        reward = 1.0 if not done else 0.0

        return np.array(self.state), reward, done, {}

    def _dsdt(self, t, x_augmented):
        u = x_augmented[-1]
        x = x_augmented[:-1]
        dx = dynamics(x, u)

        return np.append(dx, 0)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,250)
            self.viewer.set_bounds(-0.2, 1.8, -self.RADIUS_WHEEL, 0.5)

        if s is None: return None

        x = s[1] * self.RADIUS_WHEEL
        theta = -s[0]
        phi = -s[1]

        b,t,l,r = 0, self.LENGTH_PENDULUM, .01, -.01
        jtransform = rendering.Transform(rotation=theta, translation=(x,0))
        link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
        link.add_attr(jtransform)
        link.set_color(0,.8, .8)
        circ = self.viewer.draw_circle(self.RADIUS_WHEEL)
        circ.set_color(.8, .8, 0)
        circ.add_attr(jtransform)
        rim = self.viewer.draw_polygon([(-0.015,-0.002), (-0.015,0.002), (0.015,0.002), (0.015,-0.002)])
        rim.set_color(0, 0, 0)
        rim.add_attr(rendering.Transform(rotation=phi, translation=(x,0)))


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

@jit
def dynamics(x, u):
    LENGTH_PENDULUM = 0.15 # (m)
    MASS_PENDULUM = 0.5 # (kg) mass of pendulum
    MOI_PENDULUM = MASS_PENDULUM * LENGTH_PENDULUM**2 # (Nm) moment of intertia with respect to the pivot
    RADIUS_WHEEL = 0.02 # (m)
    MASS_WHEEL =  0.02# (kg)
    MOI_WHEEL = 0.5 * MASS_WHEEL * RADIUS_WHEEL**2 # (Nm)

    g = 9.81
    Jp = MOI_PENDULUM
    mp = MASS_PENDULUM
    l = LENGTH_PENDULUM / 2
    r = RADIUS_WHEEL
    mw = MASS_WHEEL
    Jw = MOI_WHEEL

    theta = x[0]
    phi = x[1]
    dtheta = x[2]
    dphi = x[3]
    den = Jp * (Jw + mp * r**2 + mw * r**2) - l**2 * mp**2 * r**2 * jnp.cos(theta)**2
    ddtheta_num = (Jw + mp * r**2 + mw * r**2) * (-u + mp * g * l * jnp.sin(theta)) - mp * l * r * jnp.cos(theta) * (u + mp * l * r * dtheta**2 * jnp.sin(theta))
    ddphi_num = - mp * l * r * jnp.cos(theta) * (-u + mp * g * l * jnp.sin(theta)) + Jp * (u + mp * l * r * dtheta**2 * jnp.sin(theta))
    ddtheta = ddtheta_num / den
    ddphi = ddphi_num / den

    return jnp.array([dtheta, dphi, ddtheta, ddphi])