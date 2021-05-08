"""Wheeled Inverted Pendulum by Aaron Schultz for optimal control"""
import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos, pi

class WIPEnv():

    dt = 0.01

   
    LENGTH_PENDULUM = 1 # (m)
    MASS_PENDULUM = 1 # (kg) mass of pendulum
    MOI_PENDULUM = 1 # (Nm) moment of intertia with respect to the pivot
    RADIUS_WHEEL = 1 # (m)
    MASS_WHEEL = 1 # (kg)
    MOI_WHEEL = 1 # (Nm)

    g = 9.81

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    def __init__(self):
        self.viewer = None
        self.state = None

    def reset(self, state=None):
        if state == None:
            self.state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.state = state
        return self.state

    def dynamics(self, x, u):
        Jp = self.MOI_PENDULUM
        mp = self.MASS_PENDULUM
        l = self.LENGTH_PENDULUM
        r = self.RADIUS_WHEEL
        mw = self.MASS_WHEEL
        Jw = self.MOI_WHEEL
        g = self.g
        theta = x[0]
        phi = x[1]
        dtheta = x[2]
        dphi = x[3]
        den = Jp * (Jw + mp * r**2 + mw * r**2) - l**2 * mp**2 * r**2 * np.cos(theta)**2
        ddtheta_num = (Jw + mp * r**2 + mw * r**2) * (-u + mp * g * l * np.sin(theta)) - mp * l * r * np.cos(theta) * (tau + mp * l * rw * dtheta**2 * np.sin(theta))
        ddphi_num = - mp * l * r * np.cos(theta) * (-u + mp * g * l * np.sin(theta)) + Jp * (tau + mp * l * rw * dtheta**2 * np.sin(theta))
        ddtheta = ddtheta_num / den
        ddphi = ddphi_num / den

        return (dtheta, dphi, ddtheta, ddphi)
    
    def linearized_dynamics(self, x_bar, u_bar):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = self.g
        theta1 = x_bar[0]
        theta2 = x_bar[1]
        dtheta1 = x_bar[2]
        dtheta2 = x_bar[3]
        tau = u_bar
        d2 = m2 * l1 * lc2 * cos(theta2) + I2
        d1 = m2 * (l1 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        phi2 = m2 * lc2 * g * sin(theta1 + theta2)
        phi1 = - m2 * l1 * lc2 * (dtheta2 ** 2 + 2 * dtheta2 * dtheta1) * sin(theta2) + (m1 * lc1 + m2 * l1) * g * sin(theta1) + phi2
        ddtheta2_num = (tau + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2)
        ddtheta2_den = I2 - d2 ** 2 / d1
        ddtheta2 = ddtheta2_num / ddtheta2_den
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dtau_dtheta1 = dtau_dtheta2 = dtau_ddtheta1 = dtau_ddtheta2 = 0
        dtau_du = 1

        dd2_dtheta2 = - m2 * l1 * lc2 * sin(theta2)
        dd1_dtheta2 = 2 * dd2_dtheta2
        dphi2_dtheta1 = dphi2_dtheta2 = m2 * lc2 * g * cos(theta1 + theta2)
        dphi1_dtheta1 = (m1 * lc1 + m2 * l1) * g * cos(theta1) + dphi2_dtheta1
        dphi1_dtheta2 = - m2 * l1 * lc2 * (dtheta2 ** 2 + 2 * dtheta2 * dtheta1) * cos(theta2) + dphi2_dtheta2
        dphi1_ddtheta1 = -2 * m2 * l1 * lc2 * dtheta2 * sin(theta2)
        dphi1_ddtheta2 = -2 * m2 * l1 * lc2 * (dtheta2 + dtheta1) * sin(theta2)
        
        ddtheta2_dtheta1 = (dtau_dtheta1 + d2 / d1 * dphi1_dtheta1 - dphi2_dtheta1) / ddtheta2_den

        ddtheta2_num_dtheta2 = dtau_dtheta2 + d2 /d1 * dphi1_dtheta2 + phi1 * (dd2_dtheta2 * d1 - d2 * dd1_dtheta2) / (d1 ** 2) - m2 * l1 * lc2 * dtheta2 ** 2 * cos(theta2) - dphi2_dtheta2
        ddtheta2_den_dtheta2 = - (2 * d2 * dd2_dtheta2 * d1 - dd1_dtheta2 * d2 ** 2) / d1 ** 2
        ddtheta2_dtheta2 = (ddtheta2_num_dtheta2 * ddtheta2_den - ddtheta2_num * ddtheta2_den_dtheta2) / ddtheta2_den ** 2

        ddtheta2_ddtheta1 = (dtau_ddtheta1 + d2 / d1 * dphi1_ddtheta1 - 2 * m2 * l1 * lc2 * dtheta1 * sin(theta2)) / ddtheta2_den
        ddtheta2_ddtheta2 = (dtau_ddtheta2 + d2 / d1 * dphi1_ddtheta2) / ddtheta2_den

        ddtheta1_dtheta1 = - (d2 * ddtheta2_dtheta1 + dphi1_dtheta1) / d1
        ddtheta1_dtheta2 = - ((dd2_dtheta2 * ddtheta2 + d2 * ddtheta2_dtheta2 + dphi1_dtheta2) * d1 - (d2 * ddtheta2 + phi1) * dd1_dtheta2) / d1 ** 2
        ddtheta1_ddtheta1 = -(d2 * ddtheta2_ddtheta1 + dphi1_ddtheta1) / d1
        ddtheta1_ddtheta2 = -(d2 * ddtheta2_ddtheta2 + dphi1_ddtheta2) / d1

        ddtheta2_du = dtau_du / ddtheta2_den
        ddtheta1_du = - d2 * ddtheta2_du / d1

        f_bar = np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [ddtheta1_dtheta1, ddtheta1_dtheta2, ddtheta1_ddtheta1, ddtheta1_ddtheta2],
                      [ddtheta2_dtheta1, ddtheta2_dtheta2, ddtheta2_ddtheta1, ddtheta2_ddtheta2]])
        B = np.array([[0],
                      [0],
                      [ddtheta1_du],
                      [ddtheta2_du]])

        return (f_bar, A, B)


    def energy(self, x, u):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = self.g
        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]

        y1 = - lc1 * cos(theta1)
        y2 = - l1 * cos(theta1) - lc2 * cos(theta1 + theta2)
        PE = m1 * g * y1 + m2 * g * y2

        T1 = 1/2 * I1 * dtheta1 ** 2
        T2 = 1/2 * (m2 * l1 ** 2 + I2 + 2 * m2 * l1 * lc2 * cos(theta2)) * dtheta1 ** 2 + 1/2 * I2 * dtheta2 ** 2 + (I2 + m2 * l1 * lc2 * cos(theta2)) * dtheta1 * dtheta2
        KE = T1 + T2

        return PE + KE


    def f(self, x, u):
        s_augmented = np.append(x, u)

        solve = solve_ivp(self._dsdt, [0, self.dt], s_augmented, t_eval=[self.dt])

        xn = np.zeros(4)
        xn[0] = solve.y[0, 0]
        xn[1] = solve.y[1, 0]
        xn[2] = bound(solve.y[2, 0], -self.MAX_VEL_1, self.MAX_VEL_1)
        xn[3] = bound(solve.y[3, 0], -self.MAX_VEL_2, self.MAX_VEL_2)

        return xn


    def step(self, u):
        x = self.state

        # Add noise to the force action
        if self.torque_noise_max > 0:
            u += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        self.state = self.f(x, u)
        
        self.state += np.random.multivariate_normal([0, 0, 0, 0], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.0001, 0], [0, 0, 0, 0.0001]])

        return self.state

    def _dsdt(self, t, x_augmented):
        u = x_augmented[-1]
        x = x_augmented[:-1]
        dx = self.dynamics(x, u)

        return np.append(dx, 0)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.04  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .03, -.03
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.03)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

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