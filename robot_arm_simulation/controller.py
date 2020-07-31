from functools import partial
import numpy as np

from system_id import RobotArmDynamics
from utils import GP, Noise


class PIDControlRobotArm(RobotArmDynamics):
    def __init__(self, K_p, T_i, T_d, max_speed=0.56, accel=0.425, **kwargs):
        self.K_p = K_p
        self.K_i = self.K_p / T_i
        self.K_d = self.K_p * T_d

        # Params for control
        self.y = None
        self.dy = None
        self.E = None

        self.max_speed = max_speed
        self.accel = accel

        super().__init__(**kwargs)

        self._init_setpoint_times()

    @property
    def y0(self):
        return np.hstack([super().y0, 0])

    def _init_setpoint_times(self):
        """Assumes a period of 20 seconds"""
        t_accel = self.max_speed / self.accel
        self.angle_diff_acc = 0.5 * self.accel * t_accel ** 2
        t_max_speed = (self.arm_angle_b - self.arm_angle_a - 2 * self.angle_diff_acc) / self.max_speed
        t_total_moving = t_accel + t_max_speed + t_accel

        times = (0, t_accel, t_max_speed, t_accel, 10 - t_total_moving,
                 t_accel, t_max_speed, t_accel, 10 - t_total_moving)
        self.c = np.cumsum(times)

    def ode(self, time, y):
        """
        The dynamics of a PID controlled robot arm system.
        Args:
            time: Used in the I and D parts of PID control
            y: vector of length 9, with elements (motor {pos, vel}, idler {pos, vel}, and arm {pos, vel}), and
            the 1 additional parameter used for PID control (E)

        Returns:
            The derivative of y with respect to time. A vector of length 9
        """

        self.y, self.dy, self.E = y[4:7]

        dy = np.empty_like(y)
        dy[:6] = super().ode(time, y[:6])
        dy[6] = self.setpoint(time) - y[4]

        return dy

    def torque(self, time):
        return self.control(time, y=self.y, dy=self.dy, E=self.E)

    def setpoint(self, time):
        # Trapezoidal velocity profile.
        # 2 parameters: max_speed, and accel

        t = time % 20  # The time since the beginning of the period.

        max_speed = self.max_speed  # Max speed of the robot arm
        accel = self.accel  # Acceleration of the robot arm
        angle_diff_acc = self.angle_diff_acc  # Displacement during acceleration phase
        c = self.c

        if c[0] <= t < c[1]:
            return 0.5 * self.accel * t ** 2 + self.arm_angle_a
        elif t < c[2]:
            return (t - c[1]) * max_speed + self.arm_angle_a + angle_diff_acc
        elif t < c[3]:
            return self.arm_angle_b - 0.5 * accel * (t - c[3]) ** 2
        elif t < c[4]:
            return self.arm_angle_b
        elif t < c[5]:
            return self.arm_angle_b - 0.5 * accel * (t - c[4]) ** 2
        elif t < c[6]:
            return self.arm_angle_b - angle_diff_acc - (t - c[5]) * max_speed
        elif t < c[7]:
            return self.arm_angle_a + 0.5 * accel * (t - c[7]) ** 2
        elif t < c[8]:
            return self.arm_angle_a
        else:
            raise ValueError("Unspecified time in setpoint")

    def dsetpoint(self, time):
        t = time % 20  # The time since the beginning of the period.

        max_speed = self.max_speed  # Max speed of the robot arm
        accel = self.accel  # Acceleration of the robot arm
        c = self.c

        if c[0] <= t < c[1]:
            return self.accel * t
        elif t < c[2]:
            return max_speed
        elif t < c[3]:
            return - accel * (t - c[3])
        elif t < c[4]:
            return 0
        elif t < c[5]:
            return - accel * (t - c[4])
        elif t < c[6]:
            return - max_speed
        elif t < c[7]:
            return accel * (t - c[7])
        elif t < c[8]:
            return 0
        else:
            raise ValueError("Unspecified time in dsetpoint")

    def J(self, t, y):
        """ The Jacobian of ode, used for improving speed of the numerical solver. Avoids having the solver
        calculate the Jacobian using finite difference formulas

        Args:
            t: time
            y: state

        Returns:
            7x7 ndarray of the Jacobian of dy/dt; i.e. d/dy(dy/dt)
        """
        jac = np.zeros((7, 7))
        jac[:6, :6] = super().J(t, y[:6])
        jac[6, :] = - np.eye(7)[4, :]
        return jac

    def control(self, t, y, dy, E):
        dyr = self.dsetpoint(t)
        yr = self.setpoint(t)
        return self.K_d * (dyr - dy) + self.K_p * (yr - y) + self.K_i * E


class PIDDisturbRobotArm(PIDControlRobotArm):
    def __init__(self, sigma, l, seed=0, use_GP=True, **kwargs):
        self.sigma = sigma
        self.l = l
        self.use_GP = use_GP
        self.seed = seed

        self.Noise = Noise(self.sigma)
        self.GP = GP(sigma=self.sigma, l=self.l)

        super().__init__(**kwargs)

    def torque(self, time):
        if self.use_GP:
            ftdisturb = self.GP.gp(seed=int(time // 20)+self.seed)
        else:
            ftdisturb = self.Noise.noise()
        return super().torque(time) + ftdisturb(time % 20 / 20)
