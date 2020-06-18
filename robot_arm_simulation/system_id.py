"""
A script for identifying the ODE parameters of the robot-arm testbed, based upon 2 sweeps (40s of data) of the arm.

Uses the torque as input, with the arm position as output. The loss is calculated as the SSE of the calculated vs true
arm positions.

Optimization is done with the hyperopt package. Distributed optimization is done using the MongoDB client, implemented
within hyperopt.
"""
import math

import numpy as np
import pandas as pd
from scipy import interpolate

from utils import get_angle, ssign, dssign


class RobotArmDynamics:
    _optimized_params = {
        # Each element is a list as a convenience - for easy import from mongodb.
        "C_L": [6.6633544478366495],
        "EA": [174388.62144965833],
        "I_a": [0.7906157135035278],
        "I_i": [0.000008558186054715617],
        "J_m": [0.00013108554681123485],
        "base_T": [110.87862925115707],
        "f1a": [1.4129360140919265e-7],
        "f1i": [0.000007626581489053154],
        "f1m": [0.0006634106668629292],
        "f2a": [0.00007716518208965043],
        "f2i": [0.00003074601549551998],
        "f2m": [0.00004162790299003434],
        "l_1": [0.12650016515777088],
        "l_2": [0.15193752939477279],
        "l_3": [0.17729639030823133],
        "r_a": [0.018934005373129833],
        "r_i": [0.010188008453955382],
        "r_m": [0.014895654953218457],
        "slope1": [-0.23723147775251083],
        "slope2": [0.21294515050284282]
    }
    optimized_params = {k: v[0] for k, v in _optimized_params.items()}

    " Default parameters (empirical values used where possible) for the robot arm simulation"
    r_m, r_i, r_a = 0.0122, 0.0122, 0.0189
    J_m, I_i, I_a = 8.235E-5, 2.4449E-6, 0.5472
    l_1, l_2, l_3 = 0.127, 0.1524, 0.1778
    EA, C_L = 7057.9, 7.449
    base_T = 150,
    slope1, slope2 = 0., 0.
    f1m, f2m, f1i, f2i, f1a, f2a = 0., 0., 0., 0., 0., 0.

    # Read data
    data = pd.read_csv(r"test_data.csv")
    data.columns = ["Time", "Run", "Direction", "Angle", "Torque", "Tension"]

    @staticmethod
    def _normal_force(T_m_i, T_i_a, T_a_m, angle_m, angle_i, angle_a):
        # Calculates the normal force applied on the belt by the pulley (used for calculating friction)
        Nm = math.sqrt(T_a_m ** 2 + T_m_i ** 2 - 2 * T_a_m * T_m_i * math.cos(math.pi - angle_m))
        Ni = math.sqrt(T_m_i ** 2 + T_i_a ** 2 - 2 * T_m_i * T_i_a * math.cos(math.pi - angle_i))
        Na = math.sqrt(T_i_a ** 2 + T_a_m ** 2 - 2 * T_i_a * T_a_m * math.cos(math.pi - angle_a))
        return Nm, Ni, Na

    @staticmethod
    def _tensions(y, base_T, d_1, d_2, d_3, k_1, k_2, k_3, r_a, r_i, r_m):
        T_m_i = base_T + k_1 * (y[2] * r_i - y[0] * r_m) + d_1 * (y[3] * r_i - y[1] * r_m)
        T_i_a = base_T + k_2 * (y[4] * r_a - y[2] * r_i) + d_2 * (y[5] * r_a - y[3] * r_i)
        T_a_m = base_T + k_3 * (y[0] * r_m - y[4] * r_a) + d_3 * (y[1] * r_m - y[5] * r_a)
        T_m_i = max(T_m_i, 0)
        T_i_a = max(T_i_a, 0)
        T_a_m = max(T_a_m, 0)
        return T_m_i, T_i_a, T_a_m

    def __init__(self, use_optimized_params=False, **kwargs):
        """
        Sets up the ordinary differential equation describing the dynamics of the robot arm.
        After initialization, there will be an ode method ode(self, time, y) that describes
            the dynamics of the robot arm:
                                    dx/dt = f(x, t)
        All values are in standard SI units. Length: m, Time: s, Mass: kg, etc.

        Notes on the equations:
            Angular values follow the convention of (+) direction being CCW. Starting from the motor,
            the chain of components (in CCW) direction is the
            1. Motor
            2. Belt (subscript 1, or _i_m)
            3. Idler (Tensioner) pulley
            4. Belt (subscript 2, or _a_i)
            5. Arm pulley
            6. Belt (subscript 3, or _m_a)

            The robot arm does not roll on a flat surface. Depending on the position of the arm, there is torque induced
            by the incline, which is estimated using the slope1 and slope2. This is calculated by
                                    T_arm = slope1 * sin(y[4]) + slope2 * cos(y[4])
                                    where y[4] is the arm angle
            both slope1 and slope2 have units of Nm

            Frictional torque will have 6 parameters: 2 for each pulley, all multiplied by the normal force on that
            pulley. f1m, f2m are for the motor. f1i, f2i are for the idler. f1a, f2a are for the arm.
                                    N = sqrt(T1**2 + T2**2 - 2 * T1 * T2 * cos(pi - theta))
                                    friction = -N * (f1 * sign(y[i]) + f2 * y[i])
                where y[i] is the [angular] velocity, and sign(y[i]) is the direction of travel. The sign function
                used is a sigmoid, to allow for differentiability.
            f1 has units of m, and f2 has units of s.

        Args:
            Use the stored 'optimal' paramters.

        Keyword Args:
            r_m: radius, motor pulley
            r_i: radius, idler pulley
            r_a: radius, arm
            J_m: Moment of inertia, motor and armature assembly
            I_i: Moment of inertia, idler
            I_a: Moment of inertia, arm
            l_1: length of belt 1 (m)
            l_2: length of belt 2
            l_3: length of belt 3
            EA: Young's modulus * cross sectional area of belt. Units of N
            C_L: Dampening per unit length. Units of N/(m/s) / m = Ns/m^2
            base_T: Static / equilibrium tension of the belt.
            slope1: same as above
            slope2: same as above
            f1m: same as above
            f2m: same as above
            f1i: same as above
            f2i: same as above
            f1a: same as above
            f2a: same as above
            jac: Boolean. Whether to return the Jacobian of the ODE instead, d/dx (dx/dt)

        Returns:
            A callable function, of the form f(t,y) which returns one of two possibilities
            if not jac:
                returns a 6x1 vector of dy/dt
            if jac:
                returns a 6x6 gradient matrix, of d/dy (dy/dt)

        """
        if not all(k in RobotArmDynamics.__dict__ for k in kwargs.keys()):
            raise ValueError("Unknown keyword argument provided")

        # Set the parameters of the robot arm
        if use_optimized_params:
            self.__dict__.update({k: v[0] for k, v in self.optimized_params.items()})
        self.__dict__.update(kwargs)  # Now all the parameters can be used with self.k_1 or self.k_2 etc...

        self.angle_m = get_angle(self.l_1, self.l_3, self.l_2)
        self.angle_i = get_angle(self.l_1, self.l_2, self.l_3)
        self.angle_a = get_angle(self.l_2, self.l_3, self.l_1)

        # Calculate the spring constant of the 3 belt lengths (N/m)
        self.k_1, self.k_2, self.k_3 = (self.EA / self.l_1, self.EA / self.l_2, self.EA / (self.l_3 * (2 / 3)))

        # Calculate the dampening of the 3 belt lengths (kg/s = Ns/m)
        self.d_1, self.d_2, self.d_3 = (self.C_L * self.l_1, self.C_L * self.l_2, self.C_L * (self.l_3 * 2 / 3))

        # Set up interpolation function for torque calculated from data (can be overwritten with a PID controller)
        self.torque_interpolator = interpolate.interp1d(
            self.data["Time"],
            self.data["Torque"],
            bounds_error=False,
            fill_value="extrapolate",
        )

    def ode(self, time, y):
        """
        The dynamics of the robot arm system. See __init__ docs for more information.

        Args:
            time: Used to get the torque input, as an input to torque_estimator.
            y: vector of length 6, with elements (motor {pos, vel}, idler {pos, vel}, and arm {pos, vel})

        Returns:
            The derivative of y with respect to time.
        """

        torque = self.torque(time)
        # torque = 0

        # Calculate tensions
        T_m_i, T_i_a, T_a_m = self._tensions(y, self.base_T, self.d_1, self.d_2, self.d_3, self.k_1, self.k_2, self.k_3,
                                             self.r_a, self.r_i, self.r_m)

        # Accounting for non-level surfaces
        T_arm = self.slope1 * math.sin(y[4]) + self.slope2 * math.cos(y[4])

        # The normal force is calculated using cosine law
        Nm, Ni, Na = self._normal_force(T_m_i, T_i_a, T_a_m, self.angle_m, self.angle_i, self.angle_a)

        # Derivatives of positions and velocities (velocity and acceleration)
        dy = np.empty_like(y)
        dy[0] = y[1]
        friction_m = -Nm * (self.f1m * ssign(y[1]) + self.f2m * y[1])
        dy[1] = 1 / self.J_m * ((T_m_i - T_a_m) * self.r_m + torque + friction_m)

        dy[2] = y[3]
        friction_i = -Ni * (self.f1i * ssign(y[3]) + self.f2i * y[3])
        dy[3] = 1 / self.I_i * ((T_i_a - T_m_i) * self.r_i + friction_i)

        dy[4] = y[5]
        friction_a = -Na * (self.f1a * ssign(y[5]) + self.f2a * y[5])
        dy[5] = 1 / self.I_a * ((T_a_m - T_i_a) * self.r_a + T_arm + friction_a)
        # dy[4:6] = 0

        return dy

    def torque(self, t):
        # Can be replaced with a controller. The default torque is interpolated from data.
        return self.torque_interpolator(t)

    def J(self, t, y):
        """ The Jacobian of ode, used for improving speed of the numerical solver. Avoids having the solver
        calculate the Jacobian using finite difference formulas

        Args:
            t: time
            y: state

        Returns:
            6x6 ndarray of the Jacobian of dy/dt; i.e. d/dy(dy/dt)
        """
        # Gradients of the tensions
        dT_m_i = np.array([-self.k_1 * self.r_m, -self.d_1 * self.r_m, self.k_1 * self.r_i, self.d_1 * self.r_i, 0, 0])
        dT_i_a = np.array([0, 0, -self.k_2 * self.r_i, -self.d_2 * self.r_i, self.k_2 * self.r_a, self.d_2 * self.r_a])
        dT_a_m = np.array([self.k_3 * self.r_m, self.d_3 * self.r_m, 0, 0, -self.k_3 * self.r_a, -self.d_3 * self.r_a])

        # Gradients of the frictions
        # Friction is calculated as
        #   friction_m = -Nm * (f1m * np.sign(y[1]) + f2m * y[1])
        # Dependent on the normal forces Nm, Ni, and Na, calculated as
        #   Nm = math.sqrt(T_a_m ** 2 + T_m_i ** 2
        #                  - 2 * T_a_m * T_m_i * math.cos(math.pi - angle_m))

        # Note that d/dx sqrt(f(x)) = f'(x) / (2 * sqrt(x))
        # and taking f(x) = a**2 + b**2 - 2 * a * b * cos(phi),
        # f'(x) = 2 * a * a' + 2 * b * b' - 2 * cos(phi) * (a' * b + a * b')
        # We have all the a', b' available.

        T_m_i, T_i_a, T_a_m = self._tensions(y, self.base_T, self.d_1, self.d_2, self.d_3, self.k_1, self.k_2, self.k_3,
                                             self.r_a, self.r_i, self.r_m)
        Nm, Ni, Na = self._normal_force(T_m_i, T_i_a, T_a_m, self.angle_m, self.angle_i, self.angle_a)

        # Gradients of normal forces
        dNm = ((T_a_m * dT_a_m
                + T_m_i * dT_m_i
                - (math.cos(math.pi - self.angle_m)
                   * (dT_a_m * T_m_i + dT_m_i * T_a_m)))
               / (Nm + 1E-15))
        dNi = ((T_m_i * dT_m_i
                + T_i_a * dT_i_a
                - (math.cos(math.pi - self.angle_i)
                   * (dT_m_i * T_i_a + dT_i_a * T_m_i)))
               / (Ni + 1E-15))
        dNa = ((T_i_a * dT_i_a
                + T_a_m * dT_a_m
                - (math.cos(math.pi - self.angle_a)
                   * (dT_i_a * T_a_m + dT_a_m * T_i_a)))
               / (Na + 1E-15))

        # Calculate the gradient of friction:
        #   friction_m = -Nm * (f1m * ssign(y[1]) + f2m * y[1])
        dfrictionm = -dNm * (self.f2m * y[1]) - np.eye(6)[1] * Nm * (self.f1m * dssign(y[1]) + self.f2m)
        dfrictioni = -dNi * (self.f2i * y[3]) - np.eye(6)[3] * Nm * (self.f1i * dssign(y[3]) + self.f2i)
        dfrictiona = -dNa * (self.f2a * y[5]) - np.eye(6)[5] * Nm * (self.f1a * dssign(y[5]) + self.f2a)

        # Calculate the final gradient: d/dy ( dy/dt )
        jac = np.zeros((6, 6))
        jac[0, :] = [0, 1, 0, 0, 0, 0]
        jac[1, :] = 1 / self.J_m * ((dT_m_i - dT_a_m) * self.r_m + dfrictionm)
        jac[2, :] = [0, 0, 0, 1, 0, 0]
        jac[3, :] = 1 / self.I_i * ((dT_i_a - dT_m_i) * self.r_i + dfrictioni)
        jac[4, :] = [0, 0, 0, 0, 0, 1]
        jac[5, :] = 1 / self.I_a * ((dT_a_m - dT_i_a) * self.r_a + dfrictiona)
        # jac[4:6, :] = 0
        return jac
