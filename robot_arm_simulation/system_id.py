"""
A script for identifying the ODE parameters of the robot-arm testbed, based upon 2 sweeps (40s of data) of the arm.

Uses the torque as input, with the arm position as output. The loss is calculated as the SSE of the calculated vs true
arm positions.

Optimization is done with the hyperopt package. Distributed optimization is done using the MongoDB client, implemented
within hyperopt.
"""
from functools import partial
import math
import time
from typing import Union

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from hyperopt_wrap_cost import wrap_cost
from scipy import integrate, interpolate

# Read data
from utils import get_angle, ssign, dssign

data = pd.read_csv(r"test_data.csv")
data.columns = ["Time", "Run", "Direction", "Angle", "Torque", "Tension"]


class SystemProperties:
    optimized_params = {
        # Each element is a list as a convenience - for easy import from mongodb.
        "C_L": [6.6713942381117635],
        "EA": [5780.2205316136415],
        "I_a": [0.7996747467139584],
        "I_i": [3.5622535551286727e-7],
        "J_m": [0.00001766879262780176],
        "base_T": [108.608767011835],
        "f1a": [0.0003955544461215904],
        "f1i": [0.00001431481389581998],
        "f1m": [9.341410254465412e-7],
        "f2a": [0.000018745852944719195],
        "f2i": [0.0003168880536570618],
        "f2m": [0.00010327106681337977],
        "l_1": [0.13609128634917547],
        "l_2": [0.17493368233291345],
        "l_3": [0.15768702528416406],
        "r_a": [0.01944555898288411],
        "r_i": [0.010198789972121942],
        "r_m": [0.014997441255462608],
        "slope1": [-0.23865604682750943],
        "slope2": [0.20331764050832446]
    }

    " Default parameters (empirical values used where possible) for the robot arm simulation"
    defaults = {"r_m": 0.0122, "r_i": 0.0122, "r_a": 0.0189,
                "J_m": 8.235E-5, "I_i": 2.4449E-6, "I_a": 0.5472,
                "l_1": 0.127, "l_2": 0.1524, "l_3": 0.1778,
                "EA": 7057.9,
                "C_L": 7.449, "base_T": 150,
                "slope1": 0., "slope2": 0.,
                "f1m": 0, "f2m": 0,
                "f1i": 0, "f2i": 0,
                "f1a": 0, "f2a": 0,
                }

    angles = {
        "m": get_angle(defaults["l_1"], defaults["l_3"], defaults["l_2"]),
        "i": get_angle(defaults["l_1"], defaults["l_2"], defaults["l_3"]),
        "a": get_angle(defaults["l_2"], defaults["l_3"], defaults["l_1"]),
    }


def ode(time, y, torque_estimator,
        r_m, r_i, r_a,
        J_m, I_i, I_a,
        k_1, k_2, k_3,
        d_1, d_2, d_3,
        base_T, slope_1, slope_2,
        f1m, f2m,
        f1i, f2i,
        f1a, f2a):
    """
    An ordinary differential equation describing the dynamics of the robot arm.

    All values are in standard SI units. Length: m, Time: s, Mass: kg, etc.

    Angular values follow the convention of (+) direction being CCW. Starting from the motor, the chain of components
    (in CCW) direction is the
    1. Motor
    2. Belt (subscript 1, or _i_m)
    3. Idler (Tensioner) pulley
    4. Belt (subscript 2, or _a_i)
    5. Arm pulley
    6. Belt (subscript 3, or _m_a)

    The robot arm does not roll on a flat surface. Depending on the position of the arm, there is torque induced by the
    incline, which is estimated using the slope1 and slope2. This is calculated by
                            T_arm = slope_1 * sin(y[4]) + slope_2 * cos(y[4])
                            where y[4] is the arm angle
    both slope_1 and slope_2 have units of Nm

    Frictional torque will have 6 parameters: 2 for each pulley, all multiplied by the normal force on that pulley
    f1m, f2m are for the motor. f1i, f2i are for the idler. f1a, f2a are for the arm.
                            N = sqrt(T1**2 + T2**2 - 2 * T1 * T2 * cos(pi - theta))
                            friction = -N * (f1 * sign(y[i]) + f2 * y[i])
                        where y[i] is the [angular] velocity, and sign(y[i]) is the direction of travel. The sign function
                        used is a sigmoid, to allow for differentiability.
    f1 has units of m, and f2 has units of s.

    Args:
        time: Used to get the torque input, as an input to torque_estimator.
        y: vector of length 6, with elements (motor {pos, vel}, idler {pos, vel}, and arm {pos, vel})
        torque_estimator: a function, where torque_estimator(time) returns the motor torque at the specified time.
        r_m: radius, motor pulley
        r_i: radius, idler pulley
        r_a: radius, arm
        J_m: Moment of inertia, motor and armature assembly
        I_i: Moment of inertia, idler
        I_a: Moment of inertia, arm
        k_1: Spring constant, belt 1 (units of N/m)
        k_2: Spring constant, belt 2
        k_3: Spring constant, belt 3
        d_1: Dampening, belt 1 (Ns/m)
        d_2: Dampening, belt 2
        d_3: Dampening, belt 3
        base_T: Static / equilibrium tension of the belt.
        slope_1: See above
        slope_2: See above
        f1m: See above
        f2m: See above
        f1i: See above
        f2i: See above
        f1a: See above
        f2a: See above

    Returns:
        The derivative of y with respect to time.

    """

    torque = torque_estimator(time)
    # torque = 0

    # Calculate tensions
    T_m_i, T_i_a, T_a_m = tensions(y, base_T, d_1, d_2, d_3, k_1, k_2, k_3, r_a, r_i, r_m)

    # Accounting for non-level surfaces
    T_arm = slope_1 * math.sin(y[4]) + slope_2 * math.cos(y[4])

    # The normal force is calculated using cosine law
    Nm, Ni, Na = normal_force(T_m_i, T_i_a, T_a_m)

    # Derivatives of positions and velocities (velocity and acceleration)
    dy = np.empty_like(y)
    dy[0] = y[1]
    friction_m = -Nm * (f1m * ssign(y[1]) + f2m * y[1])
    dy[1] = 1 / J_m * ((T_m_i - T_a_m) * r_m + torque + friction_m)

    dy[2] = y[3]
    friction_i = -Ni * (f1i * ssign(y[3]) + f2i * y[3])
    dy[3] = 1 / I_i * ((T_i_a - T_m_i) * r_i + friction_i)

    dy[4] = y[5]
    friction_a = -Na * (f1a * ssign(y[5]) + f2a * y[5])
    dy[5] = 1 / I_a * ((T_a_m - T_i_a) * r_a + T_arm + friction_a)
    # dy[4:6] = 0

    return dy


def normal_force(T_m_i, T_i_a, T_a_m):
    Nm = math.sqrt(T_a_m ** 2 + T_m_i ** 2 - 2 * T_a_m * T_m_i * math.cos(math.pi - SystemProperties.angles["m"]))
    Ni = math.sqrt(T_m_i ** 2 + T_i_a ** 2 - 2 * T_m_i * T_i_a * math.cos(math.pi - SystemProperties.angles["i"]))
    Na = math.sqrt(T_i_a ** 2 + T_a_m ** 2 - 2 * T_i_a * T_a_m * math.cos(math.pi - SystemProperties.angles["a"]))
    return Nm, Ni, Na


def tensions(y, base_T, d_1, d_2, d_3, k_1, k_2, k_3, r_a, r_i, r_m):
    T_m_i = base_T + k_1 * (y[2] * r_i - y[0] * r_m) + d_1 * (y[3] * r_i - y[1] * r_m)
    T_i_a = base_T + k_2 * (y[4] * r_a - y[2] * r_i) + d_2 * (y[5] * r_a - y[3] * r_i)
    T_a_m = base_T + k_3 * (y[0] * r_m - y[4] * r_a) + d_3 * (y[1] * r_m - y[5] * r_a)
    T_m_i = max(T_m_i, 0)
    T_i_a = max(T_i_a, 0)
    T_a_m = max(T_a_m, 0)
    return T_m_i, T_i_a, T_a_m


def ode_const(r_m=SystemProperties.defaults["r_m"], r_i=SystemProperties.defaults["r_i"],
              r_a=SystemProperties.defaults["r_a"],

              J_m=SystemProperties.defaults["J_m"], I_i=SystemProperties.defaults["I_i"],
              I_a=SystemProperties.defaults["I_a"],

              l_1=SystemProperties.defaults["l_1"], l_2=SystemProperties.defaults["l_2"],
              l_3=SystemProperties.defaults["l_3"],

              EA=SystemProperties.defaults["EA"],
              C_L=SystemProperties.defaults["C_L"], base_T=SystemProperties.defaults["base_T"],

              slope1=SystemProperties.defaults["slope1"], slope2=SystemProperties.defaults["slope2"],

              f1m=SystemProperties.defaults["f1m"], f2m=SystemProperties.defaults["f2m"],
              f1i=SystemProperties.defaults["f1i"], f2i=SystemProperties.defaults["f2i"],
              f1a=SystemProperties.defaults["f1a"], f2a=SystemProperties.defaults["f2a"],
              jac=False):
    """

    A constructor function that takes arguments describing the physical characteristics of the robot arm,
    and returns an ode function that takes only two arguments:
        dx/dt = f(x, t)
    Args:
        r_m: same as above
        r_i: same as above
        r_a: same as above
        J_m: same as above
        I_i: same as above
        I_a: same as above
        l_1: length of belt 1 (m)
        l_2: length of belt 2
        l_3: length of belt 3
        EA: Young's modulus * cross sectional area of belt. Units of N
        C_L: Dampening per unit length. Units of N/(m/s) / m = Ns/m^2
        base_T: same as above
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
    # Stiffnesses (N/m)
    # k_1, k_2, k_3 = (k * l_1, k * l_2, k * l_3 * 2 / 3)
    k_1, k_2, k_3 = (EA / l_1, EA / l_2, EA / (l_3 * (2 / 3)))

    # Dampening factors (kg/ (s))
    d_1, d_2, d_3 = (C_L * l_1, C_L * l_2, C_L * (l_3 * 2 / 3))

    if not jac:
        # Set up the torque inputs
        torque_interpolator = interpolate.interp1d(data["Time"],
                                                   data["Torque"],
                                                   bounds_error=False,
                                                   fill_value="extrapolate"
                                                   )

        return partial(ode, torque_estimator=torque_interpolator,
                       r_m=r_m, r_i=r_i, r_a=r_a,
                       J_m=J_m, I_i=I_i, I_a=I_a,
                       k_1=k_1, k_2=k_2, k_3=k_3,
                       d_1=d_1, d_2=d_2, d_3=d_3,
                       base_T=base_T, slope_1=slope1, slope_2=slope2,
                       f1m=f1m, f2m=f2m, f1i=f1i, f2i=f2i, f1a=f1a, f2a=f2a)

    else:
        def J(t, y):
            # Gradients of the tensions
            dT_m_i = np.array([-k_1 * r_m, -d_1 * r_m, k_1 * r_i, d_1 * r_i, 0, 0])
            dT_i_a = np.array([0, 0, -k_2 * r_i, -d_2 * r_i, k_2 * r_a, d_2 * r_a])
            dT_a_m = np.array([k_3 * r_m, d_3 * r_m, 0, 0, -k_3 * r_a, -d_3 * r_a])

            # Gradients of the frictions
            # Friction is calculated as
            #   friction_m = -Nm * (f1m * np.sign(y[1]) + f2m * y[1])
            # Dependent on the normal forces Nm, Ni, and Na, calculated as
            #   Nm = math.sqrt(T_a_m ** 2 + T_m_i ** 2 - 2 * T_a_m * T_m_i * math.cos(math.pi - SystemProperties.angles["m"]))

            # Note that d/dx sqrt(f(x)) = f'(x) / (2 * sqrt(x))
            # and taking f(x) = a**2 + b**2 - 2 * a * b * cos(phi),
            # f'(x) = 2 * a * a' + 2 * b * b' - 2 * cos(phi) * (a' * b + a * b')
            # We have all the a', b' available.

            T_m_i, T_i_a, T_a_m = tensions(y, base_T, d_1, d_2, d_3, k_1, k_2, k_3, r_a, r_i, r_m)
            Nm, Ni, Na = normal_force(T_m_i, T_i_a, T_a_m)

            # Gradients of normal forces
            dNm = ((T_a_m * dT_a_m
                    + T_m_i * dT_m_i
                    - (math.cos(math.pi - SystemProperties.angles["m"])
                       * (dT_a_m * T_m_i + dT_m_i * T_a_m)))
                   / (Nm + 1E-15))
            dNi = ((T_m_i * dT_m_i
                    + T_i_a * dT_i_a
                    - (math.cos(math.pi - SystemProperties.angles["i"])
                       * (dT_m_i * T_i_a + dT_i_a * T_m_i)))
                   / (Ni + 1E-15))
            dNa = ((T_i_a * dT_i_a + T_a_m * dT_a_m
                    - (math.cos(math.pi - SystemProperties.angles["a"])
                       * (dT_i_a * T_a_m + dT_a_m * T_i_a)))
                   / (Na + 1E-15))

            # Calculate the gradient of friction:
            #   friction_m = -Nm * (f1m * ssign(y[1]) + f2m * y[1])
            dfrictionm = -dNm * (f2m * y[1]) - np.eye(6)[1] * Nm * (f1m * dssign(y[1]) + f2m)
            dfrictioni = -dNi * (f2i * y[3]) - np.eye(6)[3] * Nm * (f1i * dssign(y[3]) + f2i)
            dfrictiona = -dNa * (f2a * y[5]) - np.eye(6)[5] * Nm * (f1a * dssign(y[5]) + f2a)

            # Calculate the final gradient: d/dy ( dy/dt )
            jac = np.zeros((6, 6))
            jac[0, :] = [0, 1, 0, 0, 0, 0]
            jac[1, :] = 1 / J_m * ((dT_m_i - dT_a_m) * r_m + dfrictionm)
            jac[2, :] = [0, 0, 0, 1, 0, 0]
            jac[3, :] = 1 / I_i * ((dT_i_a - dT_m_i) * r_i + dfrictioni)
            jac[4, :] = [0, 0, 0, 0, 0, 1]
            jac[5, :] = 1 / I_a * ((dT_a_m - dT_i_a) * r_a + dfrictiona)
            # jac[4:6, :] = 0
            return jac

        return J


def objective(x: Union[list, dict], tight_tol=True, argnames=None):
    """ Same function signature as ode_const(). Used to define the objective for hyperopt """
    if isinstance(x, dict):
        kwargs = x
    else:
        # sensitivity_test does not supply arguments as dict
        kwargs = {k: v for k, v in zip(argnames, x)}
    ode = ode_const(**kwargs)

    y0 = np.array([5.33249302e-01, 0, 5.33249314e-01, 0, 0.344213835078322, 0])

    start = time.time()
    tol_params = dict() if tight_tol else {"rtol": 1E-2, "atol": 1E-4}
    results = integrate.solve_ivp(
        fun=ode,
        t_span=(0, 40),
        y0=y0,
        t_eval=data["Time"],
        method='LSODA',
        jac=ode_const(jac=True, **kwargs),
        **tol_params,
    )

    return {
        # Main results
        'loss': np.sum((results.y[4, :] - data["Angle"].values) ** 2),
        'status': STATUS_OK,

        # Other results
        'start_time': start,
        'elapsed': time.time() - start,
        # 'responses': results.y.tostring()
    }


if __name__ == "__main__":
    trials = MongoTrials(r'mongo://melco.cs.ualberta.ca:27017/rasim_db/jobs',
                         exp_key='exp15-new-friction-all-params')
    space = {
        'r_m': hp.uniform('r_m', 0.01, 0.015),
        'r_i': hp.uniform('r_i', 0.01, 0.015),
        'r_a': hp.uniform('r_a', 0.017, 0.021),
        'J_m': hp.loguniform('J_m', -11, -8),
        'I_i': hp.loguniform('I_i', -15, -11),
        'I_a': hp.uniform('I_a', 0.3, 0.8),
        'l_1': hp.uniform('l_1', 0.1, 0.15),
        'l_2': hp.uniform('l_2', 0.125, 0.175),
        'l_3': hp.uniform('l_3', 0.15, 0.20),
        'slope1': hp.normal('slope1', -0.28, 0.02),
        'slope2': hp.normal('slope2', 0.21, 0.02),
        'EA': hp.loguniform('EA', 8, 12.5),
        'C_L': hp.uniform('C_L', 5, 10),
        'base_T': hp.uniform('base_T', 100, 200),
        'f1m': hp.lognormal('f1m', math.log(0.00015), 1),
        'f2m': hp.lognormal('f2m', math.log(0.00015), 1),
        'f1i': hp.lognormal('f1i', math.log(0.00001), 1),
        'f2i': hp.lognormal('f2i', math.log(0.00001), 1),
        'f1a': hp.lognormal('f1a', math.log(0.0004), 1),
        'f2a': hp.lognormal('f2a', math.log(0.0006), 1),
    }

    best = fmin(
        wrap_cost(objective, timeout=480, iters=1),
        space=space,
        algo=tpe.suggest,
        max_evals=40000,
        trials=trials,
        max_queue_len=32,
    )

    # space = hp.uniform('x', -2, 2)
    # best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
