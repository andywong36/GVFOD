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

from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.mongoexp import MongoTrials
import numpy as np
import pandas as pd
from scipy import integrate, interpolate

from hyperopt_wrap_cost import wrap_cost

# Read data
data = pd.read_csv(r"test_data.csv")
data.columns = ["Time", "Run", "Direction", "Angle", "Torque", "Tension"]


def ode(time, y, torque_estimator,
        r_m, r_i, r_a,
        J_m, I_i, I_a,
        k_1, k_2, k_3,
        d_1, d_2, d_3,
        base_T, slope_1, slope_2,
        f1, f2):
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

    Frictional torque is calculated in a method that is not necessarily grounded in physics.
                            friction = - (f1 * sign(y[5]) + f2 * y[5])
                            where y[5] is the arm [angular] velocity.
    f1 has units of Nm, and f2 has units of Nm / (m/s) = Ns.

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
        f1: See above
        f2: See above

    Returns:

    """

    torque = torque_estimator(time)
    # torque = 0

    # Calculate tensions
    T_m_i = base_T + k_1 * (y[2] * r_i - y[0] * r_m) + d_1 * (y[3] * r_i - y[1] * r_m)
    T_a_m = base_T + k_3 * (y[0] * r_m - y[4] * r_a) + d_3 * (y[1] * r_m - y[5] * r_a)
    T_i_a = base_T + k_2 * (y[4] * r_a - y[2] * r_i) + d_2 * (y[5] * r_a - y[3] * r_i)

    T_m_i = max(T_m_i, 0)
    T_a_m = max(T_a_m, 0)
    T_i_a = max(T_i_a, 0)

    # Accounting for non-level surfaces
    T_arm = slope_1 * math.sin(y[4]) + slope_2 * math.cos(y[4])

    # Derivatives of positions and velocities (velocity and acceleration)
    dy = np.empty_like(y)
    dy[0] = y[1]
    dy[1] = 1 / J_m * ((T_m_i - T_a_m) * r_m + torque)

    dy[2] = y[3]
    dy[3] = 1 / I_i * ((T_i_a - T_m_i) * r_i)

    friction = - (f1 * np.sign(y[5]) + f2 * y[5])
    dy[4] = y[5]
    dy[5] = 1 / I_a * ((T_a_m - T_i_a) * r_a + T_arm + friction)
    # dy[4:6] = 0

    return dy


def ode_const(r_m=0.0122, r_i=0.0122, r_a=0.0189,
              J_m=8.235E-5, I_i=2.4449E-6, I_a=0.5472,
              l_1=0.127, l_2=0.1524, l_3=0.1778,
              # k=2658178,
              EA=7057.9,
              C_L=7.449, base_T=150,
              slope1=0., slope2=0.,
              f1=0.2, f2=0.2,
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
        f1: same as above
        f2: same as above
        jac: Boolean. Whether to return the Jacobian of the ODE instead, d/dx (dx/dt)

    Returns:

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
                       f1=f1, f2=f2)

    else:
        grads_tensions = np.zeros((3, 6))  # Gradients of the tensions T_m_i, T_a_m, T_i_a respectively
        grads_tensions[0, :] = [-k_1 * r_m, -d_1 * r_m, k_1 * r_i, d_1 * r_i, 0, 0]  # T_m_i
        grads_tensions[1, :] = [k_3 * r_m, d_3 * r_m, 0, 0, -k_3 * r_a, -d_3 * r_a]  # T_a_m
        grads_tensions[2, :] = [0, 0, -k_2 * r_i, -d_2 * r_i, k_2 * r_a, d_2 * r_a]  # T_i_a

        jac = np.zeros((6, 6))
        jac[0, :] = [0, 1, 0, 0, 0, 0]
        jac[1, :] = 1 / J_m * r_m * (grads_tensions[0, :] - grads_tensions[1, :])
        jac[2, :] = [0, 0, 0, 1, 0, 0]
        jac[3, :] = 1 / I_i * r_i * (grads_tensions[2, :] - grads_tensions[0, :])
        jac[4, :] = [0, 0, 0, 0, 0, 1]
        jac[5, :] = 1 / I_a * r_a * (grads_tensions[1, :] - grads_tensions[2, :])
        # jac[4:6, :] = 0

        return lambda t, y: jac


def objective(x: dict):
    """ Same function signature as ode_const(). Used to define the objective for hyperopt """
    ode = ode_const(**x)

    y0 = np.array([5.33249302e-01, 0, 5.33249314e-01, 0, 0.344213835078322, 0])

    start = time.time()
    results = integrate.solve_ivp(
        fun=ode,
        t_span=(0, 40),
        y0=y0,
        t_eval=data["Time"],
        method='LSODA',
        jac=ode_const(jac=True),
        rtol=1E-2,
        atol=1E-4,
    )

    return {
        # Main results
        'loss': np.sum((results.y[4, :] - data["Angle"].values) ** 2),
        'status': STATUS_OK,

        # Other results
        'start_time': start,
        'elapsed': time.time() - start,
        'responses': results.y.tostring()
    }


if __name__ == "__main__":
    trials = MongoTrials(r'mongo://melco.cs.ualberta.ca:27017/rasim_db/jobs',
                         exp_key='exp12-final1')
    space = {
        # 'slope1': hp.uniform('slope1', 0.28, 0.30),
        # 'slope2': hp.uniform('slope2', -0.32, -0.12),
        'r_m': hp.uniform('r_m', 0.01, 0.015),
        'r_i': hp.uniform('r_i', 0.01, 0.015),
        'r_a': hp.uniform('r_a', 0.017, 0.021),
        'J_m': hp.loguniform('J_m', -11, -8),
        'I_i': hp.loguniform('I_i', -15, -11),
        'I_a': hp.uniform('I_a', 0.3, 0.8),
        'l_1': hp.uniform('l_1', 0.1, 0.15),
        'l_2': hp.uniform('l_2', 0.125, 0.175),
        'l_3': hp.uniform('l_3', 0.15, 0.20),
        'slope1': hp.uniform('slope1', -0.31, -0.28),
        'slope2': hp.uniform('slope2', 0.205, 0.24),
        'EA': hp.loguniform('EA', 8, 12.5),
        'C_L': hp.uniform('C_L', 5, 10),
        'base_T': hp.uniform('base_T', 100, 200),
        'f1': hp.uniform('f1', 0, 10),
        'f2': hp.uniform('f2', 0, 10),
    }

    best = fmin(
        wrap_cost(objective, timeout=360, iters=1),
        space=space,
        algo=tpe.suggest,
        max_evals=40000,
        trials=trials
    )

    # space = hp.uniform('x', -2, 2)
    # best = fmin(math.sin, hp.uniform('x', -2, 2), trials=trials, algo=tpe.suggest, max_evals=10)
