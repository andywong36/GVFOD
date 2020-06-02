""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from system_id import *

if __name__ == "__main__":
    args_are_lists = True
    kwargs = {
        # "f1": 0.19940315661064673,
        # "f2": 0.10,
        # "slope1": -0.2893,
        # "slope2": 0.2317,

        "C_L": [6.429838746810943],
        "EA": [10140.836964147904],
        "I_a": [0.7996811845733756],
        "I_i": [0.000011100755197769705],
        "J_m": [0.0000974240281519727],
        "base_T": [139.23187611392822],
        "f1": [0.1715605339128385],
        "f2": [0.16516476291668136],
        "l_1": [0.1185950561384755],
        "l_2": [0.17157324573272187],
        "l_3": [0.17158676110012805],
        "r_a": [0.018132430213991843],
        "r_i": [0.010307059265617799],
        "r_m": [0.01131146148931743],
        "slope1": [-0.2811648245908479],
        "slope2": [0.20732678232881055]
    }
    if args_are_lists:
        for key, val in kwargs.items():
            kwargs[key] = val[0]
    ode = ode_const(**kwargs)

    # Initial conditions: where the displacement of each pulley is equal (0 net tension), and all 3 pulley are
    # stationary

    arm_angle_0 = data["Angle"][0]
    y0 = np.array([arm_angle_0 * kwargs["r_a"] / kwargs["r_m"], 0,
                   arm_angle_0 * kwargs["r_a"] / kwargs["r_i"], 0,
                   arm_angle_0, 0])
    # y0 = np.array([5.33249302e-01, 0, 5.33249314e-01, 0, 0.344213835078322, 0])

    # Solve ODE
    start = time.time()
    results = integrate.solve_ivp(
        fun=ode,
        t_span=(0, 40),
        y0=y0,
        t_eval=data["Time"],
        method='LSODA',
        jac=ode_const(jac=True, **kwargs),
        rtol=1E-2,
        atol=1E-4,
    )
    print(f" Time taken: {time.time() - start}")
    print("Done")

    # Plot results
    plt.plot(data["Time"], data["Angle"], 'r', label="True Angle")
    plt.plot(data["Time"], results.y[4, :], 'b', label=f"Calculated Angle, EA = {kwargs['EA']}")
    plt.legend()
    plt.ylabel("Angle (Radians)")
    plt.xlabel("Time (s)")
    plt.title("Simulated Data with Global Optimization")
