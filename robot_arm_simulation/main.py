""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from system_id import *

if __name__ == "__main__":
    args_are_lists = True
    kwargs = {
        "C_L": [5.277437890838794],
        "EA": [104018.69199576143],
        "I_a": [0.33532580589768396],
        "I_i": [0.0000014731849956118856],
        "J_m": [0.0001389511570037035],
        "base_T": [161.06298403595883],
        "f1a": [0.1/200],
        "f1i": [0.15/20000],
        "f1m": [0.15/2000],
        "f2a": [0.1/200],
        "f2i": [0.15/20000],
        "f2m": [0.15/2000],
        "l_1": [0.13062371652873045],
        "l_2": [0.1312578668277771],
        "l_3": [0.18667399317200836],
        "r_a": [0.017493466668933047],
        "r_i": [0.011945599988062716],
        "r_m": [0.012194385107629014],
        "slope1": [-0.3016309553726852],
        "slope2": [0.2230871116913716],
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
    plt.plot(data["Time"], results.y[4, :], 'b', label=f"Calculated Angle")
    plt.legend()
    plt.ylabel("Angle (Radians)")
    plt.xlabel("Time (s)")
    plt.title("Simulated Data with Global Optimization")
