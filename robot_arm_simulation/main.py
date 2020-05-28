""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from system_id import *

if __name__ == "__main__":
    kwargs = {
        # "f1": 0.19940315661064673,
        # "f2": 0.10,
        # "slope1": -0.2893,
        # "slope2": 0.2317,

        "C_L": 6.411670645818824,
        "EA": 37899.2504144791,
        "I_a": 0.7999896971189395,
        "I_i": 0.000016266109268112357,
        "J_m": 0.00011629518077050503,
        "base_T": 154.80197464356823,
        "f1": 0.16773781484758757,
        "f2": 0.17108069792960157,
        "l_1": 0.11937849356032826,
        "l_2": 0.17069807994817562,
        "l_3": 0.1672728621346555,
        "r_a": 0.018552723242634416,
        "r_i": 0.011358754316090505,
        "r_m": 0.011606445505518065,
        "slope1": -0.2839912830537736,
        "slope2": 0.21304560671772632
    }
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
