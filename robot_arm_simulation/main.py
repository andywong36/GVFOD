""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """
import time

import matplotlib.pyplot as plt
from scipy import integrate

from system_id import *

if __name__ == "__main__":
    kwargs = RobotArmDynamics.optimized_params
    model = RobotArmDynamics(use_extended_data=True, **kwargs)

    # Initial conditions: where the displacement of each pulley is equal (0 net tension), and all 3 pulley are
    # stationary

    # arm_angle_0 = data["Angle"][0]
    # y0 = np.array([arm_angle_0 * kwargs["r_a"] / kwargs["r_m"], 0,
    #                arm_angle_0 * kwargs["r_a"] / kwargs["r_i"], 0,
    #                arm_angle_0, 0])
    y0 = np.array([5.33249302e-01, 0, 5.33249314e-01, 0, 0.344213835078322, 0])

    # Solve ODE
    start = time.time()
    results = integrate.solve_ivp(
        fun=model.ode,
        t_span=(0, model.data["Time"].max()),
        y0=y0,
        t_eval=model.data["Time"],
        method='LSODA',
        jac=model.J,
    )
    print(f" Time taken: {time.time() - start}")
    print("Done")

    # Plot results
    f, ax = plt.subplots(figsize=(12, 4))
    ax.plot(model.data["Time"], model.data["Angle"], 'r', label="True Angle")
    ax.plot(model.data["Time"], results.y[4, :], 'b', label=f"Calculated Angle")
    ax.legend()
    ax.set(ylabel="Angle (Radians)", xlabel="Time (s)", title="Simulated Data with Global Optimization")

    # Delineate train and test results
    ax.axvline(40)
    ax.text(15, 0.2, "Training")
    ax.text(60, 0.2, "Testing")
    ax.set(ylim=(0.1, 2.295))

    print("The training loss is {}".format(
        np.sum((results.y[4, :4000] - model.data["Angle"].values[:4000]) ** 2)
    ))

    print("The testing loss is {}".format(
        np.sum((results.y[4, 4000:] - model.data["Angle"].values[4000:]) ** 2)
    ))
