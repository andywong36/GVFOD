""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """
import time

import matplotlib.pyplot as plt
from scipy import integrate

from data.dataloader import get_robotarm_sim_data

from system_id import *

if __name__ == "__main__":
    kwargs = RobotArmDynamics.optimized_params
    # kwargs = {}
    model = RobotArmDynamics(
        replacement_data=get_robotarm_sim_data("normal", 500, 4),
        **kwargs
    )

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
    f, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(model.data["Time"], model.data["Torque"], 'r', label="Input Torque")
    axs[0].legend()
    axs[0].set(ylabel="Torque (Nm)", title="Simulator Input")

    axs[1].plot(model.data["Time"], model.data["Angle"], 'r', label="True Angle")
    axs[1].plot(model.data["Time"], results.y[4, :], 'b', label=f"Calculated Angle")
    axs[1].legend()
    axs[1].set(ylabel="Angle (Radians)", xlabel="Time (s)", title="Simulator Output vs. Real Output")

    # Delineate train and test results
    axs[1].axvline(model.period * 2)
    axs[1].text(7.5, 0.15, "Training")
    axs[1].text(32.5, 0.15, "Testing")
    # axs[1].set(ylim=(0.1, 2.295))
    # axs[1].set(ylim=(0.1, 10))

    plt.tight_layout()

    print("The training loss is {}".format(
        np.sum((results.y[4, :4000] - model.data["Angle"].values[:4000]) ** 2)
    ))

    print("The first testing loss is {}".format(
        np.sum((results.y[4, 4000:8000] - model.data["Angle"].values[4000:8000]) ** 2)
    ))

    print("The second testing loss is {}".format(
        np.sum((results.y[4, 8000:] - model.data["Angle"].values[8000:]) ** 2)
    ))
