import time
from scipy import integrate

import numpy as np
import matplotlib.pyplot as plt

from controller import PIDControlRobotArm

if __name__ == "__main__":

    data = PIDControlRobotArm.data
    kwargs = PIDControlRobotArm.optimized_params

    model = PIDControlRobotArm(**kwargs)

    # Initial conditions: where the displacement of each pulley is equal (0 net tension), and all 3 pulley are
    # stationary

    arm_angle_0 = data["Angle"][0]
    y0 = np.array([
        arm_angle_0 * kwargs["r_a"] / kwargs["r_m"], 0,
        arm_angle_0 * kwargs["r_a"] / kwargs["r_i"], 0,
        arm_angle_0, 0,
        0
    ])
    # y0 = np.array([5.33249302e-01, 0, 5.33249314e-01, 0, 0.344213835078322, 0])

    # Solve ODE
    start = time.time()
    results = integrate.solve_ivp(
        fun=model.ode,
        t_span=(0, 40),
        y0=y0,
        t_eval=data["Time"],
        method='LSODA',
        jac=model.J,
        # rtol=1E-2,
        # atol=1E-4,
    )
    print(f" Time taken: {time.time() - start}")
    print("Done")

    setpoint_f = model.setpoint
    # Plot results
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(data["Time"], data["Angle"], 'r', label="Empirical Data")
    axs[0].plot(data["Time"], results.y[4, :], 'b', label=f"Process Value")
    axs[0].plot(data["Time"], data["Time"].map(setpoint_f), label=f"Setpoint")
    axs[0].legend()
    axs[0].set(ylabel="Angle (Radians)", xlabel="Time (s)",
               title="PID Control on Simulated Robot Arm")

    velocity_scaler = len(data["Time"]) / data["Time"].iloc[-1]
    axs[1].plot(data["Time"], data["Angle"].diff() * velocity_scaler, 'r', label="Empirical Velocity")
    axs[1].plot(data["Time"], results.y[5, :], 'b', label="Process Value Velocity")
    axs[1].legend()
    axs[1].set(ylabel="Velocity (rad/s)", xlabel="Time (s)")

    axs[2].plot(data["Time"], data["Torque"], 'r', label="Empirical Torque")
    axs[2].plot(data["Time"], list(map(model.control,
        data["Time"], results.y[4], results.y[5], results.y[6]
    )), 'b', label="Simulated Torque")
    axs[2].legend()
    axs[2].set(ylabel="Torque (Nm)", xlabel="Time (s)")


    print("The loss is {}".format(
        np.sum((results.y[4, :] - data["Angle"].values) ** 2)
    ))
