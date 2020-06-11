""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """

import matplotlib.pyplot as plt

from system_id import *

if __name__ == "__main__":
    kwargs = SystemProperties.optimized_params

    for key, val in kwargs.items():
        kwargs[key] = val[0]
    ode = ode_const(**kwargs)

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
        fun=ode,
        t_span=(0, 40),
        y0=y0,
        t_eval=data["Time"],
        method='LSODA',
        jac=ode_const(jac=True, **kwargs),
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

    print("The loss is {}".format(
        np.sum((results.y[4, :] - data["Angle"].values) ** 2)
    ))
