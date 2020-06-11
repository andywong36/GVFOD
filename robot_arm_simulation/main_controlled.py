import matplotlib.pyplot as plt

from system_id import *


if __name__ == "__main__":

    kwargs = SystemProperties.optimized_params

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
        # rtol=1E-2,
        # atol=1E-4,
    )
    print(f" Time taken: {time.time() - start}")
    print("Done")

    setpoint_f = PIDController(K_p).setpoint
    # Plot results
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data["Time"], data["Angle"], 'r', label="Empirical Data")
    axs[0].plot(data["Time"], results.y[4, :], 'b', label=f"Process Value")
    axs[0].plot(data["Time"], data["Time"].map(setpoint_f), label=f"Setpoint")
    axs[0].legend()
    axs[0].set(ylabel="Angle (Radians)", xlabel="Time(s)", title="Simulated Data with Global Optimization")

    axs[1].plot(data["Time"], data["Angle"].diff(), 'r', label="Empirical Velocity")

    print("The loss is {}".format(
        np.sum((results.y[4, :] - data["Angle"].values) ** 2)
    ))