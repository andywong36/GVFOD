""" Runs a single iteration of the robot arm IVP, and plots the resulting trajectory. """

import matplotlib.pyplot as plt

from system_id import *

if __name__ == "__main__":
    args_are_lists = True
    kwargs = {
        "C_L": [5.599170154855655],
        "EA": [4602.08225127837],
        "I_a": [0.7999649787833675],
        "I_i": [7.550422916189281e-7],
        "J_m": [0.000021496744412963013],
        "base_T": [118.2741167640548],
        "f1a": [0.0005229938899602387],
        "f1i": [0.000012954392747939966],
        "f1m": [0.0000014591673085946632],
        "f2a": [0.000013725154787334402],
        "f2i": [0.0002203862711822767],
        "f2m": [0.00006320040341725074],
        "l_1": [0.12467394598102567],
        "l_2": [0.15675298814399916],
        "l_3": [0.16503183794943505],
        "r_a": [0.019284035139033706],
        "r_i": [0.01080913186452781],
        "r_m": [0.01499828260649182],
        "slope1": [-0.23709209894007333 * 1.05],
        "slope2": [0.20813318208560924]
    }
    if args_are_lists:
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
        # rtol=1E-2,
        # atol=1E-4,
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
