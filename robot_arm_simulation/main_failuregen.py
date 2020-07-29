"""
Generate normal and failure data for the robot arm.
Visualize and compare failures.

Modified parameters:
    Base Tension (N)
    Stiffness EA (N)
    Friction f2a ()
    Slope slope1 (Nm)
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from controller import PIDControlRobotArm, PIDDisturbRobotArm


def f_torque_from_state(model):
    """ Returns a function that maps from a state vector to torque

    Args:
        model: robot arm dynamics model

    Returns:
        Callable: np.ndarray of shape (8,) -> float
        The (8,) vector consists is [time, theta_m, dtheta_m, theta_i, dtheta_i, theta_a, dtheta_a, E]

    """
    def func(arr):
        t = arr[0]
        y = arr[1:7]
        E = arr[7]
        return model.control(t, y[4], y[5], E)

    return func


def f_tension_from_state(model):
    """ Returns a function that maps from a state vector to tension

    Args:
        model: robot arm dynamics model

    Returns:
        Callable: np.ndarray of shape (7,) -> float
        The (8,) vector consists is [time, theta_m, dtheta_m, theta_i, dtheta_i, theta_a, dtheta_a, E]
    """
    def func(arr):
        y = arr[1:7]
        T_m_i, T_i_a, T_a_m = model.tensions(y)
        return T_a_m

    return func


def get_data(n_periods=10, **kwargs):
    """ Returns a simulated dataset

    Keyword Args:
        n_periods: int, the number of periods worth of data to return
        updates: dict, dictionary to update the kwargs with.

    Returns:
        A pd.DataFrame consisting of the columns ["time", "direction", "position", "torque", "tension"]

    """

    # Get robot arm parameters and modify them for failure
    params = PIDControlRobotArm.optimized_params.copy()
    params.update(kwargs)

    # Set up robot arm model
    # model = PIDControlRobotArm(K_p=30, T_i=0.8, T_d=0.1, **params)
    model = PIDDisturbRobotArm(K_p=30, T_i=0.8, T_d=0.1, sigma=0.01, l=0.01, **params)

    # Solve system
    t_eval = np.arange(0, 20 * n_periods, 0.01)
    results = solve_ivp(model.ode,
                        t_span=[0, 20 * n_periods],
                        y0=model.y0,
                        method="LSODA",
                        t_eval=t_eval,
                        jac=model.J,
                        )
    state = np.vstack([results.t, results.y])

    data = {
        "time": t_eval % 20,
        "direction": np.tile(np.repeat([0, 1], 1000), n_periods),
        "angle": results.y[4],
        "torque": np.apply_along_axis(
            f_torque_from_state(model),
            arr=state,
            axis=0
        ),
        "tension": np.apply_along_axis(
            f_tension_from_state(model),
            arr=state,
            axis=0
        ),
    }

    return pd.DataFrame(data)


def _eval_base_T(base_T):
    """ see sweep_tensions()"""
    return get_data(n_periods=2, base_T=base_T)


def sweep_tensions():
    pool = Pool()

    base_tensions = np.arange(100, 200, 10)
    results = pool.imap(_eval_base_T, base_tensions)

    data_tensions = {}
    for tension, res in zip(base_tensions, results):
        data_tensions[tension] = res

    pool.close()
    pool.join()

    return data_tensions


def _eval_stiffness(EA):
    """ see sweep_stiffness """
    return get_data(n_periods=2, EA=EA)


def sweep_stiffness():
    """ Evaluate a number of different stiffnesses (EA).

    Returns:
        dict() of datasets, one for each stiffness
            key: the stiffness (EA)
            value: a pd.DataFrame of operational data,
                from get_data()
    """
    pool = Pool()

    stiffnesses = np.linspace(10000, 200000, 10)
    results = pool.imap(_eval_stiffness, stiffnesses)

    data_stiffnesses = {}
    for EA, res in zip(stiffnesses, results):
        data_stiffnesses[EA] = res

    pool.close()
    pool.join()

    return data_stiffnesses


def _eval_f1a(f2a):
    """ See sweep_f2a() """
    return get_data(n_periods=2, f2a=f2a)


def sweep_f2a():
    """ Evaluate a number of different frictions (f2a).

    Returns:
        dict() of datasets, one for each stiffness
            key: the friction (f2a)
            value: a pd.DataFrame of operational data,
                from get_data()
    """
    pool = Pool()

    frictions = np.linspace(0.00007716518208965043 / 1.44 / 2, 0.00007716518208965043 / 1.44 * 10, 10)
    results = pool.imap(_eval_f1a, frictions)

    data_frictions = {}
    for f2a, res in zip(frictions, results):
        data_frictions[f2a] = res

    pool.close()
    pool.join()

    return data_frictions


def _eval_slope(slope1):
    """ see sweep_slope() """
    return get_data(n_periods=2, slope1=slope1)


def sweep_slope():
    """ Evaluate a number of different slopes (slope1).

        Returns:
            dict() of datasets, one for each stiffness
                key: the friction (slope1)
                value: a pd.DataFrame of operational data,
                    from get_data()
        """
    pool = Pool()

    slopes = np.linspace(-0.23723147775251083 * 2, 0, 10)
    results = pool.imap(_eval_slope, slopes)

    data_slopes = {}
    for slope, res in zip(slopes, results):
        data_slopes[slope] = res

    pool.close()
    pool.join()

    return data_slopes


def plot_sweep(data_dict: dict, param="Parameter Name",
               nominal_val=None, optimal_val=None):
    """ Takes a dictionary of different
    responses """

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib as mpl

    norm = mpl.colors.Normalize(vmin=min(data_dict.keys()),
                                vmax=max(data_dict.keys()))
    cmap = cm.hot

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(3, 2, width_ratios=[10, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    axcm = fig.add_subplot(gs[:, 1])

    for t, df in data_dict.items():
        ax1.plot(df.index / 2000 * 20, df["angle"], c=m.to_rgba(t))
        ax2.plot(df.index / 2000 * 20, df["torque"], c=m.to_rgba(t))
        ax3.plot(df.index / 2000 * 20, df["tension"], c=m.to_rgba(t))

    ax1.set(ylabel="Angle (radians)")
    ax2.set(ylabel="Torque (Nm)")
    ax3.set(ylabel="Tension (N)", xlabel="Time (s)")

    plt.colorbar(m, cax=axcm)
    if (nominal_val is not None) and (optimal_val is not None):
        axcm.axhline(nominal_val, ls="--", c="blue", lw=2)
        axcm.axhline(optimal_val, c="blue", lw=2)
    axcm.set(title=param)

    plt.show()
    plt.tight_layout()
    # plt.savefig(f"failure_{param}.png")


if __name__ == "__main__":
    normal_data = get_data()
    print(normal_data)

    data_tensions = sweep_tensions()
    plot_sweep(data_tensions, "Base Tension (N)",
                   nominal_val=PIDControlRobotArm.base_T,
                   optimal_val=PIDControlRobotArm.optimized_params["base_T"])

    data_stiffnesses = sweep_stiffness()
    plot_sweep(data_stiffnesses, "Stiffness EA (N)",
               nominal_val=PIDControlRobotArm.EA,
               optimal_val=PIDControlRobotArm.optimized_params["EA"])

    data_frictions = sweep_f2a()
    plot_sweep(data_frictions, "f2a (unitless)",
               nominal_val=PIDControlRobotArm.f2a,
               optimal_val=PIDControlRobotArm.optimized_params["f2a"])

    data_slope = sweep_slope()
    plot_sweep(data_slope, "slope (Nm)",
               nominal_val=PIDControlRobotArm.slope1,
               optimal_val=PIDControlRobotArm.optimized_params["slope1"])
