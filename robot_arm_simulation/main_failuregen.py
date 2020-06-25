"""
Generate normal and failure data for the robot arm
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from controller import PIDControlRobotArm


def f_torque_from_state(model):
    def func(arr):
        t = arr[0]
        y = arr[1:7]
        E = arr[7]
        return model.control(t, y[4], y[5], E)

    return func


def f_tension_from_state(model):
    def func(arr):
        y = arr[1:7]
        T_m_i, T_i_a, T_a_m = model.tensions(y)
        return T_a_m

    return func


def get_data(n_periods=10, **kwargs):
    """

    Keyword Args:
        n_periods: int, the number of periods worth of data to return
        updates: dict, dictionary to update the kwargs with.

    Returns:
        A pd.DataFrame consisting of the columns ["time", "direction", "position", "torque", "tension"]

    """

    params = PIDControlRobotArm.optimized_params.copy()
    params.update(kwargs)
    model = PIDControlRobotArm(K_p=30, T_i=0.8, T_d=0.1, **params)

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
    return get_data(n_periods=2, base_T=base_T)


def sweep_tensions():
    pool = Pool()

    base_tensions = np.arange(100, 200, 5)
    results = pool.imap(_eval_base_T, base_tensions)

    data_tensions = {}
    for tension, res in zip(base_tensions, results):
        data_tensions[tension] = res

    pool.close()
    pool.join()

    return data_tensions

def _eval_stiffness(EA):
    return get_data(n_periods=2, EA=EA)


def sweep_stiffness():
    pool = Pool()

    stiffnesses = np.linspace(10000, 200000, 20)
    results = pool.imap(_eval_stiffness, stiffnesses)

    data_stiffnesses = {}
    for EA, res in zip(stiffnesses, results):
        data_stiffnesses[EA] = res

    pool.close()
    pool.join()

    return data_stiffnesses


def plot_sweep(data_dict: dict, param="Parameter Name"):
    """ Takes a dictionary of different
    responses """

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib as mpl

    norm = mpl.colors.Normalize(vmin=min(data_dict.keys()),
                                vmax=max(data_dict.keys()))
    cmap = cm.hot

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure()
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
    axcm.set(title=param)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()
    plt.tight_layout()



if __name__ == "__main__":
    normal_data = get_data()
    print(normal_data)

    # data_tensions = sweep_tensions()
    # plot_sweep(data_tensions, "Base Tension (N)")

    data_stiffnesses = sweep_stiffness()
    plot_sweep(data_stiffnesses, "Stiffness EA (N)")
