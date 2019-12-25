# Functions for loading data in a useful way from cached versions (pickles)
import numpy as np
import pandas as pd


def get_robot_arm_data(return_class_labels=False):
    """
    Returns multiple sensor data for outlier detection

    Returns:
        Tuple(X: np.ndarray, y: np.ndarray): X contains an array of both normal and abnormal data, where each row has
            length n * k, where n is the length of each time series, and k is the number of sensors. y is the outlier
            label (int)
    """

    # Normal data
    n_dat = pd.read_pickle("data/pickles/robotarm_normal.pkl")
    sensors = list(n_dat.columns)

    # Abnormal data
    a_list = ['loose_l1',
              'loose_l2',
              'tight',
              'sandy',
              'highT']
    a_dat = [pd.read_pickle("data/pickles/robotarm_{}.pkl".format(s)) for s in a_list]

    # Get sensors common to every dataset
    sensors.remove("time")
    sensors.remove("direction")
    for a in a_dat:
        for sensor in sensors:
            if sensor not in a.columns:
                sensors.remove(sensor)

    # print("The sensors common to every dataset are {}.".format(sensors))

    # Create dataset X
    abn_list = [format_multisensor_data(df, sensors, 2000) for df in a_dat]
    nor = format_multisensor_data(n_dat, sensors, 2000)
    abn = np.concatenate(abn_list, axis=0)

    # Create labels y
    y_list = [np.zeros(len(nor), dtype=int)]
    for i, label in enumerate(a_list):
        # normal: [0] * len(nor), a_list[1]: [1] * len(abn_list[0]), etc.
        y_list.append(np.ones(len(abn_list[i]), dtype=int) + i)

    X = np.concatenate([nor, abn], axis=0)
    y = np.concatenate(y_list)
    assert X.shape[0] == y.shape[0]

    if return_class_labels:
        label_classes = a_list
        a_list.insert(0, "Normal")
        return X, y, a_list
    else:
        return X, y


def format_multisensor_data(df, columns, period):
    """ Arranges data for use in outlier detection

    For each sensor reading from columns, reformat the data as a rectangular matrix of n_runs * period. Concatenate
    these matrices such that the final return is n_runs * (period * n_sensors)

    Args:
        df (pd.DataFrame): input dataframe, with n_runs * period rows, where each column is a sensor
        columns: sensors/columns to use
        period: period, for grouping sequential sensor readings as a single observation

    Returns:
        np.ndarray: the resulting array

    """
    dfs_to_concatenate = []
    for sensor in columns:
        dfs_to_concatenate.append(df.loc[:, sensor].values.reshape(-1, period))
    return np.concatenate(dfs_to_concatenate, axis=1)


def get_machine_torque_data():
    n_dat = pd.read_pickle("data/pickles/machine_normal.pkl")
    a_list = [
        'misaligned',
        'install_1',
        'install_2',
        'install_3',
        'install_4',
        'screw',
        'loose',
        'tight',
        'noise_power',
        'noise_ground',
        'noise_signal',
    ]
    a_dat = [pd.read_pickle("data/pickles/machine_{}.pkl".format(s)) for s in a_list]
    n_tor = n_dat["torque_master"].values.reshape(-1, 256).astype(np.int)
    a_tor = [df["torque_master"].values.reshape(-1, 256).astype(np.int) for df in a_dat]

    generalize_factor = 2

    np.floor_divide(n_tor, generalize_factor, out=n_tor)
    for a in a_tor:
        np.floor_divide(a, generalize_factor, out=a)

    return n_tor, zip(a_list, a_tor)


def get_robotarm_torque_data():
    n_dat = pd.read_pickle("data/pickles/robotarm_normal.pkl")
    a_list = ['loose_l1',
              'loose_l2',
              'tight',
              'sandy',
              'highT']
    a_dat = [pd.read_pickle("data/pickles/robotarm_{}.pkl".format(s)) for s in a_list]

    n_tor = n_dat["torque"].values.reshape(-1, 2000)
    a_tor = [df["torque"].values.reshape(-1, 2000) for df in a_dat]

    return n_tor, zip(a_list, a_tor)
