""" Evaluate training size needs of different algorithms"""
import os
import psutil
import time

from multiprocessing import pool, RawArray

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

import data.dataloader as dtl
import data.model_selection as ms
from exp_gvfod.exp_train_size_settings import *

arm_period = 10  # 10 seconds per sweep

global_data = {}


def init_data(X_nor, X_nor_scaled, y_nor, X_nor_shape,
              X_abn, X_abn_scaled, y_abn, X_abn_shape):
    global_data["X_nor"] = X_nor
    global_data["X_nor_scaled"] = X_nor_scaled
    global_data["y_nor"] = y_nor
    global_data["X_nor_shape"] = X_nor_shape
    global_data["X_abn"] = X_abn
    global_data["X_abn_scaled"] = X_abn_scaled
    global_data["y_abn"] = y_abn
    global_data["X_abn_shape"] = X_abn_shape


def test(file, scaling=True):
    """ Main testing function """

    n_experiments = 20  # Training data starts are offset between experiments
    # n_experiments = 2
    minutes_between_exps = 15 # The time between each experiment start
    # n_sweeps = 3000  # Amount of normal data per experiment, includes both training and testing sizes
    n_sweeps = 3720
    n_splits = 20
    # n_splits = 2
    min_train_size = 50
    min_test_size = 1000
    max_test_size = 1000
    delay = 720  # 720 is 2 hours
    # hours_of_data = n_sweeps * arm_period / 3600

    if os.name == "posix":
        os.nice(10)
    else:
        p = psutil.Process(os.getpid())
        p.nice(psutil.IDLE_PRIORITY_CLASS)

    start = time.time()
    save_directory = "scratch/train_size_test_raw_results/"

    ## Read all the data and store it in global memory
    X, y, class_labels = dtl.get_robot_arm_data(return_class_labels=True)

    X_nor, y_nor = X[y == 0], y[y == 0]
    X_abn, y_abn = X[y != 0], y[y != 0]

    # Scale the data
    ss = StandardScaler()
    X_nor_ss = ss.fit_transform(X_nor)
    X_abn_ss = ss.transform(X_abn)

    # Save the data into a RawArray
    def raw_array_from_ndarray(arr):
        if arr.dtype == np.float64:
            raw = RawArray('d', arr.size)
            _tempnparray = np.frombuffer(raw, np.float64).reshape(arr.shape)
        elif arr.dtype == np.int64 or arr.dtype == np.int32:
            raw = RawArray('i', arr.size)
            _tempnparray = np.frombuffer(raw, np.int32).reshape(arr.shape)
        else:
            raise TypeError(f"Unknown numpy dtype: {arr.dtype}")
        np.copyto(_tempnparray, arr)
        return raw

    X_nor_r = raw_array_from_ndarray(X_nor)
    X_nor_ss_r = raw_array_from_ndarray(X_nor_ss)
    y_nor_r = raw_array_from_ndarray(y_nor)
    X_abn_r = raw_array_from_ndarray(X_abn)
    X_abn_ss_r = raw_array_from_ndarray(X_abn_ss)
    y_abn_r = raw_array_from_ndarray(y_abn)

    p = pool.Pool(processes=8, initializer=init_data, initargs=[X_nor_r, X_nor_ss_r, y_nor_r, X_nor.shape,
                                                                 X_abn_r, X_abn_ss_r, y_abn_r, X_abn.shape])
    jobs = []  # Contains the returns from pool.apply_async. Iterate through these to get the Series.

    # Plan the experiments
    # The manipulated variables are:
    #   Algorithm
    #   Time of start
    #   Training Size
    # The return should be:
    #   (c_{class}, ic_{class} for class in [normal, abn1, abn2, ...])
    #   where c_{class} is the number of correct classifications of a sample whose true class is class
    #     and ic{class} is the number of incorrect classifications

    # The results dataframe is defined as:
    columns = [("Algorithm", str),
               ("Time of Start", int),
               ("Training Size", int),
               ("Testing Size", int),
               ("Delay", int)]
    for data_class in class_labels:
        columns.append((f"c_{data_class}", int))
        columns.append((f"ic_{data_class}", int))
    results_df = pd.DataFrame({name: pd.Series([], dtype=dtype) for name, dtype in columns})

    # Calculate start times
    # Starts every 30 minutes
    step = minutes_between_exps * 60 / arm_period
    experiment_starts = np.arange(0, n_experiments * step, step, dtype=int)

    splitter = ms.TimeSeriesFolds(n_splits=n_splits,
                                  min_train_size=min_train_size,
                                  min_test_size=min_test_size,
                                  max_test_size=max_test_size,
                                  delay=delay)

    for exp_start in reversed(experiment_starts):
        assert exp_start + n_sweeps < len(X_nor)  # check that there is enough data
        for i, (train_idx, test_idx) in enumerate(splitter.split(X_nor[exp_start:exp_start + n_sweeps])):
            for exp in [if_exp, ocsvm_exp, lof_exp, abod_exp, kNN_exp, hbos_exp, mcd_exp, pca_exp,
                        markov_exp, gvfod_exp]:
                jobs.append(
                    p.apply_async(
                        process,
                        args=(exp, train_idx + exp_start, test_idx + exp_start, class_labels)
                    )
                )

    p.close()

    for i, job in enumerate(tqdm(jobs)):
        results_df.loc[i] = job.get()

    p.join()
    results_df.to_json(file)

    print(f"Time elapsed: {time.time() - start}")


def process(experiment, train_idx, test_idx, class_labels):
    """ Evaluate a single outlier detection algorithm, on a single set of data
    Args:
        experiment (Experiment): namedtuple with fields ["clf", "clfname", "use_pca", "use_scaling", "kwargs"]
        train_idx (np.ndarray): Indices of normal data to train
        test_idx (np.ndarray): Indices of normal data to test
        class_labels (List[str]): contains names of outlier classes. class_labels[0] is 'normal'

    Returns:
        dictionary with fields "Algorithm", "Time of Start", "Training Size", and the correct/incorrect classifications
        for each sample class
    """
    _return_dict = dict()
    _return_dict["Algorithm"] = experiment.clfname
    _return_dict["Time of Start"] = train_idx[0]
    _return_dict["Training Size"] = len(train_idx)
    _return_dict["Testing Size"] = len(test_idx)
    _return_dict["Delay"] = test_idx[0] - train_idx[-1] - 1

    # Get the correct data
    if experiment.use_scaling:
        X_nor = np.frombuffer(global_data["X_nor_scaled"]).reshape(global_data["X_nor_shape"])
        X_abn = np.frombuffer(global_data["X_abn_scaled"]).reshape(global_data["X_abn_shape"])
    else:
        X_nor = np.frombuffer(global_data["X_nor"]).reshape(global_data["X_nor_shape"])
        X_abn = np.frombuffer(global_data["X_abn"]).reshape(global_data["X_abn_shape"])
    y_nor = np.frombuffer(global_data["y_nor"], dtype=np.int32).reshape(global_data["X_nor_shape"][0])
    y_abn = np.frombuffer(global_data["y_abn"], dtype=np.int32).reshape(global_data["X_abn_shape"][0])

    # Preprocess with PCA
    if experiment.use_pca:
        dc = decomposition.PCA(n_components=20)
        X_nor_train = dc.fit_transform(X_nor[train_idx])
        X_nor_test = dc.transform(X_nor[test_idx])
        X_abn = dc.transform(X_abn)
    else:
        X_nor_train = X_nor[train_idx]
        X_nor_test = X_nor[test_idx]
        X_abn = X_abn
    y_nor_test = y_nor[test_idx]
    y_abn = y_abn

    # Train the model
    # Check that n_neighbors is sufficiently small
    kwargs = experiment.kwargs.copy()
    if "n_neighbors" in experiment.kwargs:
        kwargs["n_neighbors"] = min(_return_dict["Training Size"] - 1, kwargs["n_neighbors"])
    model = experiment.clf(**kwargs)
    model.fit(X_nor_train)

    # Make predictions
    X_test = np.concatenate((X_nor_test, X_abn), axis=0)
    y_pred = model.predict(X_test)
    y_test = np.concatenate((y_nor_test, y_abn), axis=0)

    for cls_idx, cls_name in enumerate(class_labels):
        # Calculate the metrics for each outlier class separately
        # The number samples in cls_idx that was classified correctly
        _return_dict[f"c_{cls_name}"] = np.sum(y_pred[y_test == cls_idx] == bool(cls_idx))
        # The number of samples in cls_idx that was classified incorrectly
        _return_dict[f"ic_{cls_name}"] = np.sum(y_pred[y_test == cls_idx] != bool(cls_idx))

    return _return_dict


if __name__ == '__main__':
    test("exp_gvfod/exp_train_size_results.json")
    # plot("scratch/train_size_test_raw_results/")
