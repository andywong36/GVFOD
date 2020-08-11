""" Evaluate training size needs of different algorithms"""
import os
import time

from functools import partial
from multiprocessing import pool, RawArray

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score
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
    n_sweeps = 3000  # Amount of normal data per experiment, includes both training and testing sizes
    n_splits = 10
    # hours_of_data = n_sweeps * arm_period / 3600

    os.nice(10)
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
        elif arr.dtype == np.int64:
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

    p = pool.Pool(processes=12, initializer=init_data, initargs=[X_nor_r, X_nor_ss_r, y_nor_r, X_nor.shape,
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
               ("Training Size", int)]
    for data_class in class_labels:
        columns.append((f"c_{data_class}", int))
        columns.append((f"ic_{data_class}", int))
    results_df = pd.DataFrame({name: pd.Series([], dtype=dtype) for name, dtype in columns})

    # Calculate start times
    # Starts every 30 minutes
    step = 30 * 60 / arm_period
    experiment_starts = np.arange(0, n_experiments * step, step, dtype=int)

    splitter = ms.TimeSeriesFolds(n_splits=n_splits, min_train_size=600, min_test_size=1000, max_test_size=1000)

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
    model = experiment.clf(**experiment.kwargs)
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


def plot_results(filepath, linelabel=True):
    from itertools import cycle

    res = pd.read_pickle(filepath)
    res = res.sort_index()
    # title = "Outlier type: " + str.join("_", filepath.split("_")[2:-1])
    title = "Comparison of Outlier Detection Algorithms According to Training Data Requirements"

    lines = ["-", "--", "-."]
    linecycler = cycle(lines)

    for algname in res:
        x = []
        y = []
        yerr = []
        for trainsize, scores in zip(res.index, res[algname]):
            x.append(trainsize)
            y.append(np.mean(scores))
            yerr.append(np.std(scores) / np.sqrt(len(scores)))
        x = np.array(x) * arm_period / 60
        y = np.array(y)
        yerr = np.array(yerr)
        if linelabel:
            plt.errorbar(x, y, yerr=yerr, ls=next(linecycler), label=algname)
        else:
            plt.errorbar(x, y, yerr=yerr, ls=next(linecycler))
        plt.legend()
        plt.ylabel("F1-score")
        plt.xlabel("Training Data (minutes of operation)")
        plt.title(title)


def _plotall():
    for file in ["highT", "loose_l1", "loose_l2", "sandy", "tight"]:
        plt.figure(figsize=(16, 9))
        plot_results("exp_gvfod/trainsize_" + file + "_results.df")
        plt.savefig("exp_gvfod/f1/plot" + file + ".png")
    plt.show()


def plot(folder):
    """
    Goes into the scratch directory and generates plots for each algorithm
    Returns:

    """
    import re

    files = os.listdir(folder)

    # For each training size, we need metrics for each algorithm
    trnsz_regex = r"_TrnSz(\d+)_"
    training_sizes = set()
    for file in files:
        matches = re.search(trnsz_regex, file)
        training_sizes.add(int(matches[1]))
    training_sizes = sorted(training_sizes)

    # Each algorithm has its own metrics
    alg_regex = r"_Alg(.+)_Otl"
    alg_names = set()
    for file in files:
        matches = re.search(alg_regex, file)
        alg_names.add(matches[1])
    alg_names = sorted(alg_names)
    if set(alg_names) == set(algorithms):
        alg_names = list(algorithms)  # Use the order of the keys in algorithms

    # For statistical significance, there are outlier types and training start
    otl_regex = r"_Otl(.+)\.npy"
    start_regex = r"ExpStart(\d+)_"
    otl_names = set()
    expstarts = set()
    for file in files:
        matches = re.search(otl_regex, file)
        otl_names.add(matches[1])
        matches = re.search(start_regex, file)
        expstarts.add(matches[1])

    columns = ["TrainSize", "StartTime", "Algorithm", "Algorithm Class", "Outlier", "Precision", "Recall", "F1"]
    data = pd.DataFrame(columns=columns)

    for file in files:
        path = folder + file
        trnsz = int(re.search(trnsz_regex, file)[1])
        start = int(re.search(start_regex, file)[1])
        alg = re.search(alg_regex, file)[1]
        algclass = "RL-based" if alg in ["Markov Chain", "Reinforcement-Learning Outlier Detection"] else "Traditional"
        otl = re.search(otl_regex, file)[1]

        y_arr = np.load(path)
        y_test, y_pred = y_arr[:, 0].astype(bool), y_arr[:, 1].astype(bool)

        p, r, f1 = precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)

        data = data.append(
            {key: value for key, value in zip(columns, [trnsz, start, alg, algclass, otl, p, r, f1])},
            ignore_index=True
        )

    data.to_pickle("scratch/summarized_results.df")

    for outlier in sorted(data["Outlier"].unique()):
        for metric in ["F1",
                       "Recall",
                       "Precision"
                       ]:
            plt.figure(figsize=(9.5 * 1.25, 6.5 * 1.25))
            sns.lineplot(
                x='TrainSize',
                y=metric,
                hue='Algorithm',
                style='Algorithm Class',
                ci="sd",
                data=data.loc[data["Outlier"] == outlier]
            )
            headers = ["Algorithm", "Algorithm Class"]
            for line in plt.legend().get_texts():
                if line.get_text() in headers:
                    line.set_text(r"$\mathbf{" + line.get_text().split(" ")[-1] + "}$")

            plt.xlim(200, 2000)
            plt.xlabel("Training Samples")
            plt.ylim(0, 1)
            textheight = 0.05 if metric != "Recall" else 0.50
            plt.text(0.80, textheight, f"Outlier type: {outlier}", transform=plt.gca().transAxes, fontsize=14,
                     verticalalignment='top')

            plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    test("exp_gvfod/exp_train_size_results.json")
    # plot("scratch/train_size_test_raw_results/")
