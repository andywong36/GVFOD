""" Evaluate training size needs of different algorithms"""
import os
import time

from functools import partial
from multiprocessing import pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from markov.markov import Markov
from GVFOD.gvfod.RLOD import RLOD

import data.dataloader as dtl
import data.model_selection as ms

arm_period = 10  # 10 seconds per sweep

common_params = {"contamination": 0.05}
algorithms = {
    "Isolation Forest": IForest(n_estimators=150, max_features=1., bootstrap=False, **common_params),
    "One-class SVM": OCSVM(nu=0.05, gamma="scale", kernel="rbf", **common_params),
    "Local Outlier Factor": LOF(**common_params),
    "Angle-based OD": ABOD(n_neighbors=18, **common_params),
    "Histogram-based OD": HBOS(n_bins=10, alpha=0.1, tol=0.5, **common_params),
    "Principle Components": PCA(n_components=9, **common_params),
    "Markov Chain": Markov(n_sensors=3, divisions=6, resample=True, sample_period=20, **common_params),
    "Reinforcement-Learning Outlier Detection": RLOD(n_sensors=3, divisions=[4] * 3, wrap_idxs=None, int_idxs=None,
                                                     numtilings=32, state_size=2048, discount_rate=0.985,
                                                     learn_rate=0.005, lamda=0.25, beta=200, **common_params)
}
uses_pca = [
    "Histogram-based OD",
    "Local Outlier Factor",
    "Isolation Forest",
    "One-class SVM"
]


# assert all(item in algorithms.keys() for item in uses_pca)


def test():
    """ Main testing function """
    os.nice(10)
    start = time.time()
    save_directory = "scratch/train_size_test_raw_results/"

    X, y, class_labels = dtl.get_robot_arm_data(return_class_labels=True)
    print(class_labels)
    X_nor, y_nor = X[y == 0], y[y == 0]
    X_abn, y_abn = X[y != 0], y[y != 0]

    # Rescale
    ss = StandardScaler()
    X_nor = ss.fit_transform(X_nor)
    X_abn = ss.transform(X_abn)

    n_experiments = 20  # Training data starts are offset between experiments
    n_sweeps = 3000  # Amount of normal data per experiment, includes both training and testing sizes
    # hours_of_data = n_sweeps * arm_period / 3600

    # Starts every 30 minutes
    step = 30 * 60 / arm_period
    experiment_starts = np.arange(0, n_experiments * step, step, dtype=int)

    # Results are stored in results. Results is a dictionary, with keys that are outlier names, and values that are
    # DataFrames. The rows of the DataFrames are training sizes. The columns of the DataFrames are the outlier detection
    # algorithms. Each element in the DataFrames is a List[float] of scores (not floats, so that statistical
    # significance can be determined).
    results = {outlier: pd.DataFrame(columns=algorithms.keys())
               for outlier in class_labels if outlier != "Normal"}

    splitter = ms.TimeSeriesFolds(n_splits=10, min_test_size=1000)
    dc = decomposition.PCA(n_components=20)

    p = pool.Pool(processes=32)
    all_exp_results = []

    for exp_start in experiment_starts:
        assert exp_start + n_sweeps <= len(X_nor)  # check that there is enough data
        X_nor_exp, y_nor_exp = X_nor[exp_start:exp_start + n_sweeps, :], y_nor[exp_start: exp_start + n_sweeps]
        exp_results_holder = []  # List of dictionaries. Each dictionary has 2 keys: "train_size", "performances".
        for i, (train_idx, test_idx) in enumerate(splitter.split(X_nor_exp)):
            train_size = len(train_idx)
            func = partial(process,
                           X_nor_exp=X_nor_exp,
                           y_nor_exp=y_nor_exp,
                           X_abn=X_abn,
                           y_abn=y_abn,
                           train_idx=train_idx,
                           test_idx=test_idx,
                           class_labels=class_labels,
                           dc=dc,
                           results_storage=save_directory +
                                           "_".join([f"ExpStart{exp_start}",
                                                     f"TrnSz{train_size}",
                                                     f"AlgALGORITHM",
                                                     f"OtlOUTLIER"
                                                     ]) +
                                           ".npy"
                           )
            # for model in algorithms:
            #     performance = func(model)
            performances = p.imap(func, algorithms.keys())
            exp_results_holder.append({"train_size": train_size, "performances": performances})
        all_exp_results.append(exp_results_holder)

    for exp_results_holder in all_exp_results:
        for dictionary in exp_results_holder:
            train_size = dictionary["train_size"]
            performances = dictionary["performances"]
            for model, performance in zip(algorithms.keys(), list(performances)):
                for outlier_class in performance:
                    # Save the results
                    score = performance[outlier_class]
                    dataframe = results[outlier_class]
                    try:
                        if np.any(np.isnan(dataframe.loc[train_size, model])):
                            dataframe.loc[train_size, model] = [score]
                        else:
                            dataframe.loc[train_size, model].append(score)
                    except KeyError:
                        dataframe.loc[train_size, model] = [score]

    for outlier in results:
        results[outlier].to_pickle(f"exp_gvfod/trainsize_{outlier}_results.df")

    print(f"Time elapsed: {time.time() - start}")


def process(model, X_nor_exp, y_nor_exp, X_abn, y_abn, train_idx, test_idx, class_labels, dc, results_storage=None):
    """ Evaluate a single outlier detection algorithm, on a single set of data
    Args:
        model (str): key in 'algorithms'. PyOD (or similar) outlier detection model. Ex: 'Isolation Forest'
        X_nor_exp: Normal X data
        y_nor_exp: Normal classes (all 0)
        X_abn: Abnormal X data
        y_abn: Abnormal classes
        train_idx: Indices of normal data to train
        test_idx: Indices of normal data to test
        class_labels (List[str]): contains names of outlier classes. class_labels[0] is 'normal'
        dc: decomposition algorithm, typically decomposition.PCA
        results_storage: the path to store the raw results. None if to be discarded

    Returns:
        The score (F1, recall, etc.) for each outlier class in the data

    """
    process_return = dict()

    if model in uses_pca:
        X_nor_exp_train = dc.fit_transform(X_nor_exp[train_idx])
        X_nor_exp_test = dc.transform(X_nor_exp[test_idx])
        X_abn_exp_test = dc.transform(X_abn)
    else:
        X_nor_exp_train = X_nor_exp[train_idx]
        X_nor_exp_test = X_nor_exp[test_idx]
        X_abn_exp_test = X_abn

    # train the model
    algorithms[model].fit(X_nor_exp_train)

    # evaluate the model
    X_test = np.concatenate((X_nor_exp_test, X_abn_exp_test), axis=0)
    y_pred = algorithms[model].predict(X_test)
    y_test = np.concatenate((y_nor_exp[test_idx], y_abn), axis=0)

    for outlier_class in class_labels:
        # Calculate the metrics for each outlier class separately

        if class_labels.index(outlier_class) == 0:
            # Skip normal class
            continue
        # evaluate the model on outlier classes separately
        outlier_idx = class_labels.index(outlier_class)
        selection = (y_test == 0) | (y_test == outlier_idx)  # indices of normal and abnormal values

        score = f1_score(y_test[selection].astype(bool), y_pred[selection])
        if results_storage:
            file_loc = (results_storage.replace("ALGORITHM", model)).replace("OUTLIER", outlier_class)
            raw_results = np.c_[y_test[selection], y_pred[selection]]

            np.save(file_loc, raw_results.astype(np.int8))

        process_return[outlier_class] = score

    return process_return


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


def plot(folder="scratch/feb2020_raw_results_trainsizetest/"):
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
    # test()
    plot()
