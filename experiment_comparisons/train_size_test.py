""" Evaluate training size needs of different algorithms"""
import os
import time

from functools import partial
from multiprocessing import pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from markov.markov import Markov

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
    "Markov Chain": Markov(n_sensors=3, divisions=10, resample=True, sample_period=20, **common_params),
}
uses_pca = [
    "Histogram-based OD",
    "Local Outlier Factor",
    "Isolation Forest",
    "One-class SVM"
]
assert all(item in algorithms.keys() for item in uses_pca)


def test():
    os.nice(10)
    """ Main testing function """
    start = time.time()

    X, y, class_labels = dtl.get_robot_arm_data(return_class_labels=True)
    print(class_labels)
    X_nor, y_nor = X[y == 0], y[y == 0]
    X_abn, y_abn = X[y != 0], y[y != 0]

    # Rescale
    ss = StandardScaler()
    X_nor = ss.fit_transform(X_nor)
    X_abn = ss.transform(X_abn)

    n_experiments = 1  # Training data starts are offset between experiments

    # Starts every 6 minutes
    step = 6 * 60 / arm_period
    experiment_starts = np.arange(0, n_experiments * step, step, dtype=int)

    results = {outlier: pd.DataFrame(columns=algorithms.keys()) for outlier in class_labels}
    results.pop("Normal")

    splitter = ms.TimeSeriesFolds(n_splits=10, min_test_size=1000)
    dc = decomposition.PCA(n_components=20)

    p = pool.Pool(processes=12)

    for exp_start in experiment_starts:
        X_nor_exp, y_nor_exp = X_nor[exp_start:, :], y_nor[exp_start:]
        exp_results_holder = []
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
                           dc=dc,)
            # for model in algorithms:
            #     performance = func(model)
            performances = p.imap(func, algorithms.keys())
            exp_results_holder.append({"train_size": train_size, "performances": performances})
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
        results[outlier].to_pickle(f"experiment_comparisons/trainsize_{outlier}_results.df")

    print(f"Time elapsed: {time.time() - start}")


def process(model, X_nor_exp, y_nor_exp, X_abn, y_abn, train_idx, test_idx, class_labels, dc):
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
    for outlier_class in class_labels:
        if class_labels.index(outlier_class) == 0:
            # Skip normal class
            continue
        # evaluate the model on outlier classes separately
        outlier_idx = class_labels.index(outlier_class)
        X_test = np.concatenate((X_nor_exp_test, X_abn_exp_test[y_abn == outlier_idx]), axis=0)
        y_test = np.concatenate((y_nor_exp[test_idx], y_abn[y_abn == outlier_idx]), axis=0).astype(bool)

        y_pred = algorithms[model].predict(X_test)
        score = recall_score(y_test, y_pred)

        process_return[outlier_class] = score
    return process_return


def plot_results(filepath):
    from itertools import cycle

    res = pd.read_pickle(filepath)
    res = res.sort_index()
    title = "Outlier type: " + str.join("_", filepath.split("_")[2:-1])

    lines = ["-", "--", "-."]
    linecycler = cycle(lines)

    for algname in res:
        x = []
        y = []
        for idx, item in zip(res.index, res[algname]):
            x.extend([idx] * len(item))
            y.extend(item)
        x = np.array(x) / 6
        y = np.array(y)
        plt.plot(x, y, next(linecycler), label=algname)
        plt.legend()
        plt.ylabel("Recall")
        plt.xlabel("Training time (minutes)")
        plt.title(title)

def _plotall():
    for file in ["highT", "loose_l1", "loose_l2", "sandy", "tight"]:
        plt.figure(figsize=(12, 9))
        plot_results("experiment_comparisons/trainsize_" + file + "_results.df")
        plt.savefig("experiment_comparisons/recall/plot" + file + ".png")
    plt.show()

if __name__ == '__main__':
    test()
