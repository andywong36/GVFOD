from collections import namedtuple
from functools import reduce
import importlib
from multiprocessing.pool import Pool
import pickle
from typing import Union, Type, Callable
import warnings

import hyperopt
import hyperopt.hp as hp
import numpy as np

import pyod.models.base
import sklearn.metrics
import sklearn.decomposition as decomp
import sklearn.preprocessing as pp

from data.dataloader import get_robot_arm_data


def cross_val_od_score(clf_cls: Type[pyod.models.base.BaseDetector], kwargs: dict,
                       X: np.ndarray, y: np.ndarray, cv: int, scoring: Union[str, Callable],
                       transform: Union[None, Callable], pool: Pool) -> np.ndarray:
    """ Calculates a list of metrics for cross validated experiments on a set of parameters

    Args:
        clf_cls: A classifier to use. Should be a subclass of pyod.models.base.BaseDetector
        kwargs: settings for the classifier
        X: sensor readings (inputs)
        y: 0 for inlier, positive int for outlier class
        cv: folds for cross validation
        scoring: string for input into sklearn.metrics.get_scorer, or callable with arguments (clf, X_test, y_test)
        transform: a function (or None) for transforming the data to different space
        pool: processing pool

    Returns:
        Array of scores/metrics
    """
    from functools import partial
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv, shuffle=False)

    func = partial(_od_score, clf_cls=clf_cls, kwargs=kwargs, X=X, y=y,
                   scoring=scoring, transform=transform)
    iter = pool.imap(func, skf.split(X, y))
    scores = np.array(list(iter))

    results = {
        "loss": 1 - scores.mean(),
        "status": hyperopt.STATUS_OK,
        "loss_variance": scores.var(),
    }

    return results


def _od_score(indices, clf_cls: Type[pyod.models.base.BaseDetector], kwargs: dict,
              X: np.ndarray, y: np.ndarray, scoring: Union[str, Callable],
              transform: Union[None, Callable],
              ) -> np.ndarray:
    """ See cross_val_od_score """
    train_idx, test_idx = indices

    # Select the data
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    # Only normal data goes into training
    X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]

    # Scale the features
    ss = pp.StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # Transform the features if necessary
    if transform is not None:
        # print("Completing transformation")
        X_train = transform.fit_transform(X_train)
        X_test = transform.transform(X_test)

    # Ensure y labels are binary
    y_train, y_test = y_train.astype(np.bool), y_test.astype(np.bool)

    if isinstance(scoring, str):
        scoring = sklearn.metrics.get_scorer(scoring)

    clf = clf_cls(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train)
    return scoring(clf, X_test, y_test)


def run_experiment(exp_param, testrun=False):
    import os

    os.nice(10)

    from sklearn.model_selection import train_test_split

    X, y = get_robot_arm_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True, stratify=y,
                                                        random_state=97)
    # print(f"There are {sum(y == 0)} normal observations")

    # Set the experimental parameters:
    clf, clfname, metric, runs, parameters, transform = exp_param
    cv = 10
    if testrun:
        runs = 5

    # Hyperopt version
    pool = Pool(processes=cv)
    trials = hyperopt.Trials()
    objective = lambda p: cross_val_od_score(clf, p, X=X_train, y=y_train,
                                             cv=cv, scoring=metric, transform=transform, pool=pool)
    best = hyperopt.fmin(fn=objective, space=parameters, algo=hyperopt.tpe.suggest, max_evals=runs, trials=trials)
    print("The best parameters are ", best)
    print(trials.trials)

    with open("hypertune/{}_hyperopt.dat".format(clfname), 'wb') as f:
        pickle.dump(trials.trials, f)

    return 0


def main():
    run_experiment(markov_exp, testrun=False)


### Experiment settings
Experiment = namedtuple("ExperimentSettings",
                        ["clf", "clfname", "metric", "runs", "parameters", "transform"])

hbos_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.hbos"), "HBOS"),
    clfname="HBOS",
    metric="f1",
    runs=500,
    parameters={
        "alpha": hp.uniform("alpha", 0, 1),
        "contamination": 0.05,
        "n_bins": hyperopt.pyll.scope.int(hp.quniform("n_bins", 10, 1000, 1)),
        "tol": hp.uniform("tol", 0, 1)
    },
    transform=None)

ocsvm_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.ocsvm"), "OCSVM"),
    clfname="OCSVM",
    metric="f1",
    runs=200,
    parameters=hp.choice("case", [
        {"kernel": "linear",
         "gamma": hp.choice("g_l", ["auto", "scale"]),
         "coef0": hp.uniform("c", 0, 1),
         "nu": hp.uniform("nu_l", 0, 1),
         "max_iter": 1000,
         "contamination": 0.05},
        {"kernel": "rbf",
         "gamma": hp.choice("g_rbf", ["auto", "scale"]),
         "nu": hp.uniform("nu_rbf", 0, 1),
         "contamination": 0.05}
    ]),
    transform=decomp.PCA(n_components=20))

lof_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.lof"), "LOF"),
    clfname="LOF",
    metric="f1",
    runs=200,
    parameters=hp.choice("distance", [
        {
            "leaf_size": hyperopt.pyll.scope.int(hp.quniform("leaf_size_mink", 10, 100, 1)),
            "p": hp.choice("p", [1, 2]),
            "contamination": 0.05
        },
        {
            "leaf_size": hyperopt.pyll.scope.int(hp.quniform("leaf_size_cheb", 10, 100, 1)),
            "metric": "chebyshev",
            "contamination": 0.05
        }
    ]),
    transform=decomp.PCA(n_components=20)
)

if_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.iforest"), "IForest"),
    clfname="IForest",
    metric="f1",
    runs=200,
    parameters={
        "n_estimators": hyperopt.pyll.scope.int(hp.quniform("n_estimators", 10, 200, 1)),
        "contamination": 0.05,
        "max_features": hp.uniform("max_features", 0.1, 1),
        "bootstrap": hp.choice("bootstrap", [0, 1]),
    },
    transform=decomp.PCA(n_components=20)
)

abod_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.abod"), "ABOD"),
    clfname="ABOD",
    metric="f1",
    runs=100,
    parameters={
        "contamination": 0.05,
        "n_neighbors": hyperopt.pyll.scope.int(hp.quniform("n_neighbors", 5, 50, 1)),
    },
    transform=None
)

pca_exp = Experiment(
    clf=getattr(importlib.import_module("pyod.models.pca"), "PCA"),
    clfname="PCA",
    metric="f1",
    runs=200,
    parameters={
        "contamination": 0.05,
        "n_components": hyperopt.pyll.scope.int(hp.quniform("n_components", 2, 30, 1)),
        "whiten": hp.choice("whiten", [False, True]),
        "weighted": hp.choice("weighted", [False, True])
    },
    transform=None
)


def factors(n):
    return reduce(list.__add__,
                  ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))


markov_exp = Experiment(
    clf=getattr(importlib.import_module("markov.markov"), "Markov"),
    clfname="MarkovChain",
    metric="f1",
    runs=250,
    parameters={
        "n_sensors": 3,
        "contamination": 0.05,
        "divisions": hyperopt.pyll.scope.int(hp.quniform("divisions", 5, 100, 2)),
        "resample": True,
        "sample_period": hp.choice("sample_period", factors(2000))
    },
    transform=None
)

if __name__ == "__main__":
    main()
