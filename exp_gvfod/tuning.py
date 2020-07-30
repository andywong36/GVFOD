from multiprocessing import Pool, RawArray
import pickle
import sys
from typing import Union, Type, Callable
import warnings

import pyod.models.base
import sklearn.metrics
import sklearn.preprocessing as pp

from exp_gvfod.tuning_settings import *
from data.dataloader import get_robot_arm_data
from hyperopt_wrap_cost import wrap_cost

global_data = {}


def init_data(X, X_shape, y, y_shape):
    """ Used to pass data to processes without copying """
    global_data["X"] = X
    global_data["X_shape"] = X_shape
    global_data["y"] = y
    global_data["y_shape"] = y_shape


def main(exp_param, testrun=False):
    """ Finds the best hyperparameters for an outlier detection algorithm

    Saves the trials.trials object into "exp_gvfod/tuning_results/{}_hyperopt.dat".format(exp_param.clfname).

    Args:
        exp_param: the name of a namedtuple("Experiment"): See tuning_settings.py, and the Experiment namedtuple.
            fields of Experiment:
                "clf": a subclass of pyod.models.base.BaseDetector
                "clfname": string, name of "clf"
                "metric": string for input into sklearn.metrics.get_scorer, or callable with arguments
                    (clf, X_test, y_test)
                "runs": number of trials for hyperopt to run
                "parameters": parameter structure that is accepted by fmin of hyperopt
        testrun (bool): sets runs to 5 if True

    Returns:
        int: 0 if successful.

    """

    from functools import partial
    import os
    import psutil

    from sklearn.model_selection import train_test_split

    if isinstance(exp_param, str):
        from exp_gvfod import tuning_settings
        exp_param = getattr(tuning_settings, exp_param)

    if os.name == 'posix':
        os.nice(10)
    else:
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)

    X, y = get_robot_arm_data()
    ## Reserve half of the abnormal data
    X_new, y_new = X[y == 0], y[y == 0]
    for i in np.unique(y):
        if i == 0:
            continue
        X_train, _, y_train, _ = train_test_split(X[y == i], y[y == i], train_size=0.5, shuffle=False)
        X_new = np.concatenate([X_new, X_train])
        y_new = np.concatenate([y_new, y_train])

    X_new_raw = RawArray('d', X_new.shape[0] * X_new.shape[1])
    y_new_raw = RawArray('i', y_new.shape[0])
    X_new_raw_np = np.frombuffer(X_new_raw).reshape(X_new.shape)
    y_new_raw_np = np.frombuffer(y_new_raw, dtype=np.int32).reshape(y_new.shape)
    np.copyto(X_new_raw_np, X_new)
    np.copyto(y_new_raw_np, y_new)

    # multiprocessing
    cv = 10
    pool = Pool(processes=cv, initializer=init_data, initargs=[X_new_raw, X_new.shape, y_new_raw, y_new.shape])

    # Start the experiment - search for the best hyperparameters.
    trials = hyperopt.Trials()
    objective = partial(cross_val_od_score,
                        exp_param.clf,
                        X=X_new,
                        y=y_new,
                        cv=cv,
                        scoring=exp_param.metric,
                        pool=pool)
    best = hyperopt.fmin(fn=wrap_cost(objective, timeout=600),
                         space=exp_param.parameters,
                         algo=hyperopt.tpe.suggest,
                         max_evals=exp_param.runs if not testrun else 5,
                         trials=trials)
    print("The best parameters are ", best)
    print(trials.trials)

    pool.close()
    pool.join()

    with open("exp_gvfod/tuning_results/{}_hyperopt.dat".format(exp_param.clfname), 'wb') as f:
        pickle.dump(trials.trials, f)

    return 0


def cross_val_od_score(clf_cls: Type[pyod.models.base.BaseDetector], kwargs: dict,
                       X: np.ndarray, y: np.ndarray,
                       cv: int, scoring: Union[str, Callable], pool: Pool):
    """ Calculates a list of metrics for cross validated experiments on a set of parameters

    Args:
        clf_cls: A classifier to use. Should be a subclass of pyod.models.base.BaseDetector
        kwargs: settings for the classifier, as well as the transform
        X: sensor readings (inputs)
        y: 0 for inlier, positive int for outlier class
        cv: folds for cross validation
        scoring: string for input into sklearn.metrics.get_scorer, or callable with arguments (clf, X_test, y_test)
        pool: processing pool

    Returns:
        Array of scores/metrics
    """
    from functools import partial
    from sklearn.model_selection import StratifiedKFold
    from data.model_selection import TimeSeriesFolds

    # New style cross validation
    tsf = TimeSeriesFolds(n_splits=cv,
                          # min_train_size=3000, max_train_size=3000,
                          # min_test_size=1000, max_test_size=1000,
                          min_train_size=3000, max_train_size=3000,
                          min_test_size=1500, max_test_size=1500,
                          delay=0)

    def split_add_abnormal(split):
        for train_idx, test_idx in split:
            yield train_idx, np.concatenate([test_idx, np.where(y != 0)[0]])

    transform = kwargs.pop("transform")
    func = partial(_od_score, clf_cls=clf_cls, kwargs=kwargs,
                   # X=X, y=y,
                   scoring=scoring, transform=transform)
    imap_res = pool.imap(func, split_add_abnormal(tsf.split(X[y == 0])))
    scores = np.array(list(imap_res))

    return {
        "loss": 1 - scores.mean(),
        "status": hyperopt.STATUS_OK,
        "loss_variance": scores.var(),
    }


def _od_score(indices, clf_cls: Type[pyod.models.base.BaseDetector], kwargs: dict,
              # X: np.ndarray, y: np.ndarray,
              scoring: Union[str, Callable],
              transform: Union[None, Callable],
              ) -> np.ndarray:
    """ See cross_val_od_score """
    train_idx, test_idx = indices

    X = np.frombuffer(global_data["X"]).reshape(global_data["X_shape"])
    y = np.frombuffer(global_data["y"], dtype=np.int32).reshape(global_data["y_shape"])

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


if __name__ == "__main__":

    # main(pca_exp, testrun=False)  # Completed July 29 16:00

    main(sys.argv[1], testrun=int(sys.argv[2]))
