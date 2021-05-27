###
# Desired:
# Choose an algorithm (hyperparameters?)
# Pick a training size
# Pick a testing size
# Fit the algorithm
# Plot the training and testing outlier scores at different start points
###
from multiprocessing import Pool
from tqdm import tqdm

from pyod.models.lof import LOF as ALG
# from pyod.models.iforest import IForest as ALG
from .exp_train_size_settings import lof_exp as EXP
# from exp_gvfod.exp_train_size_settings import if_exp as EXP

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

from ..data.model_selection import TimeSeriesFolds
from ..data.dataloader import get_robot_arm_data

splits = 5


def process_eval(model, train, test):
    """ Returns list of training and testing outlier scores """
    m = model.fit(train)
    scores = model.decision_function(test)

    return [m.decision_scores_, scores]


def main(alg, alg_kwargs, train_size, test_size):

    X, y, labels = get_robot_arm_data(True)
    X_nor = X[y == 0]

    ss = StandardScaler()
    X_nor_ss = ss.fit_transform(X_nor)

    pca = decomposition.PCA(n_components=20)
    X_nor_ss_dc = pca.fit_transform(X_nor_ss)

    tsf = TimeSeriesFolds(splits,
                          min_train_size=train_size,
                          max_train_size=train_size,
                          min_test_size=test_size,
                          max_test_size=test_size,
                          delay=0)

    p = Pool()
    jobs = []
    start_times = []
    for i, (train_idx, test_idx) in enumerate(tsf.split(X_nor_ss_dc)):
        jobs.append(p.apply_async(process_eval, args=(
            alg(**alg_kwargs),
            X_nor_ss_dc[train_idx],
            X_nor_ss_dc[test_idx])))
        start_times.append(train_idx[0])

    p.close()

    all_scores = []
    for i, job in enumerate(tqdm(jobs, total=splits)):
        r = job.get()
        # print(r)
        all_scores.append([start_times[i], *r])

    p.join()

    return all_scores


def ma(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n] *= n / (np.arange(n) + 1)
    return ret / n

def plot(all_scores):
    import matplotlib.pyplot as plt
    from cycler import cycler

    fig, ax = plt.subplots(figsize=(10, 2))

    bw = 100

    ax.set_prop_cycle(cycler('color', plt.get_cmap("tab20").colors))
    for i, row in enumerate(all_scores):
        offset = i * 0.
        start = row[0]
        train_scores = row[1]
        test_scores = row[2]
        ax.plot(np.arange(start, start + len(train_scores)),
                ma(train_scores + offset, bw),
                # "."
                )
        ax.plot(np.arange(start + len(train_scores),
                          start + len(train_scores) + len(test_scores)),
                ma(test_scores + offset, bw),
                # "."
                )
        ax.set(ylim=[0.95, 2.5])

    plt.show()


if __name__ == "__main__":
    model = ALG
    model_kwargs = EXP.kwargs
    train_size = 1500
    test_size = 500
    scores = main(ALG, EXP.kwargs, train_size, test_size)
    # print(scores)
    plot(scores)
