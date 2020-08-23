import os

import numpy as np
from pyod.models.lof import LOF
from gvfod import GVFOD
from sklearn.decomposition import PCA

from data.model_selection import TimeSeriesFolds


def read_and_format_data(file, seed=1):
    np.random.seed(seed)
    arr = np.load(file)
    # the columns are (param_value, position, torque, tension)
    blocks = []
    for i, noise in zip([1, 2, 3], [0.02, 0.01, 0.5]):
        blocks.append(arr[:, i].reshape(-1, 2000))
        blocks[-1] += np.random.normal(0, noise, blocks[-1].shape)
    n_normal = int(os.path.basename(file).split("_")[1])
    n_abnormal = int(os.path.basename(file).split("_")[2].split(".")[0]) - n_normal
    y = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    return np.hstack(blocks), y, arr[:, 0]


def run(file):
    import matplotlib.pyplot as plt
    X, y, profile = read_and_format_data(file)
    print(X.shape, y.shape, profile.shape)

    tsf = TimeSeriesFolds(9, 100, None, 100, None, delay=0)
    # model = LOF()
    model = GVFOD(space=[[0., 2.5],  # Angle limits
                         [-1, 1],  # Torque limits
                         [0, 300]],
                  divs_per_dim=[5, 5, 5],
                  wrap_idxs=None,
                  int_idxs=None,
                  numtilings=25,
                  discount_rate=0.95,
                  learn_rate=0.5,
                  lamda=0.25,
                  beta=300, )
    use_pca = False

    for train_idx, test_idx in tsf.split(X[y == 0]):
        if use_pca:
            dc = PCA(20, whiten=True)
            dc.fit(X[train_idx])
            training = dc.transform(X[y == 0][train_idx])
            testing = dc.transform(np.vstack([X[y == 0][test_idx], X[y != 0]]))
        else:
            training = X[y == 0][train_idx]
            testing = np.vstack([X[y == 0][test_idx], X[y != 0]])

        model.fit(training)
        outlier_scores = model.decision_function(np.vstack([training, testing]))

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(np.arange(len(profile)) / 200, profile)
        axs[1].plot(np.arange(len(train_idx)) * 10, outlier_scores[:len(train_idx)], label="Training")
        axs[1].plot(np.arange(len(train_idx), len(outlier_scores)) * 10, outlier_scores[len(train_idx):],
                    label="Testing")

    return X


if __name__ == "__main__":
    X = run("data/robot_arm_gradual_failure_sim/baseT_1000_1500.npy")
