import os

import numpy as np
from pyod.models.lof import LOF
from gvfod import GVFOD, UDE
from sklearn.decomposition import PCA

from data.model_selection import TimeSeriesFolds


def read_and_format_data(file, seed=1):
    np.random.seed(seed)
    arr = np.load(file)
    # the columns are (param_value, position, torque, tension)
    blocks = []
    for i, noise in zip([1, 2, 3],
                        [0, 0, 0],
                        # [0.02, 0.01, 0.5],
                        ):
        blocks.append(arr[:, i].reshape(-1, 2000))
        blocks[-1] += np.random.normal(0, noise, blocks[-1].shape)
    n_normal = int(os.path.basename(file).split("_")[1])
    n_abnormal = int(os.path.basename(file).split("_")[2].split(".")[0]) - n_normal
    y = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)])
    return np.hstack(blocks), y, arr[:, 0]


def run(file):
    import matplotlib.pyplot as plt
    X, y, profile = read_and_format_data(file)

    X = X[:, :]

    print(X.shape, y.shape, profile.shape)

    tsf = TimeSeriesFolds(10, 200, None, 200, None, delay=0)
    odmodel = LOF()
    rlmodel = GVFOD(
        space=[
            [0., 2.5],  # Angle limits
            [-1, 1],  # Torque limits
            [0, 300]
        ],
        divs_per_dim=[50, 50, 2],
        wrap_idxs=None,
        int_idxs=None,
        numtilings=4,
        discount_rate=0.9849630913257917,
        learn_rate=0.1777732644755987,
        lamda=0.6588037848393622,
        beta=15, )
    ude = UDE(
        space=[
            [0., 2.5],  # Angle limits
            [-1, 1],  # Torque limits
            [0, 300]
        ],
        divs_per_dim=[50, 50, 2],
        wrap_idxs=None,
        int_idxs=None,
        numtilings=4,
        discount_rate=0.9849630913257917,
        learn_rate=0.00001,
        lamda=0.1,
        beta=15,
    )

    for i, (train_idx, test_idx) in enumerate(tsf.split(X[y == 0])):
        if i != 3:
            continue
        ylabel = "Static Tension ($T_{base}$)"
        print(len(train_idx))
        dc = PCA(20, whiten=True)
        dc.fit(X[train_idx])
        training_pca = dc.transform(X[y == 0][train_idx])
        testing_pca = dc.transform(np.vstack([X[y == 0][test_idx], X[y != 0]]))
        training = X[y == 0][train_idx]
        testing = np.vstack([X[y == 0][test_idx], X[y != 0]])

        odmodel.fit(training_pca)
        outlier_scores = odmodel.decision_function(np.vstack([training_pca, testing_pca]))

        fig, axs = plt.subplots(4, 1, figsize=(10,8))
        axs[0].plot(np.arange(len(profile)) / 200, profile)
        # axs[0].set(title=f"Algorithm: Local Outlier Factor", ylabel=ylabel)
        axs[0].set(ylabel=ylabel)
        axs[1].plot(np.arange(len(train_idx)) * 10, outlier_scores[:len(train_idx)], label="Training")
        axs[1].plot(np.arange(len(train_idx), len(outlier_scores)) * 10, outlier_scores[len(train_idx):],
                    label="Testing")
        axs[1].axhline(odmodel.threshold_, c="red", label="Outlier Threshold")
        axs[1].legend(loc="upper left")
        axs[1].set(ylabel=r"$score_{LOF}$")

        rlmodel.fit(training)
        outlier_scores = np.hstack([
            rlmodel.decision_scores_,
            rlmodel.decision_function(testing)
            ])

        # fig, axs = plt.subplots(5, 1)
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(np.arange(len(profile)) / 200, profile)
        # axs[0].set(title=f"Algorithm: GVFOD", ylabel=ylabel)
        axs[2].plot(np.arange(len(train_idx)) * 10, outlier_scores[:len(train_idx)], label="Training")
        axs[2].plot(np.arange(len(train_idx), len(outlier_scores)) * 10, outlier_scores[len(train_idx):],
                    label="Testing")
        # for i, label in enumerate(
        #         ["position", "torque", "tension"]
        #         ["position", "torque"]
        # ):
        #     axs[i + 2].plot(np.arange(len(outlier_scores)) * 10,
        #                     [x.mean() for x in np.split(rlmodel.tderrors[:, i], len(outlier_scores))],
        #                     label=label)
        #     axs[i + 2].plot(np.arange(len(outlier_scores)) * 10,
        #                     [x.mean() for x in np.split(rlmodel.surprises[:, i], len(outlier_scores))],
        #                     label=label)
        axs[2].axhline(rlmodel.threshold_, c="r", label="Outlier Threshold", )
        axs[2].legend(loc="upper left")
        axs[2].set(ylabel=r"$score_{GVFOD}$")

        ude.fit(np.vstack([training, testing]))
        outlier_scores = ude.decision_scores_

        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(np.arange(len(profile)) / 200, profile)
        # axs[0].set(title=f"Algorithm: Surprise", ylabel=ylabel)
        axs[3].plot(np.arange(len(outlier_scores)) * 10, outlier_scores, label="Training")
        axs[3].set_ylim(0.85, 1.1)
        axs[3].set(ylabel="Surprise / UDE", xlabel="Time (s)")
    return X


if __name__ == "__main__":
    X = run("data/robot_arm_gradual_failure_sim/baseT_1000_1500.npy")
