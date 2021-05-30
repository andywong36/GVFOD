import numpy as np

from ..data.dataloader import get_robot_arm_data
from markov import Markov


def main():
    import matplotlib.pyplot as plt
    # from pyod.models.hbos import HBOS as MODEL
    MODEL = Markov

    np.random.seed(97)
    # nor, abn = get_machine_torque_data()
    X, y, a_list = get_robot_arm_data(return_class_labels=True)
    nor = X[y == 0, :]
    abn = [X[y == i, :] for i in range(1, len(a_list))]

    model = MODEL(n_sensors=nor.shape[1] // 2000, contamination=0.05,
                  divisions=6, resample=True, sample_period=1)
    # model = Markov(contamination=0.05)
    trainsize = 0.6

    cutoff = int(nor.shape[0] * trainsize)
    model.fit(nor[:cutoff, :])
    # y_pred = model.decision_function(np.concatenate((nor[:, :], *[item[1] for item in abn])))

    plt.axhline(model.threshold_, linestyle='--', lw=0.5, color='k')

    n = 0
    plt.plot(model.decision_scores_, 'o', color='k', markersize=0.3)
    text_y_loc = np.mean(model.decision_scores_)
    test_y_offset = -2 * np.std(model.decision_scores_)
    text_y_loc += -5 * test_y_offset
    plt.text(cutoff / 2, text_y_loc, 'Training', horizontalalignment='center')
    n += cutoff
    plt.axvline(0, linestyle='--', lw=0.2)
    plt.axvline(n, linestyle='--', lw=0.2)

    plt.plot(np.arange(n, n + len(nor[cutoff:])), model.decision_function(nor[cutoff:]), 'o', color='k',
             markersize=0.3)
    plt.text(len(nor) - (len(nor) - cutoff) / 2, text_y_loc, 'Normal', horizontalalignment='center')
    n += len(nor) - cutoff
    plt.axvline(n, linestyle='--', lw=0.2)

    for i, (dat, label) in enumerate(zip(abn, a_list[1:])):
        plt.plot(np.arange(n, n + len(dat)), model.decision_function(dat), 'o', label=label, color='b',
                 markersize=0.3)
        plt.text(n + len(dat) / 2, text_y_loc + test_y_offset * (i % 3), label, horizontalalignment='center')
        n += len(dat)
        plt.axvline(n, linestyle='--', lw=0.2)

    plt.title("Algorithm: " + str(model))
    plt.ylabel("Outlier Score")
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()