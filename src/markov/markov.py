""" The purpose of this is to compare a k-order markov chain to outlier detection techniques for finding outliers
This will be a univariate analysis of torque akin to previous analysis. """

from typing import Union

import numpy as np
from pyod.models.base import BaseDetector

from exp.data import get_robot_arm_data


class Markov(BaseDetector):
    def __init__(self, n_sensors, contamination: float,
                 divisions: Union[int, None] = None,
                 resample: bool = False,
                 sample_period: int = 1):
        """ Creates PyOD-like object for outlier detection in time series using Markov chains

        Args:
            contamination: a value between 0 and 0.5
            divisions: number of bins of the input readings. None for no scaling
            resample: drop samples from the input time series. useful for signals with high sample frequency
            sample_period: only if useful if resample is True. This is the subsampling frequency. Keeps only times where
                t = i * sample_period where i is a natural number.
        """
        self.P = None  # Transition probability
        self.u = None  # Start probability

        # hyperparameters
        self.divisions = divisions
        self._scaling_factors = None  # based on divisions and the range of the training data
        self.resample = resample
        self.sample_period = sample_period

        self.states = None
        self.threshold_ = None
        self.decision_scores_ = None

        self._n_sensors = n_sensors

        super().__init__(contamination)

    def __str__(self):
        return 'MarkovChain'

    def fit(self, X, y=None):
        """ Fit training data for outlier detection

        Args:
            X: data to fit
            y: not used, for compatibility

        Returns:
            self

        """
        X = self._preprocess(X, set_scaling=True)

        for sensor_idx in range(self._n_sensors):
            self.states[sensor_idx] = np.unique(X[..., sensor_idx])
            assert len(self.states[sensor_idx] < 100), "Too many states to create discrete Markov chain"
            self.u[sensor_idx] = np.zeros(len(self.states[sensor_idx]))
            self.P[sensor_idx] = np.zeros((len(self.states[sensor_idx]), len(self.states[sensor_idx])))

            # Create local references
            _states = self.states[sensor_idx]
            _u = self.u[sensor_idx]
            _P = self.P[sensor_idx]

            for i, s in enumerate(_states):
                _u[i] = np.sum(X[:, 0, sensor_idx] == s) / X.shape[0]

            for i, si in enumerate(_states):
                fi = np.sum(X[:, :-1, sensor_idx] == si)
                for j, sj in enumerate(_states):
                    _P[i, j] = np.sum(np.multiply(X[:, :-1, sensor_idx] == si, X[:, 1:, sensor_idx] == sj))
                if fi:
                    _P[i, :] /= fi
                else:  # handle divide by zero errors
                    assert np.all(_P[i, :] == 0)

        self.decision_scores_ = self.nll(X)
        self.threshold_ = np.quantile(self.decision_scores_, (1 - self.contamination))

        self._process_decision_scores()

        return self

    def _preprocess(self, X: np.ndarray, set_scaling: bool):
        """ Rescales and reshapes X based on the hyperparameters"""
        if X.ndim == 1:
            raise ValueError("X only has 1 dimensions")
        elif X.ndim == 2:
            if self._n_sensors == 1:
                X = X[:, :, None]  # adds another axis
            else:
                # 3rd axis = sensor
                X = np.reshape(X.ravel(order="F"), (X.shape[0], -1, self._n_sensors), order="F")
        else:
            assert X.ndim == 3
            assert self._n_sensors == X.shape[2]

        assert X.ndim == 3, "X needs to have either 2 or 3 dimensions"

        # initialize some values
        # print(f"type:scaling_factors = {type(self._scaling_factors)}; type:div = {type(self.divisions)}")
        if self._scaling_factors is None and self.divisions:
            self._scaling_factors = np.ones(X.shape[2])
        if (not self.states) or set_scaling:
            self.states = [np.array([])] * self._n_sensors
        if (not self.u) or set_scaling:
            self.u = [np.array([])] * self._n_sensors
        if (not self.P) or set_scaling:
            self.P = [np.array([])] * self._n_sensors

        # Drop some values (if sampling frequency is too high)
        if self.resample:
            X = X[:, np.arange(X.shape[1], step=self.sample_period), :]

        # Rescale
        if self.divisions:  # if self.divisions is None, keep X as is
            if set_scaling:
                for sensor_idx in range(X.shape[2]):
                    self._scaling_factors[sensor_idx] = self.divisions / X[..., sensor_idx].ptp()
            X = X * self._scaling_factors  # *= doesn't work for type mismatch

        # Make sensor readings pseudo-categorical
        X = X.astype(int)
        return X

    def nll(self, X):
        """ Calculate the negative log-likelihood of an observation (row of X) based on the previous trained model
        Args:
            X: input matrix, rows are time series.

        Returns:
            vector of negative log likelihoods, length of X.shape[0]

        """
        assert self.P is not None
        p = np.zeros(X.shape[0])

        for sensor_idx in range(self._n_sensors):
            with np.errstate(divide="ignore"):
                logp = np.log(self.P[sensor_idx])
                logu = np.log(self.u[sensor_idx])
            reverse_states = {s: i for i, s in enumerate(self.states[sensor_idx])}
            for i in range(X.shape[0]):
                try:
                    p[i] -= logu[reverse_states[X[i, 0, sensor_idx]]]
                    for j in range(X.shape[1] - 1):
                        p[i] -= logp[reverse_states[X[i, j, sensor_idx]], reverse_states[X[i, j + 1, sensor_idx]]]
                except KeyError:
                    p[i] = np.inf

        return np.minimum(p, X.shape[1] * X.shape[2] / 2)

    def decision_function(self, X):
        """ for pyod """
        X = self._preprocess(X, set_scaling=False)
        return self.nll(X)


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
