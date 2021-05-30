import os
from typing import Union, List

from pomegranate import HiddenMarkovModel, NormalDistribution
import numpy as np
from pyod.models.base import BaseDetector


class HMM(BaseDetector):
    def __init__(self,
                 # Tile Coding params
                 n_sensors: int,
                 n_states: int,
                 # OD params,
                 max_iterations: int = 1e8,
                 suppress: bool = False,
                 contamination: float = 0.10):
        self.n_sensors = n_sensors
        self.n_states = n_states

        self.verbose = not suppress
        self.max_iterations = max_iterations

        self.means = np.zeros(self.n_sensors)
        self.decision_scores_ = None

        super().__init__(contamination)

    def fit(self, X, y=None):
        X_processed = self._check_and_preprocess(X, True)
        self.hmmmodel = HiddenMarkovModel.from_samples(NormalDistribution, self.n_states, X_processed,
                                                       algorithm="baum-welch",
                                                       n_jobs=8, verbose=self.verbose, batches_per_epoch=20,
                                                       max_iterations=self.max_iterations)
        self.hmmmodel.bake()

        self.decision_scores_ = np.zeros(X.shape[0])
        for i, sequence in enumerate(X_processed):
            self.decision_scores_[i] = -self.hmmmodel.log_probability(sequence)

        self._process_decision_scores()

    def _check_and_preprocess(self, X: np.ndarray, fit_means: bool):
        """ Scales and reshapes data for RL

        Args:
            X: Data to scale / reshape

        Returns:
            X: scaled and reshaped numpy array
        """
        if X.ndim != 2:
            raise ValueError(f"X has the wrong dimensions: {X.shape}")

        n_periods, period_len = X.shape[0], X.shape[1] // self.n_sensors

        if self.n_sensors == 1:
            X = X.reshape(-1, 1)
        else:
            # read all the data, column by column, and stack (ravel) them into a 1D array
            _X_1d = X.ravel(order="F")
            # put this data into a 3D array of shape (n, t_period, n_sensors):
            X = _X_1d.reshape(X.shape[0], -1, self.n_sensors, order="F")
            X = X.reshape(-1, self.n_sensors)

        if fit_means:
            self.means = X.mean(axis=0)

        return np.ascontiguousarray((X - self.means).reshape(n_periods, period_len, self.n_sensors))

    def decision_function(self, X):
        X_processed = self._check_and_preprocess(X, False)

        scores = np.zeros(X.shape[0])
        for i, sequence in enumerate(X_processed):
            scores[i] = -self.hmmmodel.log_probability(sequence)

        return scores
