# An interface to use reinforcement learning similar to a PyOD model

import os
from typing import Union, List

import numpy as np
from pyod.models.base import BaseDetector
from sklearn.preprocessing import MinMaxScaler as MMS

from . import tiles3 as tile
from .surprise import TDLambdaGVF
from . import clearn


class RLOD(BaseDetector):
    def __init__(self, n_sensors: int, divisions: List[int],
                 wrap_idxs: Union[None, List[int]] = None, int_idxs: List[int] = None, numtilings: int = 32,
                 state_size=4096, discount_rate: float = 0., learn_rate: float = 0.0,
                 lamda: float = 0., beta: int = 10,
                 contamination: float = 0.10):
        self.n_sensors = n_sensors
        assert len(divisions) == self.n_sensors

        self.int_idxs = int_idxs if int_idxs is not None else []
        self.divisions = divisions
        self.float_idxs = [i for i in range(self.n_sensors) if i not in self.int_idxs]
        # check that only floats are wrapped
        if wrap_idxs is not None:
            assert all(widx in self.float_idxs for widx in wrap_idxs), "Int is wrapped, only floats can be wrapped"
            assert all(self.divisions[widx] != 0 for widx in wrap_idxs), "Divisions does not exist for wrapped indices"
            self.wrapwidths = [self.divisions[fidx] if fidx in wrap_idxs else False for fidx in self.float_idxs]
        else:
            self.wrapwidths = [False] * len(self.float_idxs)

        self.numtilings = numtilings
        self.iht = tile.IHT(state_size)

        self._scalers = None

        self.discount_rate = discount_rate
        self.learn_rate = learn_rate / numtilings
        self.lamda = lamda
        self.beta = beta
        self.models = [TDLambdaGVF(state_size, self.discount_rate, self.learn_rate, self.lamda, self.beta)
                       for _ in range(self.n_sensors)]
        self.decision_scores_ = None

        super().__init__(contamination)

    def fit(self, X, y=None):
        """ A PyOD-like interface for RL fault detection
        Args:
            X: assume that data arrives as a 2D array, where the sensor time series are concatenated in the
                second dimensions (axis=1)
            y: Unused, for compatibility

        Returns:

        """

        # data preprocessing
        n_samples = len(X)
        X_scaled = self._preprocess(X, set_scaling=True)

        # tiling
        phi = self.tiling(X_scaled)

        # fit the models
        surprise = np.empty_like(X_scaled)
        tderrors = np.empty_like(X_scaled)
        for j in range(self.n_sensors):
            print(f"Fitting on sensor {j}")
            # This is the fast version:
            if os.name == 'nt':
                self.models[j].tderrors = np.zeros(len(phi))
                clearn.learn(phi, X_scaled[:, j], self.models[j].tderrors, self.models[j].w,
                             self.models[j].z, self.models[j].gamma, self.models[j].lamda,
                             self.models[j].alpha)
                tderrors[:, j] = self.models[j].tderrors
                self.models[j].surprise = self.models[j]._surprise(self.beta)
                surprise[:, j] = self.models[j].surprise
            else:
                tderrors[:, j], surprise[:, j] = self.models[j].learn(phi, X_scaled[:, j])

        # Calculate new surprise values using this trained model, and no training rate
        for j in range(self.n_sensors):
            if os.name == 'nt':
                # This is the fast version
                self.models[j].tderrors = np.zeros(len(phi))
                clearn.learn(phi, X_scaled[:, j], self.models[j].tderrors, self.models[j].w,
                             self.models[j].z, self.models[j].gamma, self.models[j].lamda,
                             0.)
                tderrors[:, j] = self.models[j].tderrors
                self.models[j].surprise = self.models[j]._surprise(self.beta)
                surprise[:, j] = self.models[j].surprise
            else:
                tderrors[:, j], surprise[:, j] = self.models[j].eval(phi, X_scaled[:, j])

        # set decision scores of training data
        averaged = [np.mean(arr) for arr in np.vsplit(surprise, n_samples)]
        self.decision_scores_ = np.array(averaged)
        self._process_decision_scores()
        pass

    def tiling(self, X_scaled):
        phi = np.empty((len(X_scaled), self.numtilings), dtype=int)
        float_array, int_array = X_scaled[:, self.float_idxs], X_scaled[:, self.int_idxs].astype(int)
        if any(self.wrapwidths):
            for idx, (f_row, i_row) in enumerate(zip(float_array, int_array)):
                phi[idx] = tile.tileswrap(self.iht, self.numtilings,
                                          f_row,
                                          self.wrapwidths,
                                          i_row)
        else:
            for idx, (f_row, i_row) in enumerate(zip(float_array, int_array)):
                phi[idx] = tile.tiles(self.iht, self.numtilings,
                                      f_row, i_row)
        return phi

    def _preprocess(self, X: np.ndarray, set_scaling: bool):
        """ Scales and reshapes data for RL

        Args:
            X: Data to scale / reshape
            set_scaling: True if scaling functions need to be fitted

        Returns:
            X: scaled and reshaped numpy array
        """
        X = X.copy()  # Create a copy so that original data is not overwritten

        if X.ndim == 1:
            raise ValueError("X only has 1 dimensions")
        elif X.ndim == 2:
            if self.n_sensors == 1:
                X = X.reshape(-1, 1)
            else:
                # 2nd axis = sensor
                X = np.reshape(X.ravel(order="F"), (X.shape[0], -1, self.n_sensors), order="F")  # stack into 3D
                X = X.reshape(-1, self.n_sensors)  # unstack into 2D again
        else:
            raise ValueError(f"X has the wrong dimensions: {X.shape}")

        if set_scaling:
            assert len(self.divisions) == self.n_sensors
            self._scalers = [MMS(feature_range=(0, div)).fit(X[:, None, j]) for j, div in enumerate(self.divisions)]
        else:
            if self._scalers is None:
                raise ValueError("Fitting not yet completed")

        for j in range(X.shape[1]):
            X[:, j] = self._scalers[j].transform(X[:, None, j]).flatten()
            if set_scaling:
                assert np.isclose(np.ptp(X[:, j]), self.divisions[j])

        return X

    def decision_function(self, X):
        n_samples = len(X)
        X_scaled = self._preprocess(X, set_scaling=False)

        phi = self.tiling(X_scaled)

        surprise = np.empty_like(X_scaled)
        tderrors = np.empty_like(X_scaled)
        for j in range(self.n_sensors):
            # self.models[j].alpha = 0
            tderrors[:, j], surprise[:, j] = self.models[j].eval(phi, X_scaled[:, j])
        averaged = np.array([np.mean(arr) for arr in np.vsplit(surprise, n_samples)])
        return averaged
