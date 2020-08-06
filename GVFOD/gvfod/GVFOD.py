# An interface to use reinforcement learning similar to a PyOD model

import os
from typing import Union, List

import numpy as np
from pyod.models.base import BaseDetector

from .surprise import TDLambdaGVF
from .tile_coder import TileCoder


class GVFOD(BaseDetector):
    def __init__(self,
                 # Tile Coding params
                 space: Union[List[List[float]], np.ndarray],
                 divs_per_dim: Union[List[int], np.ndarray],
                 wrap_idxs: Union[None, List[int]] = None,
                 int_idxs: Union[None, List[int]] = None,
                 numtilings: int = 32,
                 # RL params
                 discount_rate: float = 0.,
                 learn_rate: float = 0.0,
                 lamda: float = 0.,
                 beta: int = 10,
                 # OD params
                 contamination: float = 0.10):
        """ GVFOD is an algorithm that detects anomalous time series, with an interface similar to PyOD.

        Update for new tile coder: no scaling to the data is done.
        Data that exceeds the bounds of 'space' has undefined behavior.

        Args:
            space: The min and max values for each sensor. Input data X for fit() and predict() should have a shape of
                (n_samples, n_sensors * period). Hence, X.shape[1] must be divisible by n_sensors
            divs_per_dim: The number of divisions to discretize each sensor's data. The smallest resolution that the tiler
                can handle is divisions[sensor] * num_tilings.
            wrap_idxs: Not implemented
            int_idxs: The indices of discrete variables
            numtilings: Sub-discretizations. From tiling software.
            discount_rate: \gamma, in standard RL notation from Sutton and Barto (2018)
            learn_rate: float between (0, 1). It will be divided by numtilings, so \alpha = learn_rate / numtilings
            lamda: the trace decay parameter from Sutton and Barto (2018)
            beta: the bandwidth for calculating UDE (White, 2015)
            contamination: see PyOD
        """
        self.space = np.asarray(space)
        self.n_sensors = self.space.shape[0]
        self.divs_per_dim = np.asarray(divs_per_dim)

        self.int_idxs = int_idxs if int_idxs is not None else []
        self.float_idxs = [i for i in range(self.n_sensors) if i not in self.int_idxs]
        # check that only floats are wrapped
        if wrap_idxs or int_idxs:
            raise ValueError("wrap_idxs and int_idxs not implemented")

        self.numtilings = numtilings
        self.tilecoder = TileCoder(self.space, self.divs_per_dim,
                                   self.numtilings, bias_unit=True)

        self.discount_rate = discount_rate
        self.learn_rate = learn_rate / (numtilings + 1)
        self.lamda = lamda
        self.beta = beta
        self.models = [
            TDLambdaGVF(
                self.tilecoder.total_num_tiles,
                self.discount_rate,
                self.learn_rate,
                self.lamda,
                self.beta)
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
            outlier scores for each observation.
        """

        # data preprocessing
        n_samples = len(X)
        X_stacked = self._check_and_preprocess(X)

        # tiling
        phi = self.tilecoder.encode(X_stacked)

        # fit the models
        surprise = np.empty_like(X_stacked)
        tderrors = np.empty_like(X_stacked)
        for j in range(self.n_sensors):
            print(f"Fitting on sensor {j}")
            tderrors[:, j], surprise[:, j] = self.models[j].learn(phi, X_stacked[:, j])

        # Calculate new surprise values using this trained model, and no training rate
        for j in range(self.n_sensors):
            print(f"Evaluating TD errors on sensor {j}")
            tderrors[:, j], surprise[:, j] = self.models[j].eval(phi, X_stacked[:, j])

        # set decision scores of training data
        self.decision_scores_ = np.empty(n_samples)
        for i, v in enumerate(np.vsplit(surprise, n_samples)):
            self.decision_scores_[i] = np.mean(v)

        self._process_decision_scores()

    def _check_and_preprocess(self, X: np.ndarray):
        """ Scales and reshapes data for RL

        Args:
            X: Data to scale / reshape

        Returns:
            X: scaled and reshaped numpy array
        """
        if X.ndim != 2:
            raise ValueError(f"X has the wrong dimensions: {X.shape}")

        if self.n_sensors == 1:
            X = X.reshape(-1, 1)
        else:
            # read all the data, column by column, and stack (ravel) them into a 1D array
            _X_1d = X.ravel(order="F")
            # put this data into a 3D array of shape (n, t_period, n_sensors):
            X = np.reshape(_X_1d, (X.shape[0], -1, self.n_sensors), order="F")
            X = X.reshape(-1, self.n_sensors)

        return np.ascontiguousarray(X)

    def decision_function(self, X):
        n_samples = len(X)
        X_stacked = self._check_and_preprocess(X)

        phi = self.tilecoder.encode(X_stacked)

        surprise = np.empty_like(X_stacked)
        tderrors = np.empty_like(X_stacked)
        for j in range(self.n_sensors):
            # self.models[j].alpha = 0
            tderrors[:, j], surprise[:, j] = self.models[j].eval(phi, X_stacked[:, j])
        averaged = np.empty(n_samples)
        for i, v in enumerate(np.vsplit(surprise, n_samples)):
            averaged[i] = np.mean(v)

        return averaged
