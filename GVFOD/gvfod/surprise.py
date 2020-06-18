from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

from .state import Tiler
from .scaling import scaling


class TDLambdaGVF:
    FLOATEPS = np.finfo(np.float).eps

    def __init__(self, state_size, discount_rate, learn_rate, lamda, beta):
        self.state_size = state_size
        self.w = np.zeros(self.state_size)
        self.z = np.zeros(self.state_size)  # Eligibility trace
        self.gamma = discount_rate
        self.alpha = learn_rate
        self.lamda = lamda

        self.beta = beta

        self.tderrors = np.array([0])
        self.surprise = np.array([0])

    def value(self, x):
        return np.sum(self.w[x])

    def learn(self, x, y):
        assert len(y) == x.shape[0]
        self.tderrors = np.zeros(len(y))
        self.surprise = np.zeros(len(y))
        if isinstance(y, (pd.Series, pd.DataFrame)):
            for t in trange(y.shape[0] - 1, desc="Processing {}".format(y.name)):
                delta = (y.iat[t + 1]
                         + self.gamma * self.value(x[t + 1, :])
                         - self.value(x[t, :]))
                self.tderrors[t] = delta

                self.z = self.z * self.gamma * self.lamda
                for i in x[t, :]:
                    self.z[i] += 1
                self.w = self.w + self.alpha * delta * self.z
        elif isinstance(y, np.ndarray):
            assert y.ndim == 1
            for t in trange(y.shape[0] - 1):
                delta = (y[t + 1]
                         + self.gamma * self.value(x[t + 1, :])
                         - self.value(x[t, :]))
                self.tderrors[t] = delta

                self.z = self.z * self.gamma * self.lamda
                for i in x[t, :]:
                    self.z[i] += 1
                self.w = self.w + self.alpha * delta * self.z
        self.surprise = self._surprise(self.beta)

        return self.tderrors, self.surprise

    def eval(self, x, y):
        assert len(y) == x.shape[0]
        self.tderrors = np.zeros(len(y))
        self.surprise = np.zeros(len(y))
        if isinstance(y, (pd.Series, pd.DataFrame)):
            for t in trange(y.shape[0] - 1, desc="Processing {}".format(y.name)):
                delta = (y.iat[t + 1]
                         + self.gamma * self.value(x[t + 1, :])
                         - self.value(x[t, :]))
                self.tderrors[t] = delta

        elif isinstance(y, np.ndarray):
            assert y.ndim == 1
            for t in trange(y.shape[0] - 1):
                delta = (y[t + 1]
                         + self.gamma * self.value(x[t + 1, :])
                         - self.value(x[t, :]))
                self.tderrors[t] = delta

        self.surprise = self._surprise(self.beta)

        return self.tderrors, self.surprise

    def _tde_ma(self, n):
        ret = np.cumsum(self.tderrors, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n

    def _surprise(self, beta):
        std = pd.Series(self.tderrors).expanding(2).std(ddof=0)
        std = std.fillna(0).values
        surprise = np.abs(np.divide(self._tde_ma(beta), (std + self.FLOATEPS)))
        surprise[:2] = 0
        return surprise
