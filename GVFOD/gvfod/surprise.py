import numpy as np
import pandas as pd
from tqdm import trange

from gvfod import flearn


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

        # Check the types
        assert np.all(x >= 0)

        self.tderrors = np.zeros(len(y))
        flearn(np.ascontiguousarray(x, dtype=np.uintp),
               np.ascontiguousarray(y),
               self.tderrors, self.w,
               self.z, self.gamma,
               self.lamda, self.alpha)

        self.surprise = self._surprise(self.beta)

        return self.tderrors, self.surprise

    def eval(self, x, y):
        assert len(y) == x.shape[0]
        assert y.ndim == 1
        self.tderrors = np.zeros(len(y))

        flearn(np.ascontiguousarray(x, dtype=np.uintp),
               np.ascontiguousarray(y),
               self.tderrors, self.w,
               self.z, self.gamma,
               self.lamda, 0.)

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
