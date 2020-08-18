import math

from methodtools import lru_cache
import numpy as np
from numpy.random import randn
from scipy.interpolate import interp1d
from scipy.linalg import solve


def get_angle(a, b, c):
    """ Cosine law """
    from math import acos
    C = acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    return C


def tanh(x, scale=1.):
    return math.tanh(x / scale)


def dtanh(x, scale=1.):
    return (1 - tanh(x, scale) ** 2) / scale


def ssign(x):
    """ A smooth sign function """
    return tanh(x, scale=0.05)


def dssign(x):
    """ Derivative of the smooth sign function"""
    return dtanh(x, scale=0.05)


class Noise:
    def __init__(self, sigma):
        self.sigma = sigma

    def noise(self):
        def _f(time):
            return randn()

        return _f


class GP:
    def __init__(self, sigma, l, n=2000):
        self.sigma = sigma
        self.l = l
        self.n = n

        # Fix the endpoints at 0

        self.Sigma11 = np.empty((2, 2))
        self.Sigma11[0, 0] = self.Sigma11[1, 1] = self._sekernel(0, 0)
        self.Sigma11[0, 1] = self.Sigma11[1, 0] = self._sekernel(1, 0)

        self.x2 = np.linspace(0, 1, n + 1)[1:-1]  # Drop endpoints

        self.Sigma12 = np.empty((2, n - 1))
        for i in range(2):
            for j in range(n - 1):
                self.Sigma12[i, j] = self._sekernel(i, self.x2[j])

        self.Sigma22 = np.empty((n - 1, n - 1))
        for i in range(n - 1):
            for j in range(i, n - 1):
                self.Sigma22[i, j] = self.Sigma22[j, i] = self._sekernel(self.x2[i], self.x2[j])
        self.Sigma22dot1 = self.Sigma22 - solve(self.Sigma11, self.Sigma12, assume_a='pos').T @ self.Sigma12

    def _sekernel(self, x1, x2):
        return self.sigma ** 2 * math.exp(- (x1 - x2) ** 2 / (2 * self.l ** 2))

    @lru_cache(maxsize=16)
    def gp(self, seed=0):
        np.random.seed(seed)
        z2 = np.random.multivariate_normal(mean=np.zeros(self.n - 1), cov=self.Sigma22dot1)
        z = np.concatenate([[0], z2, [0]])
        x = np.concatenate([[0], self.x2, [1]])
        f = interp1d(x, z)
        return f


def visualize_random_functions():
    """ Plots the difference between GP vs white noise. """

    import matplotlib.pyplot as plt
    gp = GP(1, 0.01)
    x = np.linspace(0, 1, 2000)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    for s in range(5):
        fgp = gp.gp(seed=s)
        y = np.array([fgp(xi) for xi in x])
        axs[0].plot(x, y)
    axs[0].set(xlabel=r'$x$', ylabel=r'$y=f(x)$    $\sigma=1$, $l=0.01$', title="5 Random Samples of a Gaussian Process")

    noise = Noise(1)
    for s in range(5):
        fn = noise.noise()
        y = np.array([fn(xi) for xi in x])
        axs[1].plot(x, y)
    axs[1].set(xlabel=r'$x$', ylabel=r'$y=f(x)$    $\sigma=1$', title="5 Random Samples of White Noise")
