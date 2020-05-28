from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sampling_frequency = 100  # Hz
T = 20  # s

class Fourier:
    """
    A class for representing a fourier series as a sum of sines and cosines, with appropriated arithmetic expressions
    such as multiplication and addition, and calcuting a derivative

    Uses the following convention for defining a fourier series

    f(t) ~= a0 + \sum_{n=1}^N [a_n cos( (2pi/T) * n * t ) + b_n sin( (2pi/T) * n * t]
    where f(t) is a periodic function with period T

    A an object fourier can be called to estimate its value at time t.

    """

    def __init__(self, T: float, a0: float, a: np.ndarray, b: np.ndarray):
        self.T = T
        self.a0 = a0
        self.a = a
        self.b = b

        self.ti = 0  # Used to offset calls to this object.

    def __call__(self, t):
        """ Estimates the value a particular time t """
        n = np.arange(1, len(self.a) + 1)
        return self.a0 \
               + np.dot(self.a, np.cos(2 * np.pi * n / self.T * (t - self.ti))) \
               + np.dot(self.b, np.sin(2 * np.pi * n / self.T * (t - self.ti)))

    def similar(self, other):
        return isinstance(other, Fourier) and (self.T == other.T)

    def cut(self, n):
        """ Reduces the number of components in the fourier series, returns the new series """
        assert n < len(self.a), "n needs to be smaller than current number of components"
        return Fourier(self.T, self.a0, self.a[:n], self.b[:n])

    def to_array(self):
        return np.array([self.a0, *self.a, *self.b])

    @property
    def ddt(self):
        n = np.arange(1, len(self.a) + 1)
        anew = 2 * np.pi / self.T * n * self.b
        bnew = - 2 * np.pi / self.T * n * self.a
        a0new = 0
        return Fourier(self.T, a0new, anew, bnew)

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return Fourier(self.T, other * self.a0, other * self.a, other * self.b)

    def __rmul__(self, other):
        assert isinstance(other, (int, float))
        return self * other

    def __add__(self, other):
        assert self.similar(other)
        return Fourier(self.T, self.a0 + other.a0, self.a + other.a, self.b + other.b)

    def __eq__(self, other):
        return self.similar(other) and (self.a0 == other.a0) and np.all(self.a == other.a) and np.all(self.b == other.b)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        assert self.similar(other)
        return self + -other

    def __matmul__(self, other):
        # Returns the inner product
        assert self.similar(other)
        return sum([self.a0 * other.a0, self.a @ other.a, self.b @ other.b])

    def sum_squares(self):
        return self @ self

    def norm(self):
        return sqrt(self.sum_squares())


def fit_fourier(data, fs, T, n_terms) -> Fourier:
    n_periods = len(data) // (fs * T)
    c = np.fft.rfft(data) / len(data)
    a0 = np.real(c[0])
    a = np.real(c[n_periods::n_periods] + np.conj(c[n_periods::n_periods]))
    b = - np.imag(c[n_periods::n_periods] - np.conj(c[n_periods::n_periods]))

    if n_terms == 'all':
        return Fourier(T, a0, a, b)
    else:
        return Fourier(T, a0, a[:n_terms], b[:n_terms])


if __name__ == "__main__":
    n = 4000
    data = pd.read_csv("test_data_extended.csv")
    torque = data["Torque"].to_numpy()[:n]
    angle = data["Angle"].to_numpy()[:n]

    torque_ft = fit_fourier(torque[:n], sampling_frequency, T, n_terms='all')
    angle_ft = fit_fourier(angle[:n], sampling_frequency, T, n_terms='all')
    torque_ft.ti = 0.01
    angle_ft.ti = 0.01

    # Plot the coefficients (only for torque data)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(np.linspace(1 / T, sampling_frequency / 2, len(torque_ft.a)), torque_ft.a)
    axs[0].set(title="Fourier coefficients of Motor Torque (T = 20s)", ylabel=r"$a_n$", xlabel="Frequency (Hz)")
    axs[1].plot(np.linspace(1 / T, sampling_frequency / 2, len(torque_ft.a)), torque_ft.b)
    axs[1].set(ylabel="$b_n$", xlabel="Frequency (Hz)")

    # Calculate the percentage of variance explained by each component
    cumvar_t = np.cumsum((np.square(torque_ft.a) + np.square(torque_ft.b)) / 2) / np.var(torque[:n])
    cumvar_a = np.cumsum((np.square(angle_ft.a) + np.square(angle_ft.b)) / 2) / np.var(angle[:n])

    # Select the major components
    n_components = 40

    # Approximate the torque induced by a slanted surface
    slope1 = 0.292
    slope2 = -0.138
    slope_ft = fit_fourier(slope1 * np.sin(angle[:n]) + slope2 * np.cos(angle[:n]),
                           sampling_frequency,
                           T,
                           n_terms=n_components)
    slope_ft.ti = 0.01

    # See if the new function matches the old
    x = data["Time"][:n]

    fig2, ax2 = plt.subplots()
    ax2.plot(x, np.fromiter(map(torque_ft.cut(n_components), x), dtype=float), label="Estimated")
    ax2.plot(x, torque[:n], label="True")
    # ax2.plot(data["Time"][:data_length], data["Torque"][10000:10000 + data_length], label="True (offset)")
    ax2.set(title="Estimation of Torque with {} Fourier bases".format(n_components),
            xlabel="Time",
            ylabel=r"Torque (Nm)"
            )

    fig3, ax3 = plt.subplots()
    ax3.plot(x, np.fromiter(map(angle_ft.cut(n_components), x), dtype=float), label="Estimated")
    ax3.plot(x, angle[:n], '--', label="True")
    ax3.set(title="Estimation of Angle with {} Fourier bases".format(n_components),
            xlabel="Time",
            ylabel=r"Angle (rad)")

    fig4, ax4 = plt.subplots()
    ax4.plot(x, np.fromiter(map(slope_ft, x), dtype=float), label="Estimated")
    ax4.plot(x,
             slope1 * np.sin(angle[:n]) + slope2 * np.cos(angle[:n]),
             "--",
             label="True")
    ax4.set(title="Estimation of Slant-induced torque with {} Fourier bases".format(n_components),
            xlabel="Time",
            ylabel=r"Torque (Nm)")

    for ax in [ax2, ax3, ax4]:
        ax.legend()