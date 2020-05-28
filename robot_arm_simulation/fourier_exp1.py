"""
An experiment to solve the equation

d2theta = k1 * Torque + k2 * slope + c1 * theta + c2 * dtheta

"""

from fourier import *

if __name__ == "__main__":
    n = 4000

    data = pd.read_csv("test_data_extended.csv")
    torque = data["Torque"].to_numpy()[:n]
    angle = data["Angle"].to_numpy()[:n]

    slope1 = 0.292
    slope2 = -0.138

    n_components = 40

    torque_ft = fit_fourier(torque[:n], sampling_frequency, T, n_terms=n_components)
    angle_ft = fit_fourier(angle[:n], sampling_frequency, T, n_terms=n_components)
    slope_ft = fit_fourier(slope1 * np.sin(angle[:n]) + slope2 * np.cos(angle[:n]),
                           sampling_frequency,
                           T,
                           n_terms=n_components)

    torque_ft.ti = 0.01
    angle_ft.ti = 0.01
    slope_ft.ti = 0.01

    x, residuals, rank, s = np.linalg.lstsq(np.c_[
                                                torque_ft.to_array(),
                                                slope_ft.to_array(),
                                                angle_ft.to_array(),
                                                angle_ft.ddt.to_array()
                                            ],
                                            angle_ft.ddt.ddt.to_array()[:, None])

    # R squared value
    r2 = 1 - residuals / (len(angle_ft.ddt.ddt.to_array()) * angle_ft.ddt.ddt.to_array().var())