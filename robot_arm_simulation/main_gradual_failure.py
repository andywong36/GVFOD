from multiprocessing import Pool
import os
import sys
from time import time as curr_time

import numpy as np
from scipy.interpolate import interp1d

from controller import PIDControlRobotArm, DisturbRobotArm
from data.dataloader import get_robot_arm_data

# The times at which the parameters are at the stated values
# periods = [start, mid, end]
# periods = [0, 2, 3]


tol_limits = {"rtol": 1E-3, "atol": 1E-4}


class ChangeBaseT(PIDControlRobotArm):
    def __init__(self, startT, midT, endT, *args, **kwargs):
        """
        Generates failure data with a sharp L-shaped change in the static belt tension.
        """
        self._f_interp = interp1d(periods, [startT, midT, endT])

        super().__init__(*args, **kwargs)

    def get_data(self):
        """
        Get data from the robot arm, corrupted by noise

        Returns:
            data of shape (3, n) for the robot arm

        """
        import numpy as np
        from scipy import integrate

        t_eval = np.arange(0, periods[-1] * self.period, self.period / 2000)

        results = integrate.solve_ivp(
            self.ode,
            t_span=(0, periods[-1] * self.period),
            y0=self.y0,
            t_eval=t_eval,
            method="LSODA",
            jac=self.J,
            **tol_limits,
        )
        results.param_values = np.zeros(results.y.shape[1])
        results.tensions = np.zeros(results.y.shape[1])
        results.torques = np.zeros(results.y.shape[1])
        results.arm_position = results.y[4, :]
        for i, t in enumerate(t_eval):
            self.base_T = self.baseT_setpoint(t)
            self.y, self.dy, self.E = results.y[4:7, i]
            results.param_values[i] = self.base_T
            results.tensions[i] = self.tensions(results.y[0:6, i])[2]
            results.torques[i] = self.torque(t)
        return np.vstack([
            results.param_values,
            results.arm_position,
            results.torques,
            results.tensions,
        ]).T

    def baseT_setpoint(self, time):
        """ Returns the base tension in the robot arm.

        For periods 0 to 2000, it drops from start (155N) to (145N).
        For periods from 2000 to 2500, it drops from (145N) to (135N).
        Args:
            time: the current time

        Returns:
            the tension to use
        """

        return self._f_interp(time / self.period)

    def ode(self, time, y):
        self.base_T = self.baseT_setpoint(time)
        return super().ode(time, y)

    def J(self, time, y):
        self.base_T = self.baseT_setpoint(time)
        return super().J(time, y)


class ChangeEA(PIDControlRobotArm):
    def __init__(self, startEA, midEA, endEA, *args, **kwargs):
        """
        Generates failure data with a sharp L-shaped change in the belt stiffness.
        """
        self._f_interp = interp1d(periods, [startEA, midEA, endEA])

        super().__init__(*args, **kwargs)
        # self.k_1, self.k_2, self.k_3 = (self.EA / self.l_1, self.EA / self.l_2, self.EA / (self.l_3 * (2 / 3)))
        self.update_k(0)

    def get_data(self):
        """
        Get data from the robot arm, corrupted by noise

        Returns:
            data of shape (3, n) for the robot arm

        """
        import numpy as np
        from scipy import integrate

        t_eval = np.arange(0, periods[-1] * self.period, self.period / 2000)

        results = integrate.solve_ivp(
            self.ode,
            t_span=(0, periods[-1] * self.period),
            y0=self.y0,
            t_eval=t_eval,
            method="LSODA",
            jac=self.J,
            **tol_limits,
        )
        results.param_values = np.zeros(results.y.shape[1])
        results.tensions = np.zeros(results.y.shape[1])
        results.torques = np.zeros(results.y.shape[1])
        results.arm_position = results.y[4, :]
        for i, t in enumerate(t_eval):
            self.update_k(t)
            self.y, self.dy, self.E = results.y[4:7, i]
            results.param_values[i] = self.EA
            results.tensions[i] = self.tensions(results.y[0:6, i])[2]
            results.torques[i] = self.torque(t)
        return np.vstack([
            results.param_values,
            results.arm_position,
            results.torques,
            results.tensions,
        ]).T

    def EA_setpoint(self, time):
        """ Returns the base tension in the robot arm.

        For periods 0 to 2000, it drops from start (155N) to (145N).
        For periods from 2000 to 2500, it drops from (145N) to (135N).
        Args:
            time: the current time

        Returns:
            the tension to use
        """

        return self._f_interp(time / self.period)

    def update_k(self, t):
        self.EA = self.EA_setpoint(t)
        self.k_1, self.k_2, self.k_3 = (self.EA / self.l_1, self.EA / self.l_2, self.EA / (self.l_3 * (2 / 3)))

    def ode(self, time, y):
        self.update_k(time)
        return super().ode(time, y)

    def J(self, time, y):
        self.update_k(time)
        return super().J(time, y)


class Changef2a(PIDControlRobotArm):
    def __init__(self, startf2a, midf2a, endf2a, *args, **kwargs):
        """
        Generates failure data with a sharp L-shaped change in the static belt tension.
        """

        # Set function that maps from period -> friction
        self._f_interp = interp1d(periods, [startf2a, midf2a, endf2a])

        super().__init__(*args, **kwargs)

    def get_data(self):
        """
        Get data from the robot arm, corrupted by noise

        Returns:
            data of shape (3, n) for the robot arm

        """
        import numpy as np
        from scipy import integrate

        t_eval = np.arange(0, periods[-1] * self.period, self.period / 2000)

        results = integrate.solve_ivp(
            self.ode,
            t_span=(0, periods[-1] * self.period),
            y0=self.y0,
            t_eval=t_eval,
            method="LSODA",
            jac=self.J,
            **tol_limits,
        )
        results.param_values = np.zeros(results.y.shape[1])
        results.tensions = np.zeros(results.y.shape[1])
        results.torques = np.zeros(results.y.shape[1])
        results.arm_position = results.y[4, :]
        for i, t in enumerate(t_eval):
            self.f2a = self.f2a_setpoint(t)
            self.y, self.dy, self.E = results.y[4:7, i]
            results.param_values[i] = self.f2a
            results.tensions[i] = self.tensions(results.y[0:6, i])[2]
            results.torques[i] = self.torque(t)
        return np.vstack([
            results.param_values,
            results.arm_position,
            results.torques,
            results.tensions,
        ]).T

    def f2a_setpoint(self, time):
        """ Returns the base tension in the robot arm.

        For periods 0 to 2000, it drops from start (155N) to (145N).
        For periods from 2000 to 2500, it drops from (145N) to (135N).
        Args:
            time: the current time

        Returns:
            the tension to use
        """

        return self._f_interp(time / self.period)

    def ode(self, time, y):
        self.f2a = self.f2a_setpoint(time)
        return super().ode(time, y)

    def J(self, time, y):
        self.f2a = self.f2a_setpoint(time)
        return super().J(time, y)


class ChangeSlope1(PIDControlRobotArm):
    def __init__(self, start_slope1, mid_slope1, end_slope1, *args, **kwargs):
        """
        Generates failure data with a sharp L-shaped change in the parameter slope1.
        """
        self._f_interp = interp1d(periods, [start_slope1, mid_slope1, end_slope1])

        super().__init__(*args, **kwargs)

    def get_data(self):
        """
        Get data from the robot arm, corrupted by noise

        Returns:
            data of shape (3, n) for the robot arm

        """
        import numpy as np
        from scipy import integrate

        t_eval = np.arange(0, periods[-1] * self.period, self.period / 2000)

        results = integrate.solve_ivp(
            self.ode,
            t_span=(0, periods[-1] * self.period),
            y0=self.y0,
            t_eval=t_eval,
            method="LSODA",
            jac=self.J,
            **tol_limits,
        )
        results.param_values = np.zeros(results.y.shape[1])
        results.tensions = np.zeros(results.y.shape[1])
        results.torques = np.zeros(results.y.shape[1])
        results.arm_position = results.y[4, :]
        for i, t in enumerate(t_eval):
            self.slope1 = self.slope1_setpoint(t)
            self.y, self.dy, self.E = results.y[4:7, i]
            results.param_values[i] = self.slope1
            results.tensions[i] = self.tensions(results.y[0:6, i])[2]
            results.torques[i] = self.torque(t)
        return np.vstack([
            results.param_values,
            results.arm_position,
            results.torques,
            results.tensions,
        ]).T

    def slope1_setpoint(self, time):
        """ Returns the base tension in the robot arm.

        For periods 0 to 2000, it drops from start (155N) to (145N).
        For periods from 2000 to 2500, it drops from (145N) to (135N).
        Args:
            time: the current time

        Returns:
            the tension to use
        """

        return self._f_interp(time / self.period)

    def ode(self, time, y):
        self.slope1 = self.slope1_setpoint(time)
        return super().ode(time, y)

    def J(self, time, y):
        self.slope1 = self.slope1_setpoint(time)
        return super().J(time, y)


def run_baseT_exp():
    start = curr_time()
    baseTgen = ChangeBaseT(155, 145, 135, **ChangeBaseT.optimized_params)
    baseTgen.gradual_failure_data = baseTgen.get_data()
    np.save(f"robot_arm_simulation//gradual_failure//baseT_{periods[1]}_{periods[2]}.npy",
            baseTgen.gradual_failure_data)
    print(f"It took {curr_time() - start} seconds to generate data for {n_timesteps} timesteps")
    return baseTgen.gradual_failure_data


def run_EA_exp():
    start = curr_time()
    EAgen = ChangeEA(40000, 30000, 20000, **ChangeEA.optimized_params)
    EAgen.gradual_failure_data = EAgen.get_data()
    np.save(f"robot_arm_simulation//gradual_failure//EA_{periods[1]}_{periods[2]}.npy", EAgen.gradual_failure_data)
    print(f"It took {curr_time() - start} seconds to generate data for {n_timesteps} timesteps")
    return EAgen.gradual_failure_data


def run_f2a_exp():
    start = curr_time()
    f2agen = Changef2a(5E-5, 1E-4, 1.5E-4, **Changef2a.optimized_params)
    f2agen.gradual_failure_data = f2agen.get_data()
    np.save(f"robot_arm_simulation//gradual_failure//f2a_{periods[1]}_{periods[2]}.npy", f2agen.gradual_failure_data)
    print(f"It took {curr_time() - start} seconds to generate data for {n_timesteps} timesteps")
    return f2agen.gradual_failure_data


def run_slope1_exp():
    start = curr_time()
    Slope1gen = ChangeSlope1(0.21, 0.22, 0.23, **ChangeSlope1.optimized_params)
    Slope1gen.gradual_failure_data = Slope1gen.get_data()
    np.save(f"robot_arm_simulation//gradual_failure//slope1_{periods[1]}_{periods[2]}.npy",
            Slope1gen.gradual_failure_data)
    print(f"It took {curr_time() - start} seconds to generate data for {n_timesteps} timesteps")
    return Slope1gen.gradual_failure_data


def plot_gradual_failure(npy_file):
    import matplotlib.pyplot as plt

    filename = os.path.basename(npy_file)
    failure_type = filename.split("_")[0]
    steps = int(filename.split("_")[-1].split(".")[0]) * 2000
    a = np.load(npy_file)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    X, y = get_robot_arm_data()
    X[:, :2000] *= 2 * np.pi / 360
    for i in range(3):
        axs[i + 1].plot(np.arange(steps, dtype=float) / 2000 * PIDControlRobotArm.period,
                        X[10:10 + steps // 2000, i * 2000: (i + 1) * 2000].flatten(order="C"), label="Empirical",c="red")
    for i in range(4):
        axs[i].plot(np.arange(steps, dtype=float) / 2000 * PIDControlRobotArm.period, a[:, i],
                    label="Simulator", c="blue")

    # Plot some empirical data

    for i in range(4):
        axs[i].legend()

    # axs[0].set_title("Simulating Gradual Non-stationary Failure")
    axs[0].set(ylabel="Static Tension (N)")
    axs[1].set(ylabel="Arm Position (rad)")
    axs[2].set(ylabel="Torque (Nm)")
    axs[3].set(ylabel="Tension (N)", xlabel="Time (s)")

    plt.tight_layout()




if __name__ == "__main__":

    if os.name == "posix":
        os.nice(20)

        periods = list(map(int, sys.argv[1:4]))
        n_timesteps = periods[-1] * 2000

        results = []
        p = Pool(processes=4)
        for f in [
            run_baseT_exp,
            # run_EA_exp,
            # run_f2a_exp,
            # run_slope1_exp
        ]:
            results.append(p.apply_async(f))
        p.close()
        for r in results:
            r.get()
        p.join()
    else:
        raise OSError('This only works on linux')