import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class PIDController:
    def __init__(self, K_p, T_i=None, T_d=0):
        if T_i is not None:
            raise NotImplementedError()
        if T_d != 0:
            raise NotImplementedError()

        self.K_p = K_p

        data = pd.read_csv(r"test_data.csv")
        data.columns = ["Time", "Run", "Direction", "Angle", "Torque", "Tension"]

        self.angle_a = data["Angle"][data["Time"].between(18, 19.9)].mean()
        self.angle_b = data["Angle"][data["Time"].between(8, 9.9)].mean()

    def setpoint(self, time):
        t = time % 20  # The time since the beginning of the period.
        c = (0, 0.5, 4, 10.5, 14, 20)
        if c[0] <= t < c[1]:
            return self.angle_a
        elif c[1] <= t < c[2]:
            itrvl = c[2] - c[1]
            return (c[2] - t) / itrvl * self.angle_a + (t - c[1]) / itrvl * self.angle_b
        elif c[2] <= t < c[3]:
            return self.angle_b
        elif c[3] <= t < c[4]:
            itrvl = c[4] - c[3]
            return (c[4] - t) / itrvl * self.angle_b + (t - c[3]) / itrvl * self.angle_a
        elif c[4] <= t < c[5]:
            return self.angle_a
        else:
            raise ValueError("Invalid time in setpoint (time)")

    def control(self, t, PV):
        error = self.setpoint(t) - PV
        return self.K_p * error
