import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class PIDControlRobotArm:
    def __init__(self, K_p, T_i, T_d):
        self.K_p = K_p
        self.T_i = T_i
        self.T_d = T_d

        data = pd.read_csv(r"test_data.csv")
        data.columns = ["Time", "Run", "Direction", "Angle", "Torque", "Tension"]

        self.angle_a = data["Angle"][data["Time"].between(18, 19.9)].mean()
        self.angle_b = data["Angle"][data["Time"].between(8, 9.9)].mean()

    def setpoint(self, time):
        # Trapezoidal velocity profile.
        # 2 parameters: max_speed, and accel
        max_speed = 0.56
        accel = 0.425

        t = time % 20  # The time since the beginning of the period.
        t_accel = max_speed / accel
        angle_diff_acc = 0.5 * accel * t_accel ** 2
        t_max_speed = (self.angle_b - self.angle_a - 2 * angle_diff_acc) / max_speed
        t_total_moving = t_accel + t_max_speed + t_accel

        times = (0, t_accel, t_max_speed, t_accel, 10 - t_total_moving,
                 t_accel, t_max_speed, t_accel, 10 - t_total_moving)
        c = np.cumsum(times)
        if c[0] <= t < c[1]:
            return 0.5 * accel * t ** 2 + self.angle_a
        elif t < c[2]:
            return (t - c[1]) * max_speed + self.angle_a + angle_diff_acc
        elif t < c[3]:
            return self.angle_b - 0.5 * accel * (t - c[3]) ** 2
        elif t < c[4]:
            return self.angle_b
        elif t < c[5]:
            return self.angle_b - 0.5 * accel * (t - c[4]) ** 2
        elif t < c[6]:
            return self.angle_b - angle_diff_acc - (t - c[5]) * max_speed
        elif t < c[7]:
            return self.angle_a + 0.5 * accel * (t - c[7]) ** 2
        elif t < c[8]:
            return self.angle_a
        else:
            raise ValueError("Unspecified time in setpoint")

    def control(self, t, interr, err, derr):
        return self.K_p * (err + 1 / self.T_i * interr + self.T_d * derr)
