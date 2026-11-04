"""Non-learned baseline: cascade PID from the EKF ball estimate to servo angles.

Wanted ball accel a = -kp e - kd v - ki int(e). The plate tilt that gives it
(a_x = k g sin(pitch) cos(roll), a_y = -k g sin(roll)) goes through the
platform IK to three servo angles, returned as an env action in [-1, 1].
"""
import numpy as np

from bop_env import legs


class PID:
    def __init__(self, env_cfg, ctrl):
        self.cfg, self.c = env_cfg, ctrl
        self.q_max = np.radians(env_cfg["platform"]["range_deg"])
        self.dt = 1.0 / env_cfg["env"]["control_hz"]
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)

    def act(self, obs):
        x, y, vx, vy, ok = obs["ball_est"]
        if ok < 0.5:
            return np.zeros(3, np.float32)  # no estimate yet: hold flat
        c = self.c
        e = np.array([x, y]) - obs["target"]
        self.integral = np.clip(self.integral + e * self.dt, -0.05, 0.05)
        a = -c["kp"] * e - c["kd"] * np.array([vx, vy]) - c["ki"] * self.integral
        kg = c["roll_factor"] * 9.81
        lim = np.radians(c["max_tilt_deg"])
        roll = np.clip(-np.arcsin(np.clip(a[1] / kg, -1, 1)), -lim, lim)
        pitch = np.clip(np.arcsin(np.clip(a[0] / (kg * np.cos(roll)), -1, 1)), -lim, lim)
        q, _ = legs.ik(self.cfg["platform"], self.cfg["plate"], roll, pitch, 0.0)
        return (q / self.q_max).astype(np.float32)
