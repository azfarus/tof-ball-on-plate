"""EKF ball tracker in plate frame. State [x, y, vx, vy].

Each valid reading gives range (to ball surface) and the beam's mean yaw.
Range + R is the distance to the ball centre; corners at different places
make the fused estimate a trilateration. Bearing is weak (beam ~20 deg wide).
"""
import numpy as np

from .tof import rx_half


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Tracker:
    def __init__(self, cfg, sensor, corners, ball_r, background, limit):
        self.cfg, self.sensor = cfg, sensor
        self.corners, self.R, self.bg = corners, ball_r, background
        self.limit = limit  # max |x|, |y| of the ball centre (rim)
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.ok = False
        self.t_last = -np.inf

    def _clamp(self):
        """Keep the ball inside the rim. Removes the mirror solution of
        two-range trilateration, which lies outside the plate."""
        for k in range(2):
            if abs(self.x[k]) > self.limit:
                self.x[k] = np.sign(self.x[k]) * self.limit
                self.x[k + 2] = 0.0

    def predict(self, dt, acc_xy):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        self.x[:2] += 0.5 * acc_xy * dt * dt
        self.x[2:] += acc_xy * dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.P = F @ self.P @ F.T + G @ G.T * self.cfg["process_accel_sigma"] ** 2
        self._clamp()

    def _init(self, s, z, t):
        r, b = z
        self.x = np.array([s[0] + r * np.cos(b), s[1] + r * np.sin(b), 0, 0])
        p, v = self.cfg["init_pos_sigma_m"], self.cfg["init_vel_sigma_mps"]
        self.P = np.diag([p * p, p * p, v * v, v * v])
        self.ok, self.t_last = True, t
        self._clamp()

    def update(self, rd, t):
        """Fuse one reading. Returns a status string."""
        if not rd["valid"]:
            return "invalid"
        bg = self.bg[rd["sensor"]][rd["slot"]]
        if np.isfinite(bg) and rd["range"] > bg - self.cfg["bg_margin_m"]:
            return "background"
        s = self.corners[rd["sensor"]]
        z = np.array([rd["range"] + self.R, rd["yaw"]])
        if self.ok and t - self.t_last > self.cfg["lost_after_s"]:
            self.ok = False
        if not self.ok:
            self._init(s, z, t)
            return "init"
        dx = self.x[:2] - s
        rho = max(np.linalg.norm(dx), self.R * 1.01)
        H = np.zeros((2, 4))
        H[0, :2] = dx / rho
        H[1, :2] = [-dx[1] / rho**2, dx[0] / rho**2]
        y = z - [rho, np.arctan2(dx[1], dx[0])]
        y[1] = wrap(y[1])
        sig_r = max(rd["sigma"], self.cfg["range_sigma_floor_m"])
        # Single-zone ToF gives no direction: the ball can be anywhere in the
        # cone (plus its own angular size). Sweep smear is left out on purpose:
        # returns are weighted to the bright beam centre, and on sim data this
        # sigma already covers 81% of bearing errors at 1 sigma (ideal 68%).
        sig_b = (rx_half(self.sensor, rho) + np.arcsin(self.R / rho)) / np.sqrt(3)
        S = H @ self.P @ H.T + np.diag([sig_r**2, sig_b**2])
        if y @ np.linalg.solve(S, y) > self.cfg["gate_chi2"]:
            return "gated"
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self._clamp()
        self.t_last = t
        return "accepted"
