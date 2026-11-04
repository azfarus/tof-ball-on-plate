"""Gymnasium environment: balance the ball with the 3 plate servos.

Action = 3 servo angles / platform.range_deg in [-1, 1] (0 = flat).
Observation = EKF ball estimate, raw ToF readings, last action, target
(+ servo feedback and plate pose if enabled). See config/env.toml.

The plate has no rim in the env (env.overrides): a ball whose centre passes
the edge falls off, the step returns -escape_penalty and the episode ends.

All config values are passable: BallOnPlateEnv(overrides={"section.key": v}).
Headless by default; render_mode="human" opens the MuJoCo viewer.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config
from .sim import Sim

TOF_FIELDS = ["valid", "range_m", "sigma_m", "signal", "ambient", "yaw_rad", "age_s"]


class BallOnPlateEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, overrides=None, render_mode=None, config_dir=config.CONFIG_DIR):
        base = config.load_all(None, config_dir)
        self.base = config.apply(base, base["env"].get("overrides", {}))
        self.base = config.apply(self.base, overrides)
        self.render_mode = render_mode
        self._view = None
        self._build(self.base)

    def _build(self, cfg):
        self.cfg, self.e = cfg, cfg["env"]
        self.sim = Sim(cfg, interference=self.e["interference"])
        if self.e["servo_feedback"] and self.sim.leg_spec["feedback"] == "none":
            raise ValueError(f"servo_feedback needs a feedback servo, not "
                             f"{self.sim.leg_spec['name']} (try MG996R_FB)")
        self.q_max = np.radians(cfg["platform"]["range_deg"])
        self.half = cfg["plate"]["size_m"] / 2
        self.n_sub = int(round(1.0 / self.e["control_hz"] / self.sim.dt))
        self.max_steps = int(round(self.e["episode_s"] * self.e["control_hz"]))
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        box = lambda *shape: spaces.Box(-np.inf, np.inf, shape, np.float32)
        obs = {"ball_est": box(5), "tof": box(self.sim.n, self.e["tof_history"], len(TOF_FIELDS)),
               "servo_cmd": box(3), "target": box(2)}
        if self.e["servo_feedback"]:
            obs["servo_fb"] = box(3)
        if self.e["plate_pose"]:
            obs["plate"] = box(3)
        self.observation_space = spaces.Dict(obs)

    # ---- gym API -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rnd = self.base["env"]["randomize"]
        if options.get("overrides") or rnd:
            draw = {k: float(self.np_random.uniform(lo, hi)) for k, (lo, hi) in rnd.items()}
            self.close_view()
            self._build(config.apply(self.base, {**draw, **options.get("overrides", {})}))
        e, u = self.e, lambda a, n=2: self.np_random.uniform(-a, a, n)
        st = e["start"]
        z = float(self.np_random.uniform(0, st["drop_height_m"])) if st["drop_height_m"] > 0 else None
        self.target = options.get("target", u(st["target_xy_m"]))
        self.sim.reset(ball_xy=options.get("ball_xy", u(st["ball_xy_m"])),
                       ball_v=options.get("ball_v", u(st["ball_v_mps"])), ball_z=z,
                       forces=e["disturb"], horizon=e["episode_s"],
                       seed=int(self.np_random.integers(2**31)))
        self.hist = np.zeros((self.sim.n, e["tof_history"], len(TOF_FIELDS)), np.float32)
        self.action = np.zeros(3, np.float32)
        self.steps = 0
        return self._obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        rate = a - self.action
        self.action = a
        self.sim.send_legs(a * self.q_max)
        self.sim.new = []
        for _ in range(self.n_sub):
            self.sim.step()
            if self.sim.escaped:
                break
        self._push_readings(self.sim.new)
        self.steps += 1
        terminated = self.sim.escaped
        truncated = self.steps >= self.max_steps and not terminated
        r = self._reward(rate, terminated)
        if self.render_mode == "human":
            self.render()
        return self._obs(), r, terminated, truncated, self._info()

    def render(self):
        if self.render_mode != "human":
            return
        from . import viewer
        if self._view is None:
            self._view = viewer.open_viewer(self.sim.m, self.sim.d, config.load("viewer"))
        viewer.draw(self._view, self.sim)

    def close_view(self):
        if self._view is not None:
            self._view["v"].close()
            self._view = None

    def close(self):
        self.close_view()

    # ---- helpers -----------------------------------------------------------
    def _push_readings(self, readings):
        """Shift each sensor's history and add the new readings (newest first)."""
        t = self.sim.d.time
        self.hist[:, :, -1] += self.n_sub * self.sim.dt  # age
        s_ref = self.sim.s_ref
        for rd in readings:
            h = self.hist[rd["sensor"]]
            h[1:] = h[:-1].copy()
            ok = rd["valid"]
            h[0] = [float(ok), rd["range"] if ok else 0.0, rd["sigma"] if ok else 0.0,
                    rd["signal"] / s_ref, rd["ambient"] / s_ref, rd["yaw"], t - rd["t"]]

    def _obs(self):
        x = self.sim.trk.x
        o = {"ball_est": np.array([*x, float(self.sim.trk.ok)], np.float32),
             "tof": self.hist.copy(), "servo_cmd": self.action.copy(),
             "target": np.asarray(self.target, np.float32)}
        if self.e["servo_feedback"]:
            o["servo_fb"] = self.sim.leg_feedback().astype(np.float32)
        if self.e["plate_pose"]:
            o["plate"] = np.array(self.sim.plate_pose(), np.float32)
        return o

    def _reward(self, rate, terminated):
        w = self.e["reward"]
        if terminated:
            return -float(w["escape_penalty"])
        p, v = self.sim.ball()
        if self.e["reward_source"] == "estimate":
            p, v = self.sim.trk.x[:2], self.sim.trk.x[2:]
        err = np.linalg.norm(p[:2] - self.target) / self.half
        return float(1.0 - w["w_pos"] * err - w["w_vel"] * np.linalg.norm(v[:2])
                     - w["w_rate"] * rate @ rate - w["w_act"] * self.action @ self.action)

    def _info(self):
        p, v = self.sim.ball()
        return {"ball_xy": p[:2].copy(), "ball_v": v[:2].copy(), "est_xy": self.sim.trk.x[:2].copy(),
                "tracked": self.sim.trk.ok, "err_m": float(np.linalg.norm(p[:2] - self.target)),
                "escaped": self.sim.escaped, "plate_pose": np.array(self.sim.plate_pose()),
                "servo_saturated": self.sim.legs.saturated.copy(), "t": self.sim.d.time}


def make_vec(n, overrides=None, asynchronous=True):
    """n headless envs, one process each (gymnasium vector API)."""
    fns = [lambda: BallOnPlateEnv(overrides) for _ in range(n)]
    if not asynchronous:
        return gym.vector.SyncVectorEnv(fns)
    return gym.vector.AsyncVectorEnv(fns, context="forkserver")
