"""Core simulation: plate on 3 servos, ball, scanning ToF sensors, EKF tracker.

One Sim = one compiled MuJoCo model. step() advances one physics step and
advance(seconds) many. The host commands the plate servos with send_legs()
(servo angles, rad, 0 = flat) or command_pose() (roll, pitch, heave via IK).
The scan servos follow the scan schedule on their own.

Used by bop_env.env (RL environment) and the tools/ scripts.
"""
import numpy as np
import mujoco

from . import disturb, legs, model, scan, servo, tof
from .tracker import Tracker, wrap


class Sim:
    def __init__(self, cfg, interference=None, keep_log=False):
        self.cfg = cfg
        self.plate, self.sc, self.sn, self.pf = (cfg["plate"], cfg["scan"],
                                                 cfg["sensor"], cfg["platform"])
        pf, sc = self.pf, self.sc
        self.leg_spec = servo.spec(cfg["servos"], pf["servo_model"], pf["supply_v"])
        self.scan_spec = (servo.spec(cfg["servos"], sc["servo_model"], sc["supply_v"])
                          if sc["actuator"] == "servo" else None)
        self.m, self.d, self.ids = model.build(cfg, self.leg_spec, self.scan_spec)
        self.n = len(sc["corners"])
        self.legs = servo.ServoBank(self.leg_spec, 3, pf["servo_update_hz"],
                                    pf["pwm_resolution_us"], pf["trajectory_speed_rps"])
        self.scan_bank = (servo.ServoBank(self.scan_spec, self.n, sc["servo_update_hz"],
                                          pf["pwm_resolution_us"]) if self.scan_spec else None)
        self.interference = sc["interference"] if interference is None else interference
        self.keep_log = keep_log
        self.rays = tof.make_rays(self.sn)
        self.s_ref = tof.reference_signal(self.sn, self.rays)
        self.corners = model.corner_xy(self.plate, sc)
        self.R_ball = self.plate["ball_radius_m"]
        rim = self.plate["rim_height_m"] > 0
        self.limit = self.plate["size_m"] / 2 - (self.plate["rim_width_m"] + self.R_ball) * rim
        self.dt = self.m.opt.timestep
        self.sub_every = int(round(sc["substep_ms"] / 1000 / self.dt))
        self.n_slots = 2 * int(round(sc["sector_deg"] / sc["step_deg"]))
        self.bg = None
        self.bg = self._background()

    # ---- setup -------------------------------------------------------------
    def reset(self, ball_xy=(0.0, 0.0), ball_v=(0.0, 0.0), ball_z=None, tilt=(0.0, 0.0),
              heave=0.0, forces=None, horizon=10.0, seed=None):
        """Plate at a pose (legs consistent), ball placed, tracker cleared.
        forces: dict with kicks_per_s, kick_force_n, kick_ms, push_sigma_n,
        push_tau_s (see disturb.py); horizon = time span to draw kicks over."""
        m, d, ids = self.m, self.d, self.ids
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(m, d)
        q, knee = legs.ik(self.pf, self.plate, tilt[0], tilt[1], heave)
        d.qpos[ids["leg_qpos"]] = q
        d.qpos[ids["knee_qpos"]] = knee
        a = ids["plate_qpos"]
        d.qpos[a:a + 3] = [0, 0, legs.neutral_height(self.pf, self.plate) + heave]
        mujoco.mju_mat2Quat(d.qpos[a + 3:a + 7], legs.tilt_matrix(tilt[0], tilt[1]).ravel())
        self.legs.reset(q)
        if self.scan_bank:
            q0 = [scan.servo_angle(0.0, i, self.sc, self.sn) for i in range(self.n)]
            d.qpos[ids["scan_qpos"]] = q0
            self.scan_bank.reset(q0)
        model.place_ball(m, d, ids, self.plate, ball_xy, ball_v, ball_z)
        self.forces = forces or {}
        self.kicks = disturb.kick_list(self.forces, self.rng, horizon)
        self.push = np.zeros(2)
        self.noise = self.bg is not None  # background map is built noise free
        self.offsets = (self.rng.normal(0, self.sn["offset_sigma_m"], self.n) if self.noise
                        else np.zeros(self.n))
        self.trk = Tracker(self.cfg["tracker"], self.sn, self.corners, self.R_ball,
                           self.bg, self.limit)
        self.accs = [tof.new_acc(self.sn) for _ in range(self.n)]
        self.wins = [None] * self.n
        self.k = 0
        self.escaped = False
        self.n_saturated = 0
        self.new, self.log = [], []
        self.status = ["none"] * self.n
        self.last = [None] * self.n  # newest reading per sensor

    def _background(self):
        """Empty-plate range per sensor per scan slot (ball parked, flat, no noise)."""
        self.reset(ball_xy=(0, 0), ball_z=5.0)
        cycle = self.n_slots * scan.reading_time(self.sn)
        bg = np.full((self.n, self.n_slots), np.nan)
        for rd in self.advance(3 * cycle + 0.05):  # later cycles overwrite (servo settled)
            bg[rd["sensor"], rd["slot"]] = rd["range"] if rd["valid"] else np.nan
        return bg

    # ---- host commands -----------------------------------------------------
    def send_legs(self, q):
        """Servo angles (rad) to send; the servos latch them at the next PWM frame."""
        self.legs.send(q)

    def command_pose(self, tilt, heave=0.0):
        self.legs.send(legs.ik(self.pf, self.plate, tilt[0], tilt[1], heave)[0])

    # ---- stepping ----------------------------------------------------------
    def step(self):
        m, d, ids = self.m, self.d, self.ids
        t = d.time
        d.ctrl[:3] = self.legs.torque(t, self.leg_angles(), d.qvel[ids["leg_dof"]])
        self.n_saturated += bool(self.legs.saturated.any())
        if self.scan_bank:
            self.scan_bank.send([scan.servo_angle(t, i, self.sc, self.sn) for i in range(self.n)])
            d.ctrl[3:3 + self.n] = self.scan_bank.torque(t, self.scan_angles(),
                                                         d.qvel[ids["scan_dof"]])
        R = d.xmat[ids["plate"]].reshape(3, 3)
        f = disturb.force_at(self.forces, t, self.dt, self.kicks, self.push, self.rng)
        d.xfrc_applied[ids["ball"], :3] = R @ np.array([f[0], f[1], 0.0])
        truth, _ = model.ball_plate_frame(d, ids)
        if np.abs(truth[:2]).max() > self.plate["size_m"] / 2 or truth[2] < 0:
            self.escaped = True  # centre past the edge (it tips off) or below the plate top
        if self.k % self.sub_every == self.sub_every // 2:  # middle of each sub-sample slice
            self._tof_tick(t, truth)
        g_plate = R.T @ np.array([0, 0, -9.81])
        self.trk.predict(self.dt, self.cfg["tracker"]["roll_factor"] * g_plate[:2])
        mujoco.mj_step(m, d)
        self.k += 1

    def advance(self, seconds):
        """Run for `seconds`. Returns the ToF readings finished meanwhile."""
        self.new = []
        for _ in range(int(round(seconds / self.dt))):
            self.step()
        return self.new

    def _tof_tick(self, t, truth):
        for i in range(self.n):
            w = scan.window(t, i, self.sc, self.sn)
            if self.wins[i] is not None and w != self.wins[i]:
                self._finish(i, t, truth)
            self.wins[i] = w
            self.accs[i]["yaw"] += self._bearing(t, i)
            self.accs[i]["beam"] = self.accs[i].get("beam", 0.0) + self._beam_yaw(t, i)
        tof.sample_tick(self.m, self.d, self.sn, self.rays, self.sensor_poses(t),
                        self.ids, self.accs, self.interference)

    def _bearing(self, t, i):
        """Beam yaw the tracker is told: commanded, or measured with feedback noise."""
        if self.scan_bank is None or self.sc["yaw_source"] == "commanded":
            return scan.yaw(t, i, self.sc, self.sn)
        noise = self.rng.normal(0, np.radians(self.scan_spec["feedback_noise_deg"])) * self.noise
        return self._beam_yaw(t, i) + noise

    def _finish(self, i, t, truth):
        acc = self.accs[i]
        rd = tof.finalize(self.sn, acc, self.s_ref, self.offsets[i],
                          self.rng if self.noise else None)
        rd.update(sensor=i, slot=scan.slot(self.wins[i], self.sc), t=t,
                  yaw=wrap(acc["yaw"] / acc["n"]), beam_yaw=wrap(acc["beam"] / acc["n"]))
        self.accs[i] = tof.new_acc(self.sn)
        if self.bg is not None:
            rd["status"] = self.trk.update(rd, t)
            v = truth[:2] - self.corners[i]
            rd["true_range"] = np.linalg.norm(v) - self.R_ball
            rd["true_bearing"] = np.arctan2(v[1], v[0])
            self.status[i] = rd["status"]
        self.last[i] = rd
        self.new.append(rd)
        if self.keep_log:
            self.log.append(rd)

    # ---- state -------------------------------------------------------------
    def _beam_yaw(self, t, i):
        """Actual plate-frame beam yaw of sensor i."""
        if self.scan_bank is None:
            return scan.yaw(t, i, self.sc, self.sn)
        return np.radians(model.BISECTOR_DEG[self.sc["corners"][i]]) + self.scan_angles()[i]

    def sensor_poses(self, t=None):
        d, ids = self.d, self.ids
        t = d.time if t is None else t
        R, p = d.xmat[ids["plate"]].reshape(3, 3), d.xpos[ids["plate"]]
        h, pitch = self.sc["mount_height_m"], np.radians(self.sc["pitch_deg"])
        return [tof.sensor_frame(R, p, self.corners[i], h, self._beam_yaw(t, i), pitch)
                for i in range(self.n)]

    def leg_angles(self):
        """Horn angles (rad) of the plate servos: motor + backlash."""
        return self.d.qpos[self.ids["leg_qpos"]] + self.d.qpos[self.ids["leg_lash"]]

    def scan_angles(self):
        return self.d.qpos[self.ids["scan_qpos"]] + self.d.qpos[self.ids["scan_lash"]]

    def leg_feedback(self):
        return self.legs.feedback(self.leg_angles(), self.rng)

    def ball(self):
        """True ball centre position and velocity in plate frame."""
        return model.ball_plate_frame(self.d, self.ids)

    def plate_pose(self):
        return legs.pose(self.d.xmat[self.ids["plate"]], self.d.xpos[self.ids["plate"]],
                         self.pf, self.plate)

    def in_cone(self, rd):
        """True if the true ball direction was inside the detection cone."""
        dist = rd["true_range"] + self.R_ball
        off = abs(wrap(rd["true_bearing"] - rd["beam_yaw"]))
        return off <= tof.rx_half(self.sn, dist) + np.arcsin(min(self.R_ball / dist, 1.0))
