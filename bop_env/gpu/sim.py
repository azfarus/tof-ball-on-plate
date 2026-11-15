"""Batched MuJoCo Warp simulation of the ball-on-plate system."""
import numpy as np
import warp as wp
import mujoco_warp as mjw
from gymnasium import spaces
from mujoco_warp._src.types import vec6f

from .. import config, disturb, legs, model, scan, servo, tof
from ..sim import Sim as CpuSim
from . import kernels, tof_kernels
from .obs import ObservationMixin, _rot




class WarpSim(ObservationMixin):
    """One GPU model holding `nworld` independent simulations."""

    def __init__(self, cfg, nworld, seed=0, device="cuda:0"):
        wp.set_device(device)
        self.cfg, self.b, self.seed, self.device = cfg, nworld, int(seed), device
        self.plate, self.sc, self.sn, self.pf, self.e = (
            cfg["plate"], cfg["scan"], cfg["sensor"], cfg["platform"], cfg["env"])
        self.leg_spec = servo.spec(cfg["servos"], self.pf["servo_model"], self.pf["supply_v"])
        self.scan_servo = self.sc["actuator"] == "servo"
        self.scan_spec = (servo.spec(cfg["servos"], self.sc["servo_model"], self.sc["supply_v"])
                          if self.scan_servo else None)
        self.mjm, _, self.ids = model.build(cfg, self.leg_spec, self.scan_spec)
        self.ns = len(self.sc["corners"])
        self.nray = int(self.sn["n_rays"])
        self.nbins = int(self.sn["max_range_m"] / self.sn["bin_m"]) + 1
        self.nslots = 2 * int(round(self.sc["sector_deg"] / self.sc["step_deg"]))
        self.rays = tof.make_rays(self.sn)
        self.ray_local = wp.array(np.asarray(self.rays["local"], np.float32), dtype=wp.vec3f)
        self.ray_power = wp.array(np.asarray(self.rays["power"], np.float32))
        self.ray_theta = wp.array(np.asarray(self.rays["theta"], np.float32))
        self.pulse = tof._kernel(self.sn)
        self.pulse_w = wp.array(np.asarray(self.pulse, np.float32))
        self.albedo = wp.array(np.asarray(self.ids["albedo"], np.float32))
        self.housing = wp.array(np.asarray(self.ids["housings"], np.int32))
        self.corners = model.corner_xy(self.plate, self.sc)
        cx = np.array([p[0] for p in self.corners], np.float32)
        cy = np.array([p[1] for p in self.corners], np.float32)
        self.corner_x, self.corner_y = wp.array(cx), wp.array(cy)
        self.bisector = np.array([model.BISECTOR_DEG[c] for c in self.sc["corners"]], np.float32)
        self.bisector_wp = wp.array(self.bisector)
        self.mount_height = float(self.sc["mount_height_m"])
        self.leg_qidx = wp.from_numpy(np.asarray(self.ids["leg_qpos"], np.int32))
        self.leg_lidx = wp.from_numpy(np.asarray(self.ids["leg_lash"], np.int32))
        self.leg_didx = wp.from_numpy(np.asarray(self.ids["leg_dof"], np.int32))
        self.scan_qidx = wp.from_numpy(np.asarray(self.ids.get("scan_qpos", []), np.int32))
        self.scan_lidx = wp.from_numpy(np.asarray(self.ids.get("scan_lash", []), np.int32))
        self.scan_didx = wp.from_numpy(np.asarray(self.ids.get("scan_dof", []), np.int32))
        self.s_ref = tof.reference_signal(self.sn, self.rays)
        self.leg_frame_dt = 1.0 / min(self.pf["servo_update_hz"], self.leg_spec["max_frame_hz"])
        self.scan_frame_dt = (1.0 / min(self.sc["servo_update_hz"], self.scan_spec["max_frame_hz"])
                              if self.scan_spec else 1.0)

        self.device_obj = wp.get_device(device)
        self.m = mjw.put_model(self.mjm)
        self.m.opt.warn_overflow = 0
        self.d = mjw.make_data(self.mjm, nworld=nworld, nconmax=64, njmax=256)
        self.rc = mjw.create_render_context(self.mjm, nworld=nworld, render_rgb=False,
                                            render_depth=False, enabled_geom_groups=[0, 1, 2])

        self.q_max = np.radians(self.pf["range_deg"])
        self.half = self.plate["size_m"] / 2
        self.limit = self.half - (self.plate["rim_width_m"] + self.plate["ball_radius_m"]) * (
            self.plate["rim_height_m"] > 0)
        self.dt = float(self.mjm.opt.timestep)
        self.control_dt = 1.0 / self.e["control_hz"]
        self.n_sub = int(round(self.control_dt / self.dt))
        self.sub_every = int(round(self.sc["substep_ms"] / 1000.0 / self.dt))
        self.max_steps = int(round(self.e["episode_s"] * self.e["control_hz"]))
        self.history = int(self.e["tof_history"])
        self.max_reads = 2 * self.ns + 4
        self.max_kicks = 64
        self.k = 0

        self._alloc()
        self._background()
        self.reset()
        self.graph = None
        self._capture()
        self.metadata = {"render_modes": []}
        self.render_mode = None
        box = lambda *shape: spaces.Box(-np.inf, np.inf, shape, np.float32)
        obs = {"ball_est": box(5), "tof": box(self.ns, self.history, 7),
               "servo_cmd": box(3), "target": box(2)}
        if self.e["servo_feedback"]:
            obs["servo_fb"] = box(3)
        if self.e["plate_pose"]:
            obs["plate"] = box(3)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    def _alloc(self):
        b, ns, nr = self.b, self.ns, self.nray
        self.sensor_origin = wp.zeros((b, ns), dtype=wp.vec3f)
        self.sensor_rot = wp.zeros((b, ns), dtype=wp.mat33f)
        self.sensor_beam = wp.zeros((b, ns), dtype=wp.float32)
        self.sensor_bearing = wp.zeros((b, ns), dtype=wp.float32)
        self.primary_pnt = wp.zeros((b, ns * nr), dtype=wp.vec3f)
        self.primary_vec = wp.zeros((b, ns * nr), dtype=wp.vec3f)
        self.primary_dist = wp.zeros((b, ns * nr), dtype=wp.float32)
        self.primary_geom = wp.zeros((b, ns * nr), dtype=wp.int32)
        self.primary_normal = wp.zeros((b, ns * nr), dtype=wp.vec3f)
        naux = ns * (ns - 1) * (nr + 1)
        self.naux = naux
        self.aux_pnt = wp.zeros((b, naux), dtype=wp.vec3f)
        self.aux_vec = wp.zeros((b, naux), dtype=wp.vec3f)
        self.aux_dist = wp.zeros((b, naux), dtype=wp.float32)
        self.aux_geom = wp.zeros((b, naux), dtype=wp.int32)
        self.aux_normal = wp.zeros((b, naux), dtype=wp.vec3f)
        self.aux_valid = wp.zeros((b, naux), dtype=wp.int32)

        self.hist = wp.zeros((b, ns, self.nbins), dtype=wp.float32)
        self.amb = wp.zeros((b, ns), dtype=wp.float32)
        self.acc_yaw = wp.zeros((b, ns), dtype=wp.float32)
        self.acc_beam = wp.zeros((b, ns), dtype=wp.float32)
        self.acc_n = wp.zeros((b, ns), dtype=wp.int32)
        self.wins = wp.full((b, ns), -999999, dtype=wp.int32)
        self.offsets = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_valid = wp.zeros((b, ns), dtype=wp.int32)
        self.rd_range = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_sigma = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_signal = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_ambient = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_yaw = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_beam = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_slot = wp.zeros((b, ns), dtype=wp.int32)
        self.rd_time = wp.zeros((b, ns), dtype=wp.float32)
        self.rd_sensor = wp.zeros((b, ns), dtype=wp.int32)
        self.finished = wp.zeros((b, ns), dtype=wp.int32)
        self.read_count = wp.zeros((b,), dtype=wp.int32)
        self.read_valid = wp.zeros((b, self.max_reads), dtype=wp.int32)
        self.read_sensor = wp.zeros((b, self.max_reads), dtype=wp.int32)
        self.read_range = wp.zeros((b, self.max_reads), dtype=wp.float32)
        self.read_sigma = wp.zeros((b, self.max_reads), dtype=wp.float32)
        self.read_signal = wp.zeros((b, self.max_reads), dtype=wp.float32)
        self.read_ambient = wp.zeros((b, self.max_reads), dtype=wp.float32)
        self.read_yaw = wp.zeros((b, self.max_reads), dtype=wp.float32)
        self.read_beam = wp.zeros((b, self.max_reads), dtype=wp.float32)
        self.read_slot = wp.zeros((b, self.max_reads), dtype=wp.int32)
        self.read_time = wp.zeros((b, self.max_reads), dtype=wp.float32)

        self.x = wp.zeros((b, 4), dtype=wp.float32)
        self.P = wp.zeros((b, 16), dtype=wp.float32)
        self.ok = wp.zeros((b,), dtype=wp.int32)
        self.tlast = wp.full((b,), -1e9, dtype=wp.float32)
        self.push = wp.zeros((b, 2), dtype=wp.float32)
        self.escaped = wp.zeros((b,), dtype=wp.int32)
        self.escape_t = wp.full((b,), -1.0, dtype=wp.float32)
        self.truth_pos = wp.zeros((b, 3), dtype=wp.float32)
        self.truth_vel = wp.zeros((b, 3), dtype=wp.float32)
        self.kicks = wp.zeros((b, self.max_kicks, 4), dtype=wp.float32)
        self.nkicks = wp.zeros((b,), dtype=wp.int32)
        self.leg_sent = wp.zeros((b, 3), dtype=wp.float32)
        self.leg_latched = wp.zeros((b, 3), dtype=wp.float32)
        self.leg_next = wp.zeros((b, 3), dtype=wp.float32)
        self.leg_sat = wp.zeros((b, 3), dtype=wp.int32)
        self.scan_latched = wp.zeros((b, self.ns), dtype=wp.float32)
        self.scan_next = wp.zeros((b, self.ns), dtype=wp.float32)
        self.scan_sat = wp.zeros((b, self.ns), dtype=wp.int32)
        self.steps = np.zeros((b,), np.int32)
        self.target = np.zeros((b, 2), np.float32)
        self.action = np.zeros((b, 3), np.float32)
        self.last_action = np.zeros((b, 3), np.float32)
        self.hist_obs = np.zeros((b, self.ns, self.history, 7), np.float32)
        self.be = wp.from_numpy(np.repeat(
            np.asarray(self.ids["sensors"], np.int32), self.nray).astype(np.int32))
        body = np.repeat(np.asarray(self.ids["sensors"], np.int32), (self.ns - 1) * (self.nray + 1))
        self.aux_be = wp.from_numpy(body.astype(np.int32))

    def _background(self):
        cpu = CpuSim(self.cfg, interference=self.e["interference"])
        self.bg_host = np.asarray(cpu.bg, np.float32)
        self.bg = wp.array(self.bg_host)

    def _draw(self, rng):
        st = self.e["start"]
        u = lambda a, n=2: rng.uniform(-a, a, n)
        z = rng.uniform(0, st["drop_height_m"]) if st["drop_height_m"] > 0 else None
        return (u(st["ball_xy_m"]), u(st["ball_v_mps"]),
                u(st["target_xy_m"]), z)

    def reset(self, indices=None, seed=None, options=None):
        idx = np.arange(self.b) if indices is None else np.asarray(indices, np.int32)
        if len(idx) == 0:
            return self.obs(), self._plain_info()
        if seed is None:
            seed = self.seed
        seeds = seed + idx if np.isscalar(seed) else np.asarray(seed)
        qpos = self.d.qpos.numpy()
        qvel = self.d.qvel.numpy()
        tm = self.d.time.numpy()
        ctrl = self.d.ctrl.numpy()
        qacc = self.d.qacc.numpy()
        qacc_w = self.d.qacc_warmstart.numpy()
        hist = self.hist_obs.copy()
        kicks = self.kicks.numpy()
        nkicks = self.nkicks.numpy()
        offs = self.offsets.numpy()
        x = self.x.numpy(); P = self.P.numpy(); ok = self.ok.numpy(); tl = self.tlast.numpy()
        push = self.push.numpy(); esc = self.escaped.numpy(); et = self.escape_t.numpy()
        legsent = self.leg_sent.numpy(); leglatch = self.leg_latched.numpy()
        legnext = self.leg_next.numpy(); legsat = self.leg_sat.numpy()
        scanlatch = self.scan_latched.numpy(); scannext = self.scan_next.numpy()
        scansat = self.scan_sat.numpy()
        wins = self.wins.numpy(); accn = self.acc_n.numpy()
        accy = self.acc_yaw.numpy(); accb = self.acc_beam.numpy(); amb = self.amb.numpy()
        hist_gpu = self.hist.numpy(); finished = self.finished.numpy()
        p0 = self.pf; plate = self.plate
        neutral = legs.neutral_height(p0, plate)
        for n, w in enumerate(idx):
            rng = np.random.default_rng(int(seeds[n]))
            xy, vxy, target, z = self._draw(rng)
            self.target[w] = target
            qpos[w] = 0.0; qvel[w] = 0.0
            qpos[w, self.ids["plate_qpos"]:self.ids["plate_qpos"] + 3] = [0, 0, neutral]
            qpos[w, self.ids["plate_qpos"] + 3:self.ids["plate_qpos"] + 7] = [1, 0, 0, 0]
            R = np.eye(3)
            r = plate["ball_radius_m"]
            bq = self.ids["ball_qpos"]
            qpos[w, bq:bq + 3] = [0.0, 0.0, neutral] + R @ np.array(
                [xy[0], xy[1], r + 1e-5 if z is None else z])
            qpos[w, bq + 3:bq + 7] = [1, 0, 0, 0]
            bd = self.ids["ball_dof"]
            vel = R @ np.array([vxy[0], vxy[1], 0.0])
            qvel[w, bd:bd + 3] = vel
            if z is None:
                qvel[w, bd + 3:bd + 6] = np.cross(R[:, 2], vel) / r
            q, knee = legs.ik(p0, plate, 0.0, 0.0, 0.0)
            qpos[w, self.ids["leg_qpos"]] = q
            qpos[w, self.ids["knee_qpos"]] = knee
            if self.scan_servo:
                q0 = np.array([scan.servo_angle(0.0, i, self.sc, self.sn)
                               for i in range(self.ns)])
                qpos[w, self.ids["scan_qpos"]] = q0
                scanlatch[w] = self._to_us(q0, self.scan_spec)
                scannext[w] = 0.0
            tm[w] = 0.0
            ctrl[w] = 0.0
            qacc[w] = 0.0; qacc_w[w] = 0.0
            us = self._to_us(q)
            legsent[w] = us; leglatch[w] = us; legnext[w] = 0.0; legsat[w] = 0
            scansat[w] = 0
            x[w] = 0.0; P[w] = np.eye(4).ravel(); ok[w] = 0; tl[w] = -1e9
            push[w] = 0.0; esc[w] = 0; et[w] = -1.0
            offs[w] = rng.normal(0, self.sn["offset_sigma_m"], self.ns)
            kk = disturb.kick_list(self.e["disturb"], rng, self.e["episode_s"])
            nkicks[w] = min(len(kk), self.max_kicks)
            kicks[w] = 0.0
            if len(kk):
                kicks[w, :min(len(kk), self.max_kicks)] = kk[:self.max_kicks]
            hist[w] = 0.0; hist_gpu[w] = 0.0
            wins[w] = -999999; accn[w] = 0; accy[w] = 0.0; accb[w] = 0.0; amb[w] = 0.0
            finished[w] = 0
        rc = self.read_count.numpy()
        if indices is None:
            self.k = 0
            self.steps[:] = 0
            self.action[:] = 0.0; self.last_action[:] = 0.0
            rc[:] = 0
        else:
            rc[idx] = 0
        self.d.qpos.assign(qpos); self.d.qvel.assign(qvel); self.d.time.assign(tm)
        self.d.ctrl.assign(ctrl); self.d.qacc.assign(qacc); self.d.qacc_warmstart.assign(qacc_w)
        self.offsets.assign(offs); self.x.assign(x); self.P.assign(P); self.ok.assign(ok)
        self.tlast.assign(tl); self.push.assign(push); self.escaped.assign(esc)
        self.escape_t.assign(et); self.leg_sent.assign(legsent); self.leg_latched.assign(leglatch)
        self.leg_next.assign(legnext); self.leg_sat.assign(legsat)
        self.scan_latched.assign(scanlatch); self.scan_next.assign(scannext)
        self.scan_sat.assign(scansat); self.hist_obs = hist
        self.kicks.assign(kicks); self.nkicks.assign(nkicks); self.wins.assign(wins)
        self.acc_n.assign(accn); self.acc_yaw.assign(accy); self.acc_beam.assign(accb)
        self.amb.assign(amb); self.hist.assign(hist_gpu); self.finished.assign(finished)
        self.read_count.assign(rc.astype(np.int32))
        self.steps[idx] = 0
        self.action[idx] = 0.0; self.last_action[idx] = 0.0
        mjw.forward(self.m, self.d)
        return self.obs(), self._plain_info()

    def _to_us(self, q, spec=None, res=None):
        s = self.leg_spec if spec is None else spec
        r = self.pf["pwm_resolution_us"] if res is None else res
        p = s["pulse_centre_us"] + np.asarray(q) * s["us_per_rad"]
        p = np.round(p / r) * r
        return np.clip(p, s["pulse_min_us"], s["pulse_max_us"]).astype(np.float32)

    def _to_sent(self, action):
        s = self.leg_spec
        q = np.clip(action, -1.0, 1.0) * self.q_max
        p = s["pulse_centre_us"] + q * s["us_per_rad"]
        p = np.round(p / self.pf["pwm_resolution_us"]) * self.pf["pwm_resolution_us"]
        return np.clip(p, s["pulse_min_us"], s["pulse_max_us"]).astype(np.float32)

    def step(self, actions):
        a = np.asarray(actions, np.float32)
        a = np.clip(a, -1.0, 1.0)
        rate = a - self.last_action
        self.action = a.copy()
        self.last_action = a.copy()
        self.leg_sent.assign(self._to_sent(a))
        if self.graph is not None and self.n_sub % self.sub_every == 0:
            nblock = self.n_sub // self.sub_every
            for _ in range(nblock):
                wp.capture_launch(self.graph)
            self.k += self.n_sub
        else:
            for k in range(self.n_sub):
                self._substep(k % self.sub_every == self.sub_every // 2)
        self.steps += 1
        wp.launch(kernels.truth_step, dim=self.b,
                  inputs=[self.d.qpos, self.d.qvel, self.d.time, self.ids["plate_qpos"],
                          self.ids["ball_qpos"], self.ids["ball_dof"], self.half,
                          self.escaped, self.escape_t, self.truth_pos, self.truth_vel])
        obs, info = self._after(rate)
        return obs, info["reward"], info["escaped"], info["truncated"], info

    def _capture(self):
        if self.b > 64 or self.n_sub % self.sub_every != 0:
            return
        for j in range(self.sub_every):
            self._substep(j == self.sub_every // 2)
        wp.synchronize()
        self.reset()
        wp.synchronize()
        try:
            with wp.ScopedCapture() as cap:
                for j in range(self.sub_every):
                    self._substep(j == self.sub_every // 2)
            self.graph = cap.graph
        except Exception as exc:
            print(f"GPU graph capture disabled: {exc}")
            self.graph = None
        self.reset()

    def _substep(self, do_tof=False):
        wp.launch(kernels.leg_servo_step, dim=self.b,
                  inputs=[self.d.qpos, self.d.qvel, self.d.ctrl, self.d.time,
                          self.leg_qidx, self.leg_lidx, self.leg_didx,
                          self.leg_sent, self.leg_latched, self.leg_next, self.leg_sat,
                          self.leg_frame_dt, self.leg_spec["us_per_rad"],
                          self.leg_spec["pulse_centre_us"], self.leg_spec["pulse_min_us"],
                          self.leg_spec["pulse_max_us"], self.pf["pwm_resolution_us"],
                          np.radians(self.leg_spec["prop_band_deg"]) * self.leg_spec["us_per_rad"],
                          self.leg_spec["deadband_us"], self.leg_spec["w_noload"],
                          self.leg_spec["t_stall"], self.pf["trajectory_speed_rps"]])
        if self.scan_servo:
            wp.launch(kernels.scan_servo_step, dim=self.b,
                      inputs=[self.d.qpos, self.d.qvel, self.d.ctrl, self.d.time,
                              self.scan_qidx, self.scan_lidx, self.scan_didx, 3,
                              self.scan_latched, self.scan_next, self.scan_sat,
                              self.sc["sector_deg"], self.sc["step_deg"],
                              self.sn["timing_budget_ms"] / 1000.0,
                              self.sc["sector_deg"] / self.sc["step_deg"] * self.sn["timing_budget_ms"] / 1000.0,
                              1.0 / self.ns if self.sc["phase_stagger"] else 0.0,
                              1 if self.sc["mode"] == "step" else 0,
                              self.scan_frame_dt, self.scan_spec["us_per_rad"],
                              self.scan_spec["pulse_centre_us"],
                              self.scan_spec["pulse_min_us"], self.scan_spec["pulse_max_us"],
                              self.pf["pwm_resolution_us"],
                              np.radians(self.scan_spec["prop_band_deg"]) * self.scan_spec["us_per_rad"],
                              self.scan_spec["deadband_us"], self.scan_spec["w_noload"],
                              self.scan_spec["t_stall"]])
        wp.launch(kernels.force_step, dim=self.b,
                  inputs=[self.d.qpos, self.d.xfrc_applied, self.d.time,
                          self.ids["plate_qpos"], self.ids["ball"], self.push,
                          self.kicks, self.nkicks, self.dt, self.seed,
                          self.e["disturb"].get("push_sigma_n", 0.0),
                          self.e["disturb"].get("push_tau_s", 0.3)])
        wp.launch(kernels.truth_step, dim=self.b,
                  inputs=[self.d.qpos, self.d.qvel, self.d.time,
                          self.ids["plate_qpos"], self.ids["ball_qpos"],
                          self.ids["ball_dof"], self.half, self.escaped,
                          self.escape_t, self.truth_pos, self.truth_vel])
        if do_tof:
            self._tof()
        wp.launch(kernels.tracker_predict, dim=self.b,
                  inputs=[self.d.qpos, self.ids["plate_qpos"], self.x, self.P,
                          self.dt, self.cfg["tracker"]["roll_factor"],
                          self.cfg["tracker"]["process_accel_sigma"], self.limit])
        mjw.step(self.m, self.d)
        self.k += 1

    def _tof(self):
        sec = self.sc["sector_deg"]; step = self.sc["step_deg"]
        rt = self.sn["timing_budget_ms"] / 1000.0
        sweep = sec / step * rt
        phase = 1.0 / self.ns if self.sc["phase_stagger"] else 0.0
        sm = 1 if self.sc["mode"] == "step" else 0
        wp.launch(kernels.sensor_poses, dim=self.b,
                  inputs=[self.d.qpos, self.d.time, self.ids["plate_qpos"],
                          self.corner_x, self.corner_y, self.bisector_wp,
                          self.scan_qidx, self.scan_lidx,
                          sec, step, rt, sweep, phase, sm,
                          1 if self.scan_servo else 0,
                          1 if self.sc["yaw_source"] == "measured" else 0,
                          np.radians(self.scan_spec["feedback_noise_deg"]) if self.scan_spec else 0.0,
                          self.mount_height, np.radians(self.sc["pitch_deg"]), self.seed,
                          self.sensor_origin, self.sensor_rot, self.sensor_beam,
                          self.sensor_bearing])
        wp.launch(kernels.primary_rays, dim=(self.b, self.ns * self.nray),
                  inputs=[self.sensor_origin, self.sensor_rot, self.ray_local, self.nray,
                          self.primary_pnt, self.primary_vec])
        mjw.refit_bvh(self.m, self.d, self.rc)
        mjw.rays(self.m, self.d, self.primary_pnt, self.primary_vec,
                 vec6f(-1, -1, -1, -1, -1, -1), True, self.be,
                 self.primary_dist, self.primary_geom, self.primary_normal, self.rc)
        if self.naux:
            wp.launch(kernels.aux_rays, dim=(self.b, self.naux),
                      inputs=[self.sensor_origin, self.primary_pnt, self.primary_vec,
                              self.primary_dist, self.nray, self.ns,
                              self.aux_pnt, self.aux_vec, self.aux_valid])
            mjw.rays(self.m, self.d, self.aux_pnt, self.aux_vec,
                     vec6f(-1, -1, -1, -1, -1, -1), True, self.aux_be,
                     self.aux_dist, self.aux_geom, self.aux_normal, self.rc)
        wp.launch(tof_kernels.tof_tick, dim=self.b,
                  inputs=[self.d.time, self.sensor_origin, self.sensor_rot, self.sensor_beam,
                          self.sensor_bearing, self.primary_pnt, self.primary_vec,
                          self.primary_dist, self.primary_geom, self.primary_normal,
                          self.aux_pnt, self.aux_vec, self.aux_dist, self.aux_geom,
                          self.aux_normal, self.aux_valid, self.albedo, self.housing,
                          self.ray_power, self.ray_theta, self.pulse_w, self.hist,
                          self.amb, self.acc_yaw, self.acc_beam, self.acc_n, self.wins,
                          self.rd_valid, self.rd_range, self.rd_sigma, self.rd_signal,
                          self.rd_ambient, self.rd_yaw, self.rd_beam, self.rd_slot,
                          self.rd_time, self.rd_sensor, self.finished, self.read_count,
                          self.read_valid, self.read_sensor, self.read_range,
                          self.read_sigma, self.read_signal, self.read_ambient,
                          self.read_yaw, self.read_beam, self.read_slot, self.read_time,
                          self.offsets, self.nray, self.ns, self.nbins, self.nslots,
                          self.sn["bin_m"], self.sn["max_range_m"], float(self.s_ref),
                          rt, sec, step, sweep, phase, sm,
                          self.sn["offset_sigma_m"], self.sn["sigma_ref_m"],
                          self.sn["timing_budget_ms"], self.sn["ref_budget_ms"],
                          self.sn["sigma_threshold_m"], self.sn["ambient_frac"],
                          self.sn["ambient_span_m"], self.sn["nearest_frac"],
                          1 if self.sn["peak_select"] == "nearest" else 0,
                          self.sn["pulse_sigma_m"], len(self.pulse),
                          1 if self.e["interference"] else 0,
                          np.radians(self.sn["fov_full_deg_near"] / 2),
                          np.radians(self.sn["fov_full_deg_far"] / 2),
                          float(self.sn["fov_near_m"]), float(self.sn["fov_far_m"]),
                          np.radians(self.sn["foi_half_deg"]), self.seed])
        wp.launch(tof_kernels.tracker_from_readings, dim=self.b,
                  inputs=[self.x, self.P, self.ok, self.tlast, self.finished,
                          self.rd_valid, self.rd_range, self.rd_sigma, self.rd_yaw,
                          self.rd_slot, self.rd_time, self.corner_x, self.corner_y,
                          self.bg, self.plate["ball_radius_m"], self.limit,
                          self.cfg["tracker"]["lost_after_s"],
                          self.cfg["tracker"]["range_sigma_floor_m"],
                          self.cfg["tracker"]["bg_margin_m"],
                          self.cfg["tracker"]["gate_chi2"],
                          self.cfg["tracker"]["init_pos_sigma_m"],
                          self.cfg["tracker"]["init_vel_sigma_mps"],
                          np.radians(self.sn["fov_full_deg_near"] / 2),
                          np.radians(self.sn["fov_full_deg_far"] / 2),
                          float(self.sn["fov_near_m"]), float(self.sn["fov_far_m"])])
