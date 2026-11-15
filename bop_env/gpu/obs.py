"""Observation and diagnostic assembly for the GPU backend."""
import numpy as np

from .. import legs


def _rot(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class ObservationMixin:
    def _after(self, rate):
        tm = self.d.time.numpy()
        counts = self.read_count.numpy()
        rv = self.read_valid.numpy(); rs = self.read_sensor.numpy()
        rr = self.read_range.numpy(); rg = self.read_sigma.numpy()
        rsi = self.read_signal.numpy(); ra = self.read_ambient.numpy()
        ry = self.read_yaw.numpy()
        rt = self.read_time.numpy()
        for w in range(self.b):
            for k in range(min(int(counts[w]), self.max_reads)):
                s = int(rs[w, k])
                self.hist_obs[w, s, 1:] = self.hist_obs[w, s, :-1].copy()
                ok = int(rv[w, k])
                self.hist_obs[w, s, 0] = [ok, rr[w, k] if ok else 0.0,
                                           rg[w, k] if ok else 0.0,
                                           rsi[w, k] / max(self.s_ref, 1e-12),
                                           ra[w, k] / max(self.s_ref, 1e-12),
                                           ry[w, k], tm[w] - rt[w, k]]
        x = self.x.numpy(); ok = self.ok.numpy()
        esc = self.escaped.numpy().astype(bool)
        qpos = self.d.qpos.numpy()
        qvel = self.d.qvel.numpy()
        q = qpos[:, self.ids["plate_qpos"] + 3:self.ids["plate_qpos"] + 7]
        R = _rot(q)
        rp = np.einsum("bji,bj->bi", R, qpos[:, self.ids["ball_qpos"]:self.ids["ball_qpos"] + 3]
                       - qpos[:, self.ids["plate_qpos"]:self.ids["plate_qpos"] + 3])
        rv_ = np.einsum("bji,bj->bi", R, qvel[:, self.ids["ball_dof"]:self.ids["ball_dof"] + 3])
        p = x[:, :2] if self.e["reward_source"] == "estimate" else rp[:, :2]
        v = x[:, 2:] if self.e["reward_source"] == "estimate" else rv_[:, :2]
        err = np.linalg.norm(p - self.target, axis=1) / self.half
        w = self.e["reward"]
        rew = (1.0 - w["w_pos"] * err - w["w_vel"] * np.linalg.norm(v, axis=1)
               - w["w_rate"] * np.sum(rate * rate, axis=1)
               - w["w_act"] * np.sum(self.action * self.action, axis=1))
        term = esc.copy()
        rew[term] = -float(w["escape_penalty"])
        trunc = (self.steps >= self.max_steps) & ~term
        self.read_count.assign(np.zeros(self.b, np.int32))
        obs = {"ball_est": np.concatenate([x, ok[:, None]], axis=1).astype(np.float32),
               "tof": self.hist_obs.copy(), "servo_cmd": self.action.copy(),
               "target": self.target.copy()}
        if self.e["servo_feedback"]:
            obs["servo_fb"] = (qpos[:, self.ids["leg_qpos"]] + qpos[:, self.ids["leg_lash"]]).astype(np.float32)
        if self.e["plate_pose"]:
            pose = np.stack([np.arctan2(-R[:, 1, 2], R[:, 2, 2]),
                             np.arcsin(np.clip(R[:, 0, 2], -1, 1)),
                             qpos[:, self.ids["plate_qpos"] + 2] - legs.neutral_height(self.pf, self.plate)], 1)
            obs["plate"] = pose.astype(np.float32)
        info = {"ball_xy": rp[:, :2].copy(), "ball_v": rv_[:, :2].copy(),
                "est_xy": x[:, :2].copy(), "tracked": ok.astype(bool),
                "err_m": np.linalg.norm(rp[:, :2] - self.target, axis=1),
                "escaped": esc, "plate_pose": np.stack([np.arctan2(-R[:, 1, 2], R[:, 2, 2]),
                                                        np.arcsin(np.clip(R[:, 0, 2], -1, 1)),
                                                        qpos[:, self.ids["plate_qpos"] + 2] - legs.neutral_height(self.pf, self.plate)], 1),
                "servo_saturated": self.leg_sat.numpy().astype(bool),
                "t": tm.copy(), "reward": rew, "truncated": trunc}
        return obs, info

    def obs(self):
        x = self.x.numpy(); ok = self.ok.numpy()
        obs = {"ball_est": np.concatenate([x, ok[:, None]], axis=1).astype(np.float32),
               "tof": self.hist_obs.copy(), "servo_cmd": self.action.copy(),
               "target": self.target.copy()}
        if self.e["servo_feedback"]:
            qpos = self.d.qpos.numpy()
            obs["servo_fb"] = (qpos[:, self.ids["leg_qpos"]] + qpos[:, self.ids["leg_lash"]]).astype(np.float32)
        if self.e["plate_pose"]:
            qpos = self.d.qpos.numpy()
            R = _rot(qpos[:, self.ids["plate_qpos"] + 3:self.ids["plate_qpos"] + 7])
            obs["plate"] = np.stack([np.arctan2(-R[:, 1, 2], R[:, 2, 2]),
                                     np.arcsin(np.clip(R[:, 0, 2], -1, 1)),
                                     qpos[:, self.ids["plate_qpos"] + 2] - legs.neutral_height(self.pf, self.plate)], 1).astype(np.float32)
        return obs

    def _plain_info(self):
        tm = self.d.time.numpy()
        x = self.x.numpy()
        ok = self.ok.numpy().astype(bool)
        esc = self.escaped.numpy().astype(bool)
        qpos = self.d.qpos.numpy()
        qvel = self.d.qvel.numpy()
        q = qpos[:, self.ids["plate_qpos"] + 3:self.ids["plate_qpos"] + 7]
        R = _rot(q)
        rp = np.einsum("bji,bj->bi", R, qpos[:, self.ids["ball_qpos"]:self.ids["ball_qpos"] + 3]
                       - qpos[:, self.ids["plate_qpos"]:self.ids["plate_qpos"] + 3])
        rv = np.einsum("bji,bj->bi", R, qvel[:, self.ids["ball_dof"]:self.ids["ball_dof"] + 3])
        pose = np.stack([np.arctan2(-R[:, 1, 2], R[:, 2, 2]),
                         np.arcsin(np.clip(R[:, 0, 2], -1, 1)),
                         qpos[:, self.ids["plate_qpos"] + 2] - legs.neutral_height(self.pf, self.plate)], 1)
        return {"ball_xy": rp[:, :2].copy(), "ball_v": rv[:, :2].copy(),
                "est_xy": x[:, :2].copy(), "tracked": ok,
                "err_m": np.linalg.norm(rp[:, :2] - self.target, axis=1),
                "escaped": esc, "plate_pose": pose,
                "servo_saturated": self.leg_sat.numpy().astype(bool),
                "t": tm.copy(), "target": self.target.copy()}

    def info(self):
        return self._plain_info()

    def close(self):
        pass
