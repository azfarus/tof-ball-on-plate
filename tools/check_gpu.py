"""Check the GPU (MuJoCo Warp) backend against the CPU reference, then optionally benchmark.

Checks:
  1. one EKF update and one EKF initialisation, GPU kernel vs tracker.Tracker
  2. ToF history ages in the observation, GPU vs CPU env (tof_history 10)
  3. a rolling ball on a tilted plate with servo scanning: ball path and EKF error, GPU vs CPU

Run from the repository root:
  .venv/bin/python tools/check_gpu.py
  .venv/bin/python tools/check_gpu.py --benchmark 64
"""
import argparse
import time

import numpy as np
import warp as wp

from bop_env import BallOnPlateEnv, config
from bop_env.gpu import kernels
from bop_env.gpu.sim import WarpSim
from bop_env.tracker import Tracker


def env_config(overrides):
    cfg = config.load_all()
    cfg = config.apply(cfg, cfg["env"].get("overrides", {}))
    return config.apply(cfg, overrides)


@wp.kernel
def _one_update(rng: float, sig: float, yaw: float, cx: wp.array(dtype=wp.float32),
                cy: wp.array(dtype=wp.float32), bg: wp.array2d(dtype=wp.float32),
                x: wp.array2d(dtype=wp.float32), P: wp.array2d(dtype=wp.float32),
                ok: wp.array(dtype=wp.int32), tl: wp.array(dtype=wp.float32), R: float,
                lim: float, lost: float, floor: float, margin: float, gate: float,
                p0: float, v0: float, near: float, far: float, dnear: float, dfar: float):
    kernels.tracker_update(0, 0, 1, rng, sig, yaw, 0.01, 0, cx, cy, bg, x, P, ok, tl, R, lim,
                           lost, floor, margin, gate, p0, v0, near, far, dnear, dfar)


def check_ekf(cfg):
    sn, tr = cfg["sensor"], cfg["tracker"]
    R, lim = cfg["plate"]["ball_radius_m"], cfg["plate"]["size_m"] / 2
    corner = np.array([[-0.142, -0.142]])
    bg = np.full((1, 10), np.nan, np.float32)
    d = np.array([0.03, 0.05]) - corner[0]
    rd = {"range": float(np.linalg.norm(d) - R), "sigma": 0.005,
          "yaw": float(np.arctan2(d[1], d[0]) + 0.05), "valid": True, "sensor": 0, "slot": 0}
    P0 = np.diag([0.02**2, 0.02**2, 0.2**2, 0.2**2])
    P0[0, 1] = P0[1, 0] = 1e-4
    for name, x0, ok0 in (("update", [0.0, 0.07, 0.1, -0.05], 1), ("init", [0.0] * 4, 0)):
        cpu = Tracker(tr, sn, corner, R, bg, lim)
        cpu.x, cpu.P, cpu.ok, cpu.t_last = np.array(x0, float), P0.copy(), bool(ok0), 0.0
        cpu.update(rd, 0.01)
        x = wp.array(np.array([x0], np.float32))
        P = wp.array(np.array([P0.ravel()], np.float32))
        args = [rd["range"], rd["sigma"], rd["yaw"], wp.array(corner[:, 0].astype(np.float32)),
                wp.array(corner[:, 1].astype(np.float32)), wp.array(bg), x, P,
                wp.array(np.array([ok0], np.int32)), wp.array(np.zeros(1, np.float32)), R, lim,
                tr["lost_after_s"], tr["range_sigma_floor_m"], tr["bg_margin_m"], tr["gate_chi2"],
                tr["init_pos_sigma_m"], tr["init_vel_sigma_mps"],
                np.radians(sn["fov_full_deg_near"] / 2), np.radians(sn["fov_full_deg_far"] / 2),
                sn["fov_near_m"], sn["fov_far_m"]]
        wp.launch(_one_update, dim=1, inputs=args)
        xg, Pg = x.numpy()[0], P.numpy()[0].reshape(4, 4)
        dx = 1000 * np.linalg.norm(cpu.x - xg)
        dP = np.abs(cpu.P - Pg).max() if ok0 else 0.0
        print(f"EKF {name:6s}: state diff {dx:.3f} (mm, mm/s), max cov diff {dP:.1e}  "
              f"{'OK' if dx < 0.1 and dP < 1e-7 else 'MISMATCH'}")


def check_ages():
    ov = {"env.tof_history": 10, "env.start.ball_xy_m": 0.0, "env.start.ball_v_mps": 0.0}
    gpu = WarpSim(env_config(ov), 1, seed=0)
    cpu = BallOnPlateEnv(ov)
    cpu.reset(seed=0)
    for _ in range(12):
        gpu.step(np.zeros((1, 3), np.float32))
        obs, *_ = cpu.step(np.zeros(3, np.float32))
    a_gpu, a_cpu = gpu.hist_obs[0, 0, :, 6], obs["tof"][0, :, 6]
    ok = np.allclose(a_gpu, a_cpu, atol=2e-3)
    print(f"ToF ages : GPU {np.round(a_gpu, 3)}\n           CPU {np.round(a_cpu, 3)}  "
          f"{'OK' if ok else 'MISMATCH'}")


def check_rolling(seconds=1.0):
    """Ball starts at rest in the centre, plate tilted by a fixed action: compare paths and EKF."""
    ov = {"env.start.ball_xy_m": 0.0, "env.start.ball_v_mps": 0.0,
          "env.disturb.kicks_per_s": 0.0, "env.disturb.push_sigma_n": 0.0}
    act = np.array([0.03, -0.015, -0.015], np.float32)  # gentle tilt: ball stays on the plate
    gpu = WarpSim(env_config(ov), 4, seed=0)
    cpu = BallOnPlateEnv(ov)
    cpu.reset(seed=0)
    n = int(seconds * cpu.e["control_hz"])
    pos, est = {"cpu": [], "gpu": []}, {"cpu": [], "gpu": []}
    for _ in range(n):
        _, _, _, _, ic = cpu.step(act)
        _, _, _, _, ig = gpu.step(np.tile(act, (4, 1)))
        pos["cpu"].append(ic["ball_xy"]); pos["gpu"].append(ig["ball_xy"][0])
        if ic["tracked"]:
            est["cpu"].append(np.linalg.norm(ic["ball_xy"] - ic["est_xy"]))
        est["gpu"] += [np.linalg.norm(ig["ball_xy"][w] - ig["est_xy"][w])
                       for w in range(4) if ig["tracked"][w]]
    if ic["escaped"] or ig["escaped"][0]:
        print("Rolling  : ball left the plate, tilt too steep for this check  MISMATCH")
        return
    gap = 1000 * np.linalg.norm(pos["cpu"][-1] - pos["gpu"][-1])
    moved = 1000 * np.linalg.norm(pos["cpu"][-1])
    rmse = {k: 1000 * np.sqrt(np.mean(np.square(v))) for k, v in est.items()}
    print(f"Rolling  : ball moved {moved:.0f} mm, CPU/GPU end gap {gap:.1f} mm  "
          f"{'OK' if gap < 0.05 * moved + 2 else 'MISMATCH'}")
    print(f"           EKF RMSE CPU {rmse['cpu']:.1f} mm, GPU {rmse['gpu']:.1f} mm  "
          f"{'OK' if rmse['gpu'] < 2 * rmse['cpu'] + 5 else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=0, help="world count for a timing run")
    args = ap.parse_args()
    check_ekf(env_config({}))
    check_ages()
    check_rolling()
    if args.benchmark:
        sim = WarpSim(env_config({}), args.benchmark, seed=0)
        a = np.zeros((args.benchmark, 3), np.float32)
        sim.step(a)
        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            sim.step(a)
        dt = time.perf_counter() - t0
        print(f"{args.benchmark} GPU worlds: {dt / n * 1000:.1f} ms/control step, "
              f"{args.benchmark * n / dt:.0f} transitions/s")


if __name__ == "__main__":
    main()
