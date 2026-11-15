"""Compare one CPU and one GPU physics step, then optionally benchmark.

Run from the repository root:
  .venv/bin/python tools/check_gpu.py
  .venv/bin/python tools/check_gpu.py --benchmark 64
"""
import argparse
import time

import numpy as np

from bop_env import config
from bop_env.sim import Sim
from bop_env.gpu.sim import WarpSim


def make_config():
    cfg = config.load_all()
    cfg = config.apply(cfg, cfg["env"].get("overrides", {}))
    return config.apply(cfg, {
        "scan.actuator": "ideal",
        "scan.interference": False,
        "env.start.ball_xy_m": 0.0,
        "env.start.ball_v_mps": 0.0,
        "env.disturb.kicks_per_s": 0.0,
        "env.disturb.push_sigma_n": 0.0,
        "sensor.offset_sigma_m": 0.0,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=int, default=0, help="world count for a timing run")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    cfg = make_config()

    cpu = Sim(cfg)
    cpu.reset(ball_xy=(0, 0), ball_v=(0, 0), seed=0)
    gpu = WarpSim(cfg, 1, seed=0)
    cpu_pos = []
    for _ in range(args.steps):
        cpu.send_legs(np.zeros(3))
        for _ in range(gpu.n_sub):
            cpu.step()
        gpu.step(np.zeros((1, 3), np.float32))
        cpu_pos.append(cpu.d.qpos[cpu.ids["ball_qpos"]:cpu.ids["ball_qpos"] + 3].copy())
    gpu_pos = gpu.d.qpos.numpy()[0, gpu.ids["ball_qpos"]:gpu.ids["ball_qpos"] + 3]
    print(f"CPU/GPU final ball position difference: "
          f"{np.linalg.norm(gpu_pos - cpu_pos[-1]) * 1000:.3f} mm")
    print(f"CPU state checked against {len(cpu_pos)} control periods")

    if args.benchmark:
        sim = WarpSim(cfg, args.benchmark, seed=0)
        sim.step(np.zeros((args.benchmark, 3), np.float32))
        n = 5
        t0 = time.perf_counter()
        for _ in range(n):
            sim.step(np.zeros((args.benchmark, 3), np.float32))
        dt = time.perf_counter() - t0
        print(f"{args.benchmark} GPU worlds: {dt / n * 1000:.1f} ms/control step, "
              f"{args.benchmark * n / dt:.0f} transitions/s")


if __name__ == "__main__":
    main()
