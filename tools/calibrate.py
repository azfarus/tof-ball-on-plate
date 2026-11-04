"""Sanity checks: ToF accuracy vs the datasheet, spot size, ball rolling physics,
servo step response. S_ref is computed on the fly (tof.reference_signal); nothing
is written.

Usage: .venv/bin/python tools/calibrate.py
"""
import numpy as np

from bop_env import Sim, config, tof


def physics_check(cfg, tilt_deg=2.0):
    """Ball acceleration on a held tilt vs rolling theory."""
    sim = Sim(cfg)
    a = np.radians(tilt_deg)
    sim.reset(ball_xy=(0.0, 0.0), tilt=(a, 0.0), seed=0)
    sim.command_pose((a, 0.0))
    sim.advance(0.05)  # let the servos take the load
    ts, ys = [], []
    for k in range(int(0.3 / sim.dt)):
        sim.step()
        if k % 40 == 0:
            ts.append(sim.d.time)
            ys.append(sim.ball()[0][1])
    acc = 2 * np.polyfit(ts, ys, 2)[0]
    k = 2 / 3 if cfg["plate"]["ball_hollow"] else 2 / 5
    roll = sim.plate_pose()[0]
    return acc, -9.81 * np.sin(roll) / (1 + k)


def step_check(cfg, deg=8.0):
    """Time for the plate to reach 90% of a roll step."""
    sim = Sim(cfg)
    sim.reset(ball_xy=(0, 0), ball_z=5.0, seed=0)
    sim.command_pose((np.radians(deg), 0.0))
    for k in range(int(0.5 / sim.dt)):
        sim.step()
        if np.degrees(sim.plate_pose()[0]) >= 0.9 * deg:
            return sim.d.time
    return np.nan


def main():
    cfg = config.load_all()
    sensor = cfg["sensor"]
    rays = tof.make_rays(sensor)
    s_ref = tof.reference_signal(sensor, rays)
    print(f"S_ref (white 88%, 200 mm, full FoV): {s_ref:.4e}")
    rng = np.random.default_rng(0)
    for name, alb, dist in [("white 88%", 0.88, 0.2), ("gray 17%", 0.17, 0.475)]:
        for budget in [33.0, sensor["timing_budget_ms"]]:
            outs = [tof.wall_reading(sensor, rays, dist, alb, s_ref, budget, rng)
                    for _ in range(300)]
            valid = np.mean([o["valid"] for o in outs])
            err = np.array([o["range"] - dist for o in outs if o["valid"]])
            p90 = np.percentile(np.abs(err), 90) * 1000 if len(err) else np.nan
            print(f"{name} @ {dist * 1000:.0f} mm, {budget:.0f} ms: sigma "
                  f"{outs[0]['sigma'] * 1000:.1f} mm, valid {valid:.0%}, |err| p90 {p90:.1f} mm")
    for dist in [0.1, 0.212, 0.3, 0.424]:
        half = tof.rx_half(sensor, dist)
        print(f"detection cone {np.degrees(2 * half):.1f} deg -> spot "
              f"{2 * dist * np.tan(half) * 1000:.0f} mm wide at {dist * 1000:.0f} mm")
    acc, theory = physics_check(cfg)
    print(f"ball accel on 2 deg tilt: sim {acc:.4f} m/s^2, theory {theory:.4f} m/s^2")
    for model in ["SG90", "MG996R", "DS3218", "TD8135MG"]:
        t90 = step_check(config.load_all({"platform.servo_model": model}))
        print(f"8 deg roll step, {model}: 90% in {t90 * 1000:.0f} ms")


if __name__ == "__main__":
    main()
