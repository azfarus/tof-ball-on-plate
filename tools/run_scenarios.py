"""Run the scripted tilt scenarios (config/scenarios.toml) and score the tracker.

The host commands each scenario's tilt profile through the IK; the ball is
tracked with the ToF sensors. Writes per-run trace/readings CSVs, PNGs and
summary.csv.

Usage (from project root):
  .venv/bin/python tools/run_scenarios.py                    # interference on and off
  .venv/bin/python tools/run_scenarios.py --scenario circle --interference on
  .venv/bin/python tools/run_scenarios.py --view --scenario servo_brisk --interference on
  .venv/bin/python tools/run_scenarios.py --set scan.actuator=ideal --out sim/out/ideal_scan
"""
import argparse
import csv
import json
import pathlib

import numpy as np

import plots
from bop_env import Sim, config, disturb

OUT = config.ROOT / "sim" / "out"
TRACE_COLS = ("t,true_x,true_y,est_x,est_y,tracked,true_z,cmd_roll,cmd_pitch,cmd_heave,"
              "act_roll,act_pitch,act_heave,servo_saturated_frac")


def run_scenario(sim, sc, seed, duration, view_cfg=None):
    """Returns (trace, readings, escaped_at_s)."""
    rng = np.random.default_rng(seed)
    rp = disturb.random_tilt(sc, rng) if sc["motion"] == "random" else None
    sim.keep_log = True
    sim.reset(ball_xy=sc["ball_xy_m"], ball_v=sc["ball_v_mps"], ball_z=sc.get("ball_z_m"),
              tilt=disturb.tilt_at(sc, 0.0, rp), heave=disturb.heave_at(sc, 0.0, rp),
              forces=sc, horizon=duration, seed=seed)
    view = None
    if view_cfg:
        from bop_env import viewer
        view = viewer.open_viewer(sim.m, sim.d, view_cfg)
    log_every = int(round(0.001 / sim.dt))
    trace, escaped = [], np.nan
    for k in range(int(duration / sim.dt)):
        t = sim.d.time
        cmd, cmd_h = disturb.tilt_at(sc, t, rp), disturb.heave_at(sc, t, rp)
        sim.command_pose(cmd, cmd_h)
        sim.step()
        if sim.escaped:
            escaped = t  # ball left the plate: stop, results after this are void
            break
        if k % log_every == 0:
            p, _ = sim.ball()
            act = sim.plate_pose()
            trace.append([t, p[0], p[1], sim.trk.x[0], sim.trk.x[1], float(sim.trk.ok), p[2],
                          cmd[0], cmd[1], cmd_h, *act, sim.n_saturated / (k + 1)])
        if view and not viewer.draw(view, sim):
            break
    if view:
        view["v"].close()
    for rd in sim.log:
        rd["in_cone"] = sim.in_cone(rd)
    return np.array(trace), sim.log, escaped


def metrics(trace, readings, duration, ball_r=0.02):
    ok = trace[:, 5] > 0
    err = np.linalg.norm(trace[ok, 1:3] - trace[ok, 3:5], axis=1) * 1000
    on = ok & (trace[:, 6] < ball_r + 0.003)  # ball touching the plate
    err_on = np.linalg.norm(trace[on, 1:3] - trace[on, 3:5], axis=1) * 1000
    tilt_err = np.linalg.norm(trace[:, 10:12] - trace[:, 7:9], axis=1)  # actual vs commanded
    st = [r["status"] for r in readings]
    cone = np.array([r["in_cone"] for r in readings])
    det = np.array([r["status"] in ("accepted", "init", "gated") for r in readings])
    good = [r for r in readings if r["status"] == "accepted"]
    rerr = [abs(r["range"] - r["true_range"]) * 1000 for r in good]
    nan = float("nan")
    return {
        "rmse_mm": float(np.sqrt(np.mean(err**2))) if len(err) else nan,
        "max_speed_mps": float(np.max(np.linalg.norm(np.diff(trace[:, 1:3], axis=0), axis=1)
                                      / np.diff(trace[:, 0]))),
        "p95_mm": float(np.percentile(err, 95)) if len(err) else nan,
        "rmse_on_plate_mm": float(np.sqrt(np.mean(err_on**2))) if len(err_on) else nan,
        "tilt_err_rms_deg": float(np.degrees(np.sqrt(np.mean(tilt_err**2)))),
        "tilt_err_max_deg": float(np.degrees(tilt_err.max())),
        "heave_err_rms_mm": float(np.sqrt(np.mean((trace[:, 12] - trace[:, 9]) ** 2)) * 1000),
        "max_tilt_deg": float(np.degrees(np.abs(trace[:, 10:12]).max())),
        "servo_saturated_frac": float(trace[-1, 13]),
        "airborne_frac": float(np.mean(trace[:, 6] > ball_r + 0.003)),
        "max_mm": float(err.max()) if len(err) else nan,
        "tracked_frac": float(ok.mean()),
        "readings_per_s": len(readings) / duration,
        "accepted_per_s": st.count("accepted") / duration,
        "detect_when_in_cone": float(det[cone].mean()) if cone.any() else nan,
        "false_when_out_cone": float(det[~cone].mean()) if (~cone).any() else nan,
        "invalid_frac": st.count("invalid") / len(st),
        "gated_frac": st.count("gated") / len(st),
        "range_err_p50_mm": float(np.median(rerr)) if rerr else nan,
        "interfered_frac": float(np.mean([r["ambient"] > 0 for r in readings])),
    }


def save_readings(path, readings):
    keys = ["t", "sensor", "slot", "yaw", "beam_yaw", "valid", "range", "sigma", "signal",
            "ambient", "status", "in_cone", "true_range"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in readings:
            w.writerow([r[k] for k in keys])


def parse_sets(items):
    """--set section.key=value (value parsed as JSON, else string)."""
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--interference", choices=["on", "off", "both"], default="both")
    ap.add_argument("--set", action="append", help="config override section.key=value")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--view", action="store_true", help="live MuJoCo viewer")
    args = ap.parse_args()

    cfg = config.load_all(parse_sets(args.set))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration, seed = cfg["scenarios"]["duration_s"], cfg["scenarios"]["seed"]
    modes = {"on": [True], "off": [False], "both": [True, False]}[args.interference]
    sims = {inter: Sim(cfg, interference=inter) for inter in modes}
    rows = []
    for sc in cfg["scenarios"]["scenario"]:
        if args.scenario and sc["name"] != args.scenario:
            continue
        for inter in modes:
            tag = f"{sc['name']}_{'interf' if inter else 'clean'}"
            trace, readings, escaped = run_scenario(
                sims[inter], sc, seed, duration, config.load("viewer") if args.view else None)
            met = metrics(trace, readings, trace[-1, 0] + 0.001, cfg["plate"]["ball_radius_m"])
            met["escaped_at_s"] = escaped
            rows.append({"run": tag, **met})
            np.savetxt(out_dir / f"{tag}_trace.csv", trace, delimiter=",", header=TRACE_COLS,
                       comments="")
            save_readings(out_dir / f"{tag}_readings.csv", readings)
            plots.scenario_plot(out_dir / f"{tag}.png", tag, trace, readings,
                                sims[inter].corners, cfg)
            print(tag, {k: round(v, 3) for k, v in met.items()})

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plots.summary_plot(out_dir / "summary.png", rows)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
