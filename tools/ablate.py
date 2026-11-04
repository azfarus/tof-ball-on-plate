"""Ablation study: every variant in config/ablation.toml on every scenario, per seed,
in parallel. Writes sim/out/ablation/{results,summary,variants}.csv + heatmaps.

Usage (from project root):
  .venv/bin/python tools/ablate.py
  .venv/bin/python tools/ablate.py --variants 4_sensors,1_sensor --scenarios drop_50cm
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import plots
import run_scenarios as rs
from bop_env import Sim, config

OUT = config.ROOT / "sim" / "out" / "ablation"
METRICS = ["rmse_mm", "p95_mm", "max_mm", "rmse_on_plate_mm", "accepted_per_s",
           "tilt_err_rms_deg", "tilt_err_max_deg", "servo_saturated_frac",
           "detect_when_in_cone", "tracked_frac", "max_speed_mps"]


def job(args):
    vname, overrides, scname, seed = args
    ov = {k: v for k, v in overrides.items() if k != "run.interference"}
    cfg = config.load_all(ov)
    sim = Sim(cfg, interference=overrides.get("run.interference"))
    sc = next(s for s in cfg["scenarios"]["scenario"] if s["name"] == scname)
    trace, readings, escaped = rs.run_scenario(sim, sc, seed, cfg["scenarios"]["duration_s"])
    met = rs.metrics(trace, readings, trace[-1, 0] + 0.001, cfg["plate"]["ball_radius_m"])
    return {"variant": vname, "scenario": scname, "seed": seed, **met, "escaped_at_s": escaped}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarise(results, variants, scenarios):
    """Mean over seeds per variant x scenario, and per variant over all scenarios."""
    summary, per_variant = [], []
    for v in variants:
        vrows = [r for r in results if r["variant"] == v]
        for s in scenarios:
            rows = [r for r in vrows if r["scenario"] == s]
            row = {"variant": v, "scenario": s}
            for m in METRICS:
                vals = np.array([r[m] for r in rows], float)
                row[m] = float(np.nanmean(vals))
                row[m + "_std"] = float(np.nanstd(vals))
            row["escapes"] = int(np.sum(np.isfinite([r["escaped_at_s"] for r in rows])))
            summary.append(row)
        srows = [r for r in summary if r["variant"] == v]
        per_variant.append({
            "variant": v,
            "mean_rmse_mm": float(np.mean([r["rmse_mm"] for r in srows])),
            "worst_rmse_mm": float(np.max([r["rmse_mm"] for r in srows])),
            "mean_p95_mm": float(np.mean([r["p95_mm"] for r in srows])),
            "mean_tilt_err_rms_deg": float(np.mean([r["tilt_err_rms_deg"] for r in srows])),
            "mean_accepted_per_s": float(np.mean([r["accepted_per_s"] for r in srows])),
            "mean_tracked_frac": float(np.mean([r["tracked_frac"] for r in srows])),
            "escapes": int(sum(r["escapes"] for r in srows)),
        })
    return summary, per_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None, help="comma list of variant names")
    ap.add_argument("--scenarios", default=None, help="comma list of scenario names")
    args = ap.parse_args()

    base = config.load_all()
    abl = config.load("ablation")
    variants = [v for v in abl["variant"]
                if not args.variants or v["name"] in args.variants.split(",")]
    scenarios = [s["name"] for s in base["scenarios"]["scenario"]
                 if not args.scenarios or s["name"] in args.scenarios.split(",")]
    jobs = [(v["name"], v["overrides"], s, seed)
            for v in variants for s in scenarios for seed in abl["seeds"]]
    print(f"{len(jobs)} runs: {len(variants)} variants x {len(scenarios)} scenarios "
          f"x {len(abl['seeds'])} seeds on {abl['workers']} workers")

    with ProcessPoolExecutor(abl["workers"]) as ex:
        results = []
        for k, r in enumerate(ex.map(job, jobs), 1):
            results.append(r)
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done")

    OUT.mkdir(parents=True, exist_ok=True)
    vnames = [v["name"] for v in variants]
    summary, per_variant = summarise(results, vnames, scenarios)
    write_csv(OUT / "results.csv", results)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "variants.csv", per_variant)
    plots.ablation_plot(OUT / "rmse_heatmap.png", summary, vnames, scenarios, "rmse_mm",
                        "Position RMSE (mm), mean of seeds")
    plots.ablation_plot(OUT / "p95_heatmap.png", summary, vnames, scenarios, "p95_mm",
                        "Position error p95 (mm), mean of seeds")
    for r in per_variant:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
