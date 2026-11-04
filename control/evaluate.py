"""Evaluate a controller on the env: PID, zero action, or a trained SB3 model.

Usage (from project root):
  .venv/bin/python control/evaluate.py --controller pid --episodes 10
  .venv/bin/python control/evaluate.py --controller model --path runs/ppo_xxx/final.zip
  .venv/bin/python control/evaluate.py --controller pid --view --set scan.actuator=ideal
"""
import argparse
import json
import pathlib
import tomllib

import numpy as np

from bop_env import BallOnPlateEnv, config
from pid import PID


def parse_sets(items):
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_model(path):
    """SB3 model + its VecNormalize stats (saved next to it by train.py)."""
    import pickle
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    path = pathlib.Path(path)
    ckpt = path.parent.name == "ckpt"  # checkpoint: <run>/ckpt/<algo>_<n>_steps.zip
    run = path.parent.parent if ckpt else path.parent
    meta = json.loads((run / "meta.json").read_text())
    algo = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "ddpg": DDPG}[meta["algo"]]
    model = algo.load(path, device="cpu")
    norm = (path.parent / path.name.replace("_steps.zip", "_steps.pkl")
            .replace(f"{meta['algo']}_", f"{meta['algo']}_vecnormalize_", 1)
            if ckpt else run / "vecnormalize.pkl")
    vn = pickle.loads(norm.read_bytes()) if norm.exists() else None
    return model, vn, meta["overrides"]


def run(env, policy, episodes, seed):
    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        errs, est, ret, done = [], [], 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            ret += r
            errs.append(info["err_m"])
            if info["tracked"]:
                est.append(np.linalg.norm(info["ball_xy"] - info["est_xy"]))
            done = term or trunc
        rows.append({"return": ret, "fell": info["escaped"], "t_end": info["t"],
                     "rms_err_mm": 1000 * float(np.sqrt(np.mean(np.square(errs)))),
                     "est_rmse_mm": 1000 * float(np.sqrt(np.mean(np.square(est)))) if est else np.nan})
        print(ep, {k: round(float(v), 2) for k, v in rows[-1].items()})
    return rows


class ModelPolicy:
    def __init__(self, model, vn):
        self.model, self.vn = model, vn

    def act(self, obs):
        o = {k: v[None] for k, v in obs.items()}
        if self.vn is not None:
            o = self.vn.normalize_obs(o)
        return self.model.predict(o, deterministic=True)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["pid", "zero", "model"], default="pid")
    ap.add_argument("--path", help="model .zip for --controller model")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--set", action="append", help="config override section.key=value")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    overrides = parse_sets(args.set)
    if args.controller == "model":
        model, vn, train_ov = load_model(args.path)
        overrides = {**train_ov, **overrides}
    env = BallOnPlateEnv(overrides, render_mode="human" if args.view else None)
    with open(config.CONFIG_DIR / "control.toml", "rb") as f:
        ctrl = tomllib.load(f)
    if args.controller == "pid":
        policy = PID(env.cfg, ctrl["pid"])
    elif args.controller == "zero":
        policy = type("Zero", (), {"act": staticmethod(lambda o: np.zeros(3, np.float32))})()
    else:
        policy = ModelPolicy(model, vn)
    rows = run(env, policy, args.episodes, args.seed)
    keys = rows[0].keys()
    print("mean", {k: round(float(np.nanmean([r[k] for r in rows])), 2) for k in keys})
    env.close()


if __name__ == "__main__":
    main()
