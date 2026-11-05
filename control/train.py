"""Train an RL agent on the env with Stable-Baselines3.

Physics runs headless on CPU worker processes (one env each, SubprocVecEnv);
the policy network trains on the GPU. Settings in config/control.toml [train]
and [algos.*]; any env/sim config value can be overridden with --set.

Usage (from project root):
  .venv/bin/python control/train.py --algo ppo --n-envs 14 --steps 2000000
  .venv/bin/python control/train.py --algo sac --set scan.actuator=ideal
  .venv/bin/python control/train.py --algo sac --net tof   # raw-ToF 3-layer net (tof_net.py)
  .venv/bin/python control/train.py --resume runs/<run>/final.zip --steps 500000
  .venv/bin/tensorboard --logdir runs
Weights & Biases logging is on by default ([wandb] in control.toml, --no-wandb to skip).
"""
import argparse
import json
import os
import time
import tomllib

import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from bop_env import BallOnPlateEnv, config
from evaluate import parse_sets, run_files
from tof_net import ToFNet

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "ddpg": DDPG}


def hyperparams(algo, hp, net=None):
    """SB3 kwargs from the [algos.<name>] table (+ [tof_net] if given)."""
    hp = dict(hp)
    kw = {"policy_kwargs": {"net_arch": hp.pop("net_arch")}}
    if net is not None:
        critic = "vf" if algo in ("ppo", "a2c") else "qf"
        kw["policy_kwargs"] = {
            "features_extractor_class": ToFNet,
            "features_extractor_kwargs": {"layers": net["layers"], "dropout": net["dropout"]},
            "net_arch": {"pi": net["actor_head"], critic: net["critic_head"]}}
    sigma = hp.pop("action_noise_sigma", None)
    if sigma:
        kw["action_noise"] = NormalActionNoise(np.zeros(3), sigma * np.ones(3))
    return {**kw, **hp}


class EpisodeStats(BaseCallback):
    """Logs fall-off rate and final distance to target per finished episode."""

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.logger.record_mean("episode/fell", float(info["escaped"]))
                self.logger.record_mean("episode/final_err_mm", 1000 * info["err_m"])
                self.logger.record_mean("episode/length_s", info["t"])
        return True


def load_dotenv(path):
    """KEY=VALUE lines from the git-ignored .env (e.g. WANDB_API_KEY) into os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and not k.strip().startswith("#") and v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def start_wandb(wb, out, args, overrides, ctrl, run_id=None):
    """W&B run that mirrors the tensorboard scalars (SB3 logs to tensorboard).
    run_id continues the same W&B run when resuming."""
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=wb["project"], entity=wb["entity"] or None, name=out.name,
               id=run_id, resume="allow" if run_id else None,
               mode=args.wandb_mode, dir=str(out), sync_tensorboard=True, save_code=True,
               config={"args": vars(args), "overrides": overrides,
                       "algo": ctrl["algos"][args.algo], "tof_net": ctrl["tof_net"]})
    return WandbCallback(model_save_path=str(out / "wandb_model"),
                         model_save_freq=wb["model_save_every"] // args.n_envs, verbose=1)


def main():
    with open(config.CONFIG_DIR / "control.toml", "rb") as f:
        ctrl = tomllib.load(f)
    tr = ctrl["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=tr["algo"], choices=list(ALGOS))
    ap.add_argument("--n-envs", type=int, default=tr["n_envs"])
    ap.add_argument("--steps", type=int, default=tr["total_steps"])
    ap.add_argument("--device", default=tr["device"])
    ap.add_argument("--seed", type=int, default=tr["seed"])
    ap.add_argument("--net", choices=["mlp", "tof"], default="mlp",
                    help="mlp: SB3 MLP on all obs; tof: tof_net.py on raw ToF only")
    ap.add_argument("--set", action="append", help="env/sim override section.key=value")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction,
                    default=ctrl["wandb"]["enabled"], help="log to Weights & Biases")
    ap.add_argument("--wandb-mode", default=ctrl["wandb"]["mode"],
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--resume", default=None,
                    help="continue a run from its final.zip or ckpt/<algo>_<n>_steps.zip; "
                         "--steps more steps, same run folder, algo, net and env overrides")
    args = ap.parse_args()

    meta = {}
    if args.resume:
        out, meta, norm_path, buf_path = run_files(args.resume)
        args.algo, args.net = meta["algo"], meta["args"]["net"]
        overrides = meta["overrides"]
    else:
        net = ctrl["tof_net"] if args.net == "tof" else None
        overrides = {**(net["env"] if net else {}), **parse_sets(args.set)}
        out = config.ROOT / (args.out or
                             f"runs/{args.algo}_{args.net}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    callbacks = [EpisodeStats()]
    if args.wandb:
        load_dotenv(config.ROOT / ".env")
        callbacks.append(start_wandb(ctrl["wandb"], out, args, overrides, ctrl,
                                     meta.get("wandb_id")))
        import wandb
        meta["wandb_id"] = wandb.run.id
    meta.update(algo=args.algo, overrides=overrides,
                args=meta.get("args", vars(args)), resumed=meta.get("resumed", 0) + bool(args.resume))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    venv = make_vec_env(BallOnPlateEnv, n_envs=args.n_envs, seed=args.seed,
                        env_kwargs={"overrides": overrides}, vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs={"start_method": "forkserver"})
    venv = VecMonitor(venv)
    algo = ALGOS[args.algo]
    off_policy = args.algo in ("sac", "td3", "ddpg")
    if args.resume:
        venv = VecNormalize.load(str(norm_path), venv)
        model = algo.load(args.resume, env=venv, device=args.device, tensorboard_log=str(out))
        if off_policy and buf_path.exists():
            model.load_replay_buffer(str(buf_path))
        print(f"resumed {args.resume} at {model.num_timesteps} steps"
              + (f", replay buffer {model.replay_buffer.size()}" if off_policy else ""))
    else:
        venv = VecNormalize(venv, norm_obs=tr["norm_obs"], norm_reward=tr["norm_reward"])
        model = algo("MultiInputPolicy", venv, device=args.device, seed=args.seed,
                     tensorboard_log=str(out), verbose=1,
                     **hyperparams(args.algo, ctrl["algos"][args.algo], ctrl["tof_net"]
                                   if args.net == "tof" else None))
    every = max(tr["checkpoint_every"] // args.n_envs, 1)
    ckpt = CheckpointCallback(every, str(out / "ckpt"), name_prefix=args.algo,
                              save_vecnormalize=True,
                              save_replay_buffer=off_policy and tr["ckpt_replay_buffer"])
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, *callbacks],
                    progress_bar=False, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("interrupted: saving final.zip so the run can be resumed")
    model.save(out / "final.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    if off_policy and tr["save_replay_buffer"]:
        model.save_replay_buffer(str(out / "replay_buffer.pkl"))
    try:
        venv.close()
    except (EOFError, BrokenPipeError):
        pass  # Ctrl-C already stopped the worker processes
    if args.wandb:
        import wandb
        wandb.finish()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
