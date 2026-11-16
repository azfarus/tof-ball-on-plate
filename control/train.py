"""Train an RL agent on the env with Stable-Baselines3.

Training uses batched MuJoCo Warp physics on the GPU by default. The policy
network also trains on the GPU. Use --physics cpu for the original one-process-
per-environment CPU backend. Settings are in config/control.toml [train] and
[algos.*]; any env/sim config value can be overridden with --set.

Usage (from project root):
  .venv/bin/python control/train.py --algo ppo --physics warp --steps 2000000
  .venv/bin/python control/train.py --algo sac --physics cpu --n-envs 14
  .venv/bin/python control/train.py --algo sac --set scan.actuator=ideal
  .venv/bin/python control/train.py --algo sac --net tof   # raw-ToF 3-layer net (tof_net.py)
  .venv/bin/python control/train.py --algo sac --net res   # wide residual MLP (res_net.py)
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
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from bop_env import BallOnPlateEnv, config
from callbacks import EpisodeStats, RollingCheckpoint, SaveBest
from evaluate import parse_sets, run_files
from res_net import ResMLP
from tof_net import ToFNet

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "ddpg": DDPG}


def hyperparams(algo, hp, kind="mlp", net=None):
    """SB3 kwargs from the [algos.<name>] table, plus [tof_net] or [res_net] for --net tof/res."""
    hp = dict(hp)
    kw = {"policy_kwargs": {"net_arch": hp.pop("net_arch")}}
    critic = "vf" if algo in ("ppo", "a2c") else "qf"
    if kind == "tof":
        kw["policy_kwargs"] = {
            "features_extractor_class": ToFNet,
            "features_extractor_kwargs": {"layers": net["layers"], "dropout": net["dropout"]},
            "net_arch": {"pi": net["actor_head"], critic: net["critic_head"]}}
    elif kind == "res":
        kw["policy_kwargs"] = {
            "features_extractor_class": ResMLP,
            "features_extractor_kwargs": {k: net[k] for k in
                                          ("width", "blocks", "expansion", "dropout", "keys")},
            "net_arch": {"pi": net["actor_head"], critic: net["critic_head"]},
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": net["weight_decay"]}}
    sigma = hp.pop("action_noise_sigma", None)
    if sigma:
        kw["action_noise"] = NormalActionNoise(np.zeros(3), sigma * np.ones(3))
    return {**kw, **hp}


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
                       "algo": ctrl["algos"][args.algo], "tof_net": ctrl["tof_net"],
                       "res_net": ctrl["res_net"]})
    return WandbCallback(model_save_path=str(out / "wandb_model"),
                         model_save_freq=wb["model_save_every"] // args.n_envs, verbose=1)


def main():
    with open(config.CONFIG_DIR / "control.toml", "rb") as f:
        ctrl = tomllib.load(f)
    tr = ctrl["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default=tr["algo"], choices=list(ALGOS))
    ap.add_argument("--physics", choices=["warp", "cpu"], default=tr.get("physics", "cpu"))
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="SB3 minibatch size")
    ap.add_argument("--utd", type=float, default=None, help="gradient updates per env transition")
    ap.add_argument("--hp", action="append",
                    help="algo hyperparameter key=value, e.g. --hp buffer_size=4000000 "
                         "--hp gradient_steps=16 (overrides [algos.*] and --utd)")
    ap.add_argument("--steps", type=int, default=tr["total_steps"])
    ap.add_argument("--device", default=tr["device"])
    ap.add_argument("--seed", type=int, default=tr["seed"])
    ap.add_argument("--net", choices=["mlp", "tof", "res"], default="mlp",
                    help="mlp: SB3 MLP on all obs; tof: tof_net.py on raw ToF only; "
                         "res: res_net.py wide residual MLP on all obs")
    ap.add_argument("--width", type=int, default=None, help="res net width ([res_net])")
    ap.add_argument("--blocks", type=int, default=None, help="res net residual blocks")
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
    torch.backends.cuda.matmul.allow_tf32 = tr["tf32"]  # TF32 tensor cores for the big nets
    torch.backends.cudnn.allow_tf32 = tr["tf32"]
    nets = {"tof": dict(ctrl["tof_net"]), "res": dict(ctrl["res_net"])}
    for k in ("width", "blocks"):
        if getattr(args, k) is not None:
            nets["res"][k] = getattr(args, k)

    meta = {}
    if args.resume:
        out, meta, norm_path, buf_path = run_files(args.resume)
        args.algo, args.net = meta["algo"], meta["args"]["net"]
        args.physics = meta["args"].get("physics", "cpu")
        overrides = meta["overrides"]
        if args.n_envs is None:
            args.n_envs = meta["args"].get("n_envs", tr["n_envs"])
    else:
        net = nets.get(args.net)
        overrides = {**(net["env"] if net else {}), **parse_sets(args.set)}
        out = config.ROOT / (args.out or
                             f"runs/{args.algo}_{args.net}_{time.strftime('%Y%m%d_%H%M%S')}")
        if args.n_envs is None:
            args.n_envs = tr.get("warp_envs", tr["n_envs"]) if args.physics == "warp" else tr["n_envs"]
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

    if args.physics == "warp":
        from warp_vec import WarpVecEnv, make_warp_sim
        venv = WarpVecEnv(make_warp_sim(args.n_envs, overrides, seed=args.seed), seed=args.seed)
        venv = VecMonitor(venv)
    else:
        venv = make_vec_env(BallOnPlateEnv, n_envs=args.n_envs, seed=args.seed,
                            env_kwargs={"overrides": overrides}, vec_env_cls=SubprocVecEnv,
                            vec_env_kwargs={"start_method": "forkserver"})
        venv = VecMonitor(venv)
    algo = ALGOS[args.algo]
    off_policy = args.algo in ("sac", "td3", "ddpg")
    hp = dict(ctrl["algos"][args.algo])
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if off_policy:
        utd = args.utd if args.utd is not None else tr.get("utd")
        if utd is not None:
            hp["gradient_steps"] = max(1, int(round(float(utd) * args.n_envs)))
    hp.update(parse_sets(args.hp))
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
                     **hyperparams(args.algo, hp, args.net, nets.get(args.net)))
    n_par = sum(p.numel() for p in model.policy.parameters())
    print(f"policy parameters (actor + critics + targets): {n_par / 1e6:.1f} M")
    every = max(tr["checkpoint_every"] // args.n_envs, 1)
    ckpt = RollingCheckpoint(tr["keep_last_ckpt"], every, str(out / "ckpt"),
                             name_prefix=args.algo, save_vecnormalize=True,
                             save_replay_buffer=off_policy and tr["ckpt_replay_buffer"])
    best = SaveBest(out, max(tr["best_check_every"] // args.n_envs, 1),
                    tr["best_min_episodes"], verbose=1)
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt, best, *callbacks],
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
