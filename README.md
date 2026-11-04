# Ball-on-plate with ToF sensing

This repository simulates closed-loop 2-D ball-on-plate balancing, where time-of-flight sensors locate the ball:
- a 40 mm ball on a 30 cm plate;
- three hobby PWM servos (3-RRS) tilt the plate;
- four corner VL53L4CD ToF sensors track the ball, fused by an EKF.

It is built on MuJoCo, provides a Gymnasium environment (`BallOnPlateToF-v0`), and trains with Stable-Baselines3.

Docs:
- `docs/ENV.md` is the full environment reference: action, observation, reward, servo, ToF and EKF models, and config keys.
- `CLAUDE.md` covers the architecture and the study results.

All commands run from the repository root.

## 1. Install

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). The GPU training stack is torch with CUDA 13.2. For a CPU-only install, remove the `--extra-index-url` line from `requirements.txt`.

```bash
uv venv .venv -p 3.13
uv pip install --python .venv/bin/python -r requirements.txt     # also installs bop_env (editable)
.venv/bin/python -c "import bop_env, torch; print('ok, cuda =', torch.cuda.is_available())"
```

With pip instead of uv: `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## 2. Weights & Biases (training dashboard)

Put your API key (https://wandb.ai/authorize) in `.env` in the repo root. That file is git-ignored; `.env.example` is the template:

```bash
cp .env.example .env      # then edit: WANDB_API_KEY=<your key>
```

`control/train.py` loads `.env` at start. Alternatively, `.venv/bin/wandb login` stores the key in `~/.netrc`.

Every `control/train.py` run then logs to the W&B project `ball-on-plate-tof`, set in `[wandb]` in `config/control.toml`. Each run logs:
- SB3 scalars: `rollout/ep_rew_mean`, `train/*` losses, fps;
- episode stats: `episode/fell` (fraction of episodes where the ball fell off), `episode/final_err_mm`, `episode/length_s`;
- the config, the code, and model snapshots every 200k steps.

| Want | Flag |
|---|---|
| No internet | `--wandb-mode offline`, later `.venv/bin/wandb sync runs/<run>/wandb/offline-run-*` |
| No W&B at all | `--no-wandb` |
| Other team / project | edit `entity` / `project` in `[wandb]` |

TensorBoard works too, without an account: `.venv/bin/tensorboard --logdir runs`.

## 3. Train

```bash
# 3-layer LayerNorm + dropout net on the raw ToF readings only, SAC (continuous actions)
.venv/bin/python control/train.py --algo sac --net tof --n-envs 14 --steps 1000000 --out runs/sac_tof_1m

# SB3 default MLP on all observations (EKF estimate, ToF, last action, target)
.venv/bin/python control/train.py --algo ppo --n-envs 14 --steps 2000000
```

| Option | Meaning |
|---|---|
| `--algo` | `sac`, `td3`, `ddpg`, `ppo`, `a2c` (hyperparameters in `[algos.*]`, `config/control.toml`) |
| `--net` | `tof`: `control/tof_net.py` on `obs["tof"]` only (settings in `[tof_net]`); `mlp`: SB3 MLP on everything |
| `--n-envs` | Parallel headless env processes (physics on CPU, one core each) |
| `--steps` | Total env steps |
| `--device` | `cuda` (default) or `cpu`, for the network |
| `--set k=v` | Override any env or sim config value, e.g. `--set scan.actuator=ideal`, `--set platform.servo_model=DS3218` |
| `--out` | Run folder (default `runs/<algo>_<net>_<time>`) |

Throughput is about 335 env steps/s with 14 workers, so 1M steps take roughly 1 h.

Each run folder `runs/<run>/` holds:
- `final.zip`, the model;
- `vecnormalize.pkl`, the observation statistics, which evaluation needs;
- `meta.json`, the algo and env overrides;
- `ckpt/`, checkpoints every 200k steps;
- the TensorBoard logs and `wandb/`.

## 4. Evaluate

```bash
.venv/bin/python control/evaluate.py --controller model --path runs/sac_tof_1m/final.zip --episodes 10
.venv/bin/python control/evaluate.py --controller model --path runs/sac_tof_1m/ckpt/sac_<steps>_steps.zip   # a checkpoint
.venv/bin/python control/evaluate.py --controller pid --episodes 10      # non-learned baseline
.venv/bin/python control/evaluate.py --controller zero                   # plate held flat
.venv/bin/python control/evaluate.py --controller pid --view             # live MuJoCo viewer
```

Evaluation applies the training run's env overrides from `meta.json` automatically. It prints per-episode return, whether the ball fell, RMS distance to the target, and EKF error. The PID baseline holds the ball in every episode at 26–29 mm RMS. With zero action the ball falls off in about 2 s.

## 5. Use the environment directly

```python
import gymnasium as gym, bop_env

env = gym.make("BallOnPlateToF-v0", overrides={"scan.actuator": "ideal"})
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

venv = bop_env.make_vec(16)          # 16 headless copies, one process each
```

Scripts that start worker processes need an `if __name__ == "__main__":` guard.

## 6. Simulation studies (scripted tilt, no controller)

```bash
.venv/bin/python tools/calibrate.py                    # ToF vs datasheet, rolling physics, servo step times
.venv/bin/python tools/run_scenarios.py                # all scenarios, interference on + off -> sim/out/
.venv/bin/python tools/run_scenarios.py --scenario circle --interference on --view
.venv/bin/python tools/ablate.py                       # variants in config/ablation.toml, parallel
```

`sim/out/` keeps the plots and summary CSVs. The per-run trace and readings CSVs are not in git; rerun the tools to regenerate them.

## Layout

| Path | Contents |
|---|---|
| `bop_env/` | Simulation and Gymnasium env (no learning code) |
| `control/` | PID, ToF net, SB3 training, evaluation |
| `tools/` | Scripted scenario runs, ablation, calibration, plots |
| `config/` | Every parameter, one TOML per part |
| `docs/ENV.md` | Environment reference |
