"""SB3 callbacks for train.py: episode stats, rolling checkpoints, best model."""
import json
import pathlib
import re

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class EpisodeStats(BaseCallback):
    """Logs fall-off rate and final distance to target per finished episode."""

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.logger.record_mean("episode/fell", float(info["escaped"]))
                self.logger.record_mean("episode/final_err_mm", 1000 * info["err_m"])
                self.logger.record_mean("episode/length_s", info["t"])
        return True


class RollingCheckpoint(CheckpointCallback):
    """CheckpointCallback that keeps only the newest `keep` checkpoints
    (model zip plus its vecnormalize / replay buffer pkl)."""

    def __init__(self, keep, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep = keep

    def _on_step(self):
        saved = self.n_calls % self.save_freq == 0
        ok = super()._on_step()
        if saved:
            self.prune()
        return ok

    def prune(self):
        files = list(pathlib.Path(self.save_path).glob(f"{self.name_prefix}_*_steps.*"))
        step = lambda f: int(re.search(r"_(\d+)_steps\.", f.name).group(1))
        steps = sorted({step(f) for f in files})
        for f in files:
            if step(f) in steps[:-self.keep]:
                f.unlink()


class SaveBest(BaseCallback):
    """Saves best.zip + best_vecnormalize.pkl when the mean raw return of the
    last episodes (VecMonitor buffer, up to 100) beats the best so far.
    Training-time returns: the policy still explores (SAC samples, TD3 noise)."""

    def __init__(self, out, check_every, min_episodes, verbose=0):
        super().__init__(verbose)
        self.out, self.every, self.min_eps = out, check_every, min_episodes
        info = out / "best.json"
        self.best = json.loads(info.read_text())["mean_return"] if info.exists() else -np.inf

    def _on_step(self):
        if self.n_calls % self.every:
            return True
        eps = self.model.ep_info_buffer
        if len(eps) < self.min_eps:
            return True
        ret = float(np.mean([e["r"] for e in eps]))
        if ret > self.best:
            self.best = ret
            self.model.save(self.out / "best.zip")
            self.model.get_vec_normalize_env().save(str(self.out / "best_vecnormalize.pkl"))
            (self.out / "best.json").write_text(json.dumps(
                {"step": self.num_timesteps, "mean_return": ret, "episodes": len(eps)}, indent=1))
            if self.verbose:
                print(f"new best {ret:.1f} at {self.num_timesteps} steps")
        self.logger.record("best/mean_return", self.best)
        return True
