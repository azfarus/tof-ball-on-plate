"""Stable-Baselines3 VecEnv adapter for the batched MuJoCo Warp simulation."""
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from bop_env import config


def make_warp_sim(n_envs, overrides=None, seed=0):
    from bop_env.gpu.sim import WarpSim
    cfg = config.load_all()
    cfg = config.apply(cfg, cfg["env"].get("overrides", {}))
    cfg = config.apply(cfg, overrides)
    if cfg["env"].get("randomize"):
        raise NotImplementedError("Warp physics does not support [env.randomize] yet")
    return WarpSim(cfg, n_envs, seed=seed)


def _split_info(info, n):
    out = []
    for i in range(n):
        row = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n:
                row[k] = v[i].copy()
            else:
                row[k] = v
        out.append(row)
    return out


class WarpVecEnv(VecEnv):
    def __init__(self, sim, seed=0):
        self.sim = sim
        self.render_mode = None
        self._base_seed = int(seed)
        self._actions = None
        super().__init__(sim.b, sim.observation_space, sim.action_space)

    def reset(self):
        obs, info = self.sim.reset(seed=self._base_seed)
        self.reset_infos = _split_info(info, self.num_envs)
        self._base_seed += self.num_envs
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions, np.float32)

    def step_wait(self):
        obs, rew, term, trunc, info = self.sim.step(self._actions)
        done = np.logical_or(term, trunc)
        infos = _split_info(info, self.num_envs)
        if done.any():
            idx = np.flatnonzero(done)
            terminal = {k: v[done].copy() for k, v in obs.items()}
            self.sim.reset(indices=idx, seed=self._base_seed)
            obs = self.sim.obs()
            for j, i in enumerate(idx):
                infos[i]["terminal_observation"] = {k: v[j].copy() for k, v in terminal.items()}
            self._base_seed += self.num_envs
        return obs, np.asarray(rew, np.float32), done, infos

    def get_attr(self, attr_name, indices=None):
        value = getattr(self.sim, attr_name, getattr(self, attr_name, None))
        return [value for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.sim, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.sim, method_name)(*args, **kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        self._base_seed = 0 if seed is None else int(seed)
        return [self._base_seed + i for i in range(self.num_envs)]
