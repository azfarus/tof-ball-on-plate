"""Ball-on-plate with scanning ToF sensors: MuJoCo simulation and RL environment.

    import gymnasium as gym, bop_env
    env = gym.make("BallOnPlateToF-v0", overrides={"platform.servo_model": "DS3218"})
    venv = bop_env.make_vec(16)            # headless, one process per env
"""
import gymnasium as gym

from .config import load_all
from .sim import Sim
from .env import BallOnPlateEnv, make_vec

gym.register(id="BallOnPlateToF-v0", entry_point="bop_env.env:BallOnPlateEnv")

__all__ = ["BallOnPlateEnv", "Sim", "load_all", "make_vec"]
