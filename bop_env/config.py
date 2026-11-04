"""Load TOML configs from config/ and apply overrides.

Overrides are a dict. Keys are dotted paths ("plate.ball_mass_kg",
"servos.MG996R.deadband_us") or section names with a nested dict
({"scan": {"corners": [0, 2]}}), merged into the loaded configs.
"""
import copy
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SECTIONS = ["plate", "sensor", "scan", "tracker", "scenarios", "platform", "servos", "env"]


def load(name, config_dir=CONFIG_DIR):
    with open(pathlib.Path(config_dir) / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply(cfg, overrides):
    """Copy of cfg with overrides applied."""
    cfg = copy.deepcopy(cfg)
    for key, val in (overrides or {}).items():
        path = key.split(".")
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        if isinstance(val, dict) and isinstance(node.get(path[-1]), dict):
            _merge(node[path[-1]], val)
        else:
            node[path[-1]] = copy.deepcopy(val)
    return cfg


def load_all(overrides=None, config_dir=CONFIG_DIR):
    """All config sections as one dict, with overrides applied."""
    return apply({n: load(n, config_dir) for n in SECTIONS}, overrides)
