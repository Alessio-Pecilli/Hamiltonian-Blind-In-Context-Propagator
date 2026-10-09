"""Config loading: common section + preset, with dotted --set overrides."""
from __future__ import annotations

import copy
import subprocess

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str = "configs/config.yaml", preset: str = "local",
                overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = deep_update(raw["common"], raw["presets"][preset])
    cfg["preset"] = preset
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *path_, last = key.split(".")
        for p in path_:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"
