"""Config loading. Every threshold lives in configs/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load the default config, then an optional override file, then
    ``section.key=value`` overrides (values parsed as YAML)."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg
