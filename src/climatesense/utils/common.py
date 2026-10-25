"""Shared helpers: paths, config loading, atomic IO, seeding, geodesy."""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing variables."""
    env = path or ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_yaml(name: str) -> dict:
    p = Path(name)
    if not p.is_absolute():
        p = ROOT / "configs" / (name if name.endswith(".yaml") else f"{name}.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def atomic_write_json(obj, path: Path | str) -> None:
    """Write JSON via a temp file + rename so interrupted runs never leave a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
    os.replace(tmp, path)


def read_json(path: Path | str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised, broadcasting)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
