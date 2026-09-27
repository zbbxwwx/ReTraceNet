"""Configuration, reproducibility, and checkpoint helpers."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    base_value = config.pop("base_config", None)
    if base_value:
        base_path = Path(base_value)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = load_config(base_path)

        def merge(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
            for key, value in update.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key] = merge(dict(target[key]), value)
                else:
                    target[key] = value
            return target

        config = merge(base, config)
    return config


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(finite(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class RunningAverages:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, torch.Tensor], batch_size: int) -> None:
        self.count += batch_size
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value.detach()) * batch_size

    def compute(self) -> dict[str, float]:
        return {key: value / max(1, self.count) for key, value in self.totals.items()}
