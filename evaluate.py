#!/usr/bin/env python
"""Evaluate a checkpoint without leaking an oracle threshold into the headline."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrace.data import build_loader
from retrace.engine import evaluate_model
from retrace.losses import ReTraceLoss
from retrace.model import build_model
from retrace.utils import choose_device, load_checkpoint, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(config["model"]).to(device)
    state = checkpoint["model"]
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    load_result = model.load_state_dict(compatible_state, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing or unexpected or len(compatible_state) != len(state):
        print(
            f"non-strict checkpoint load: loaded={len(compatible_state)} "
            f"missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    loader_config = {**config["data"], "batch_size": config["evaluation"]["batch_size"]}
    loader = build_loader(loader_config, args.split, shuffle=False)
    losses, metrics = evaluate_model(
        model,
        criterion,
        loader,
        device,
        bool(config["training"].get("amp", True) and device.type == "cuda"),
        float(config["evaluation"].get("threshold", 0.5)),
        int(config["evaluation"].get("thresholds", 101)),
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "samples": len(loader.dataset),
        "losses": losses,
        "metrics": metrics,
        "warning": "oracle metrics are diagnostic; select thresholds on validation data only",
    }
    write_json(payload, args.output)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
