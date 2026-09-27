"""Training and evaluation loops."""

from __future__ import annotations

import random
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .degradations import degrade_batch
from .metrics import ForensicMetrics
from .utils import RunningAverages


def _forward_from_batch(model: nn.Module, image: torch.Tensor, batch: dict, device: torch.device, intervention_mask: torch.Tensor | None = None) -> dict:
    kwargs = {"intervention_mask": intervention_mask}
    if "dct" in batch:
        kwargs["dct"] = batch["dct"].to(device, non_blocking=True)
    if "qtable" in batch:
        kwargs["qtable"] = batch["qtable"].to(device, non_blocking=True)
    if "dct_available" in batch:
        kwargs["dct_available"] = batch["dct_available"].to(device, non_blocking=True)
    return model(image, **kwargs)


def _assert_finite_losses(losses: dict[str, torch.Tensor], batch_index: int, epoch: int) -> None:
    bad = []
    for name, value in losses.items():
        if torch.isfinite(value).all():
            continue
        safe = value.detach().float().nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)
        bad.append(f"{name}=min{safe.min().item():.6g}/max{safe.max().item():.6g}")
    if bad:
        raise FloatingPointError(
            f"non-finite ReTrace losses at epoch={epoch} batch={batch_index}: "
            + ", ".join(bad)
        )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    consistency_probability: float,
    intervention_teacher_probability: float,
    accumulation_steps: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    averages = RunningAverages()
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    accumulation_steps = max(1, int(accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    loader_length = len(loader)
    for batch_index, batch in enumerate(progress):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        teacher_probability = intervention_teacher_probability * (0.95**epoch)
        intervention_mask = mask if random.random() < teacher_probability else None
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = _forward_from_batch(model, image, batch, device, intervention_mask=intervention_mask)
            degraded_outputs = None
            if random.random() < consistency_probability:
                degraded_outputs = _forward_from_batch(
                    model, degrade_batch(image), batch, device, intervention_mask=intervention_mask
                )
            losses = criterion(outputs, mask, label, degraded_outputs)
            _assert_finite_losses(losses, batch_index, epoch)

        group_start = (batch_index // accumulation_steps) * accumulation_steps
        current_accumulation = min(accumulation_steps, loader_length - group_start)
        scaled_loss = losses["total"] / current_accumulation
        scaler.scale(scaled_loss).backward()

        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or (batch_index + 1) == loader_length
        )
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        averages.update(losses, image.shape[0])
        progress.set_postfix(loss=f"{float(losses['total'].detach()):.4f}")
    return averages.compute()


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    fixed_threshold: float = 0.5,
    threshold_count: int = 101,
    image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    gate_threshold: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    averages = RunningAverages()
    metrics = ForensicMetrics(fixed_threshold=fixed_threshold, gate_threshold=gate_threshold)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if image_transform is not None:
            image = image_transform(image)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = _forward_from_batch(model, image, batch, device)
            losses = criterion(outputs, mask, label)
        averages.update(losses, image.shape[0])
        metrics.update(outputs, mask, label)
    return averages.compute(), metrics.compute(threshold_count)
