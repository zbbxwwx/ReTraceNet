#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import pil_to_tensor, read_manifest
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.final_modules import build_final_model, dice_loss_from_prob


class SyntheticHistoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[Any], image_size: int, min_rect: float = 0.18, max_rect: float = 0.55) -> None:
        self.records = records
        self.image_size = int(image_size)
        self.min_rect = float(min_rect)
        self.max_rect = float(max_rect)

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, image: Image.Image) -> Image.Image:
        target = self.image_size
        scale = random.uniform(0.85, 1.35)
        shorter = max(1, min(image.size))
        ratio = target * scale / shorter
        new_size = (max(target, round(image.width * ratio)), max(target, round(image.height * ratio)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        left = random.randint(0, image.width - target)
        top = random.randint(0, image.height - target)
        image = image.crop((left, top, left + target, top + target))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    @staticmethod
    def _jpeg(image: Image.Image, quality: int) -> Image.Image:
        import io

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _process_patch(self, patch: Image.Image, process_id: int) -> Image.Image:
        if process_id == 0:
            return self._jpeg(patch, random.randint(35, 75))
        if process_id == 1:
            return patch.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.6)))
        if process_id == 2:
            return ImageEnhance.Contrast(patch).enhance(random.uniform(0.65, 1.45))
        return ImageEnhance.Color(patch).enhance(random.uniform(0.5, 1.6))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.image) as handle:
            image = self._crop(handle.convert("RGB"))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
        process_id = random.randint(0, 3)
        target = self.image_size
        rw = random.randint(int(target * self.min_rect), int(target * self.max_rect))
        rh = random.randint(int(target * self.min_rect), int(target * self.max_rect))
        left = random.randint(0, target - rw)
        top = random.randint(0, target - rh)
        patch = image.crop((left, top, left + rw, top + rh))
        patch = self._process_patch(patch, process_id)
        mixed = image.copy()
        mixed.paste(patch, (left, top))
        mask = Image.new("L", (target, target), color=0)
        mask.paste(Image.new("L", (rw, rh), color=255), (left, top))
        return {
            "image": pil_to_tensor(mixed),
            "mask": (pil_to_tensor(mask) > 0.5).float(),
            "process": torch.tensor(process_id, dtype=torch.long),
        }


class ForensicPretrainHead(nn.Module):
    def __init__(self, channels: tuple[int, ...], hidden: int = 96, classes: int = 4) -> None:
        super().__init__()
        self.proj = nn.ModuleList(nn.Conv2d(int(c), hidden, 1) for c in channels[:3])
        self.mask_head = nn.Sequential(nn.Conv2d(hidden * 3, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden * 3, classes))

    def forward(self, detail: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        size = detail[0].shape[-2:]
        feats = []
        for layer, x in zip(self.proj, detail[:3]):
            y = layer(x)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            feats.append(y)
        fused = torch.cat(feats, dim=1)
        return self.mask_head(fused), self.cls_head(fused)


def load_initial(model: nn.Module, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 3407)))
    device = choose_device(args.device)
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    model = build_final_model(cfg["model"]).to(device)
    load_initial(model, args.init or cfg["training"].get("init_checkpoint"))
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("forensic_stream."))
    channels = tuple(int(x) for x in cfg["model"].get("forensic_channels", (64, 128, 256, 512)))
    head = ForensicPretrainHead(channels).to(device)
    records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "train")
    dataset = SyntheticHistoryDataset(
        records,
        int(cfg["data"].get("image_size", 640)),
        float(cfg["data"].get("pretrain_min_rect", 0.18)),
        float(cfg["data"].get("pretrain_max_rect", 0.55)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=int(cfg["data"].get("num_workers", 4)) > 0,
    )
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["training"].get("learning_rate", 1e-4)), weight_decay=float(cfg["training"].get("weight_decay", 0.01)))
    amp = bool(cfg["training"].get("amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    epochs = int(cfg["training"].get("epochs", 3))
    grad_acc = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    history = []
    for epoch in range(epochs):
        model.train()
        head.train()
        sums: dict[str, float] = {}
        count = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            process = batch["process"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                detail = model.forensic_stream(image)
                logits_low, cls_logits = head(detail)
                target = F.interpolate(mask, size=logits_low.shape[-2:], mode="area").clamp(0, 1)
                prob = logits_low.sigmoid()
                loc = F.binary_cross_entropy_with_logits(logits_low, target) + dice_loss_from_prob(prob, target)
                cls = F.cross_entropy(cls_logits, process)
                loss = loc + float(cfg["training"].get("class_loss_weight", 0.2)) * cls
            scaler.scale(loss / grad_acc).backward()
            if (step + 1) % grad_acc == 0 or step + 1 == len(loader):
                if float(cfg["training"].get("grad_clip", 0.0)) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(cfg["training"].get("grad_clip", 1.0)))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            bs = image.shape[0]
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * bs
            sums["localization"] = sums.get("localization", 0.0) + float(loc.detach().cpu()) * bs
            sums["classification"] = sums.get("classification", 0.0) + float(cls.detach().cpu()) * bs
            count += bs
        row = {k: v / max(1, count) for k, v in sums.items()}
        row["epoch"] = epoch
        history.append(row)
        print(yaml.safe_dump(row, sort_keys=False).strip(), flush=True)
        model_state = copy.deepcopy(model.state_dict())
        forensic_only = {k: v.detach().cpu() for k, v in model_state.items() if k.startswith("forensic_stream.")}
        atomic_torch_save(
            {
                "model": forensic_only,
                "full_model": {k: v.detach().cpu() for k, v in model_state.items()},
                "pretrain_head": head.state_dict(),
                "config": cfg,
                "history": history,
            },
            outdir / "last_forensic_pretrain.pt",
        )
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
