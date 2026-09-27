"""E25: splicing-oriented compression history discrepancy pretraining.

This script keeps the ReTraceNet innovation path intact.  It first teaches the
existing forensic stream (plus the DCT output adapters) to localize synthetic
local JPEG-history discrepancies, then fine-tunes the existing QTR/HCV/DHER and
decoder stack on CASIA with small learning rates.  No test evaluation is run.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_e21_jpeg_history_branch as e21
from retrace.data import pil_to_tensor, read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, dher_v2_aux_loss, tversky_loss, train_loader_for, weighted_deep_supervision_loss


EXPERIMENT = "E25_compression_discrepancy_pretrain"
SOURCE_BEST_F1 = 0.7982278425423237


def dump(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), subsampling=0)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_color_jitter(image: Image.Image, strength: float = 0.12) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - strength, 1 + strength))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - strength, 1 + strength))
    image = ImageEnhance.Color(image).enhance(random.uniform(1 - strength, 1 + strength))
    return image


def dct_volume_from_coef(coef: np.ndarray, height: int, width: int) -> torch.Tensor:
    coef = coef[:height, :width].astype("float32", copy=False)
    dct = np.zeros((21, height, width), dtype=np.float32)
    dct[0] = (coef == 0)
    abs_coef = np.abs(coef)
    for value in range(1, 20):
        dct[value] = (abs_coef == value)
    dct[20] = (abs_coef >= 20)
    return torch.from_numpy(dct)


def jpeg_info_from_pil(image: Image.Image, quality: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import jpegio
    rgb = image.convert("RGB")
    width, height = rgb.size
    pad_w = ((width + 7) // 8) * 8
    pad_h = ((height + 7) // 8) * 8
    if (pad_w, pad_h) != (width, height):
        padded = Image.new("RGB", (pad_w, pad_h), color=(128, 128, 128))
        padded.paste(rgb, (0, 0))
        rgb = padded
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            tmp_name = handle.name
        rgb.save(tmp_name, format="JPEG", quality=int(quality), subsampling=0)
        jpeg = jpegio.read(tmp_name)
        coef = jpeg.coef_arrays[0].astype("float32")
        ci = jpeg.comp_info[0]
        qtable = jpeg.quant_tables[ci.quant_tbl_no].astype("float32")
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return dct_volume_from_coef(coef, height, width), torch.from_numpy(qtable[None, :, :]), torch.tensor([1.0], dtype=torch.float32)


class CompressionDiscrepancyGenerator:
    def __init__(self, records, image_size: int = 640) -> None:
        self.records = [r for r in records if r.image.exists()]
        if not self.records:
            raise RuntimeError("no usable training images for compression discrepancy generation")
        self.image_size = int(image_size)

    def _load_image(self, index: int | None = None) -> Image.Image:
        record = self.records[random.randrange(len(self.records))] if index is None else self.records[index % len(self.records)]
        with Image.open(record.image) as handle:
            image = handle.convert("RGB")
        # Cover both full-image context and local patches while avoiding tiny degenerate crops.
        short = max(1, min(image.size))
        scale = random.uniform(0.9, 1.4)
        ratio = self.image_size * scale / short
        new_size = (max(self.image_size, round(image.width * ratio)), max(self.image_size, round(image.height * ratio)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        if image.width > self.image_size or image.height > self.image_size:
            left = random.randint(0, image.width - self.image_size)
            top = random.randint(0, image.height - self.image_size)
            image = image.crop((left, top, left + self.image_size, top + self.image_size))
        return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    def _mask_and_box(self) -> tuple[Image.Image, tuple[int, int, int, int], float]:
        size = self.image_size
        bucket = random.choices(["lt_1", "1_5", "5_20", "gte_20"], weights=[0.12, 0.28, 0.32, 0.28], k=1)[0]
        if bucket == "lt_1":
            area = random.uniform(0.004, 0.01)
        elif bucket == "1_5":
            area = random.uniform(0.012, 0.05)
        elif bucket == "5_20":
            area = random.uniform(0.055, 0.20)
        else:
            area = random.uniform(0.20, 0.45)
        aspect = math.exp(random.uniform(math.log(0.45), math.log(2.2)))
        h = int(round(size * math.sqrt(area / aspect)))
        w = int(round(size * math.sqrt(area * aspect)))
        h = max(12, min(size - 4, h))
        w = max(12, min(size - 4, w))
        left = random.randint(0, size - w)
        top = random.randint(0, size - h)
        box = (left, top, left + w, top + h)
        mask = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(mask)
        if random.random() < 0.5:
            draw.ellipse(box, fill=255)
        else:
            radius = random.randint(0, max(1, min(w, h) // 5))
            try:
                draw.rounded_rectangle(box, radius=radius, fill=255)
            except Exception:
                draw.rectangle(box, fill=255)
        if random.random() < 0.7:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.4)))
            mask = mask.point(lambda p: 255 if p > 96 else 0)
        return mask, box, area

    def make(self) -> dict[str, torch.Tensor | int | float]:
        host_q = random.choice([82, 86, 90, 94, 98])
        patch_q = random.choice([35, 45, 55, 65, 75])
        if random.random() < 0.25:
            # Include reverse cases so the encoder learns discrepancy, not just low-quality patches.
            host_q, patch_q = patch_q, host_q
        final_q = random.choice([70, 78, 86, 94, 100])
        host = jpeg_roundtrip(self._load_image(), host_q)
        source = jpeg_roundtrip(self._load_image(), patch_q)
        source = random_color_jitter(source, strength=random.uniform(0.04, 0.16))
        if random.random() < 0.35:
            source = source.filter(ImageFilter.GaussianBlur(random.uniform(0.15, 0.8)))
        mask, box, area = self._mask_and_box()
        patch = source.crop(box)
        alpha = mask.crop(box).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
        mixed = host.copy()
        mixed.paste(patch, box[:2], alpha)
        if random.random() < 0.85:
            mixed = jpeg_roundtrip(mixed, final_q)
        dct, qtable, available = jpeg_info_from_pil(mixed, final_q)
        return {
            "image": pil_to_tensor(mixed).float(),
            "mask": (pil_to_tensor(mask) > 0.5).float(),
            "dct": dct.float(),
            "qtable": qtable.float(),
            "dct_available": available.float(),
            "host_q": host_q,
            "patch_q": patch_q,
            "final_q": final_q,
            "area": float(area),
        }


def stack_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([s["image"] for s in samples]).cuda(non_blocking=True),
        "mask": torch.stack([s["mask"] for s in samples]).cuda(non_blocking=True),
        "dct": torch.stack([s["dct"] for s in samples]).cuda(non_blocking=True),
        "qtable": torch.stack([s["qtable"] for s in samples]).cuda(non_blocking=True),
        "dct_available": torch.stack([s["dct_available"] for s in samples]).cuda(non_blocking=True),
        "meta": [{k: v for k, v in s.items() if k not in {"image", "mask", "dct", "qtable", "dct_available"}} for s in samples],
    }


class CompressionPretrainHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, forensic_n1: torch.Tensor, dct_n1: torch.Tensor | None = None) -> torch.Tensor:
        if dct_n1 is None:
            x = torch.cat([forensic_n1, torch.zeros_like(forensic_n1)], dim=1)
        else:
            if dct_n1.shape[-2:] != forensic_n1.shape[-2:]:
                dct_n1 = F.interpolate(dct_n1, size=forensic_n1.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([forensic_n1, dct_n1], dim=1)
        return self.net(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.sigmoid()
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(1)
    den = prob.sum(1) + target.sum(1)
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def set_pretrainable(model: torch.nn.Module, head: torch.nn.Module) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for name, p in model.named_parameters():
        trainable = name.startswith("forensic_stream.") or name.startswith("dct_branch.output_projections.")
        p.requires_grad_(trainable)
        if trainable:
            names.append(name)
    for p in head.parameters():
        p.requires_grad_(True)
    model.eval()
    model.forensic_stream.train()
    if model.dct_branch is not None:
        model.dct_branch.eval()
        model.dct_branch.output_projections.train()
    head.train()
    return names


def pretrain_forward(model, head, batch):
    forensic = model.forensic_stream(batch["image"])
    dct_features = model.dct_branch(batch["dct"], batch["qtable"]) if model.dct_branch is not None else None
    logits_low = head(forensic[0], dct_features[0] if dct_features is not None else None)
    target_low = F.interpolate(batch["mask"], size=logits_low.shape[-2:], mode="nearest")
    return logits_low, target_low, forensic


def pretrain(model, cfg, outdir: Path, summary: dict, epochs: int, steps_per_epoch: int, batch_size: int, lr: float, align_weight: float) -> dict:
    records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "train")
    gen = CompressionDiscrepancyGenerator(records, image_size=int(cfg["data"].get("image_size", 640)))
    head = CompressionPretrainHead().cuda()
    names = set_pretrainable(model, head)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n.startswith("forensic_stream.") and p.requires_grad], "lr": lr, "weight_decay": 0.01, "name": "forensic_stream."},
        {"params": [p for n, p in model.named_parameters() if n.startswith("dct_branch.output_projections.") and p.requires_grad], "lr": lr * 0.5, "weight_decay": 0.01, "name": "dct_branch.output_projections."},
        {"params": list(head.parameters()), "lr": lr, "weight_decay": 0.01, "name": "compression_pretrain_head."},
    ])
    source_model = build_final_model(cfg["model"]).cuda()
    source_model.load_state_dict(summary["source_state_for_alignment"], strict=True)
    source_model.eval()
    del summary["source_state_for_alignment"]
    history = []
    summary["pretrain"] = {"epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_size": batch_size, "lr": lr, "align_weight": align_weight, "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_examples": names[:80], "history": history}
    for epoch in range(1, epochs + 1):
        started = time.time()
        totals: dict[str, float] = {}
        for step in range(1, steps_per_epoch + 1):
            batch = stack_samples([gen.make() for _ in range(batch_size)])
            logits_low, target_low, forensic = pretrain_forward(model, head, batch)
            bce = F.binary_cross_entropy_with_logits(logits_low, target_low)
            dsc = dice_loss(logits_low, target_low)
            with torch.no_grad():
                source_features = source_model.forensic_stream(batch["image"])
            align = sum(F.mse_loss(F.normalize(a.float(), dim=1), F.normalize(b.float(), dim=1)) for a, b in zip(forensic, source_features)) / len(forensic)
            loss = bce + dsc + float(align_weight) * align
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0, error_if_nonfinite=True)
            opt.step()
            vals = {"loss": loss, "bce": bce, "dice": dsc, "align": align}
            for k, v in vals.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            if step == 1 or step % 50 == 0 or step == steps_per_epoch:
                progress = {
                    "stage": "compression_pretrain",
                    "epoch": epoch,
                    "epochs": epochs,
                    "step": step,
                    "steps": steps_per_epoch,
                    "loss": totals["loss"] / step,
                    "bce": totals["bce"] / step,
                    "dice": totals["dice"] / step,
                    "align": totals["align"] / step,
                    "elapsed_minutes": (time.time() - started) / 60,
                    "eta_minutes": (time.time() - started) / step * (steps_per_epoch - step) / 60,
                }
                dump(outdir / "progress.yaml", progress)
                print("[pretrain]", json.dumps(progress), flush=True)
        row = {"epoch": epoch, "losses": {k: v / steps_per_epoch for k, v in totals.items()}, "elapsed_seconds": time.time() - started}
        history.append(row)
        checkpoint = {"epoch": epoch, "model": e21.copy_state(model), "head": head.state_dict(), "config": cfg, "phase": "compression_discrepancy_pretrain", "pretrain_losses": row["losses"]}
        atomic_torch_save(checkpoint, outdir / "pretrained_forensic.pt")
        dump(outdir / "summary.yaml", summary)
    return {"status": "completed", "last_losses": history[-1]["losses"], "checkpoint": str(outdir / "pretrained_forensic.pt")}


def real_forward(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        batch["image"].cuda(non_blocking=True),
        dct=batch["dct"].cuda(non_blocking=True),
        qtable=batch["qtable"].cuda(non_blocking=True),
        dct_available=batch["dct_available"].cuda(non_blocking=True),
    )


def objective(out: dict, batch: dict, seg: ReTraceLoss, cfg: dict) -> dict[str, torch.Tensor]:
    target = batch["mask"].cuda(non_blocking=True)
    label = batch["label"].cuda(non_blocking=True)
    terms = seg(out, target, label)
    aux = dher_v2_aux_loss(out, target, cfg)
    deep = weighted_deep_supervision_loss(out, target, cfg)
    tv = tversky_loss(out["mask_logits"], target, float(cfg["loss"].get("tversky_alpha_fp", 0.3)), float(cfg["loss"].get("tversky_beta_fn", 0.7)))
    terms["total"] = terms["total"] + aux["dher_aux_total"] + deep + float(cfg["loss"]["weights"].get("tversky", 0.0)) * tv
    terms.update(weighted_deep_supervision=deep, tversky=tv)
    return terms


def set_finetuneable(model: torch.nn.Module) -> list[str]:
    prefixes = (
        "forensic_stream.",
        "dct_branch.output_projections.",
        "dct_inject_adapters.",
        "qtr.",
        "hcv2.",
        "dher.",
        "decoder.",
        "mask_head.",
        "log_var_head.",
        "aux_heads.",
        "dher_visibility.",
        "dher_calibrated_writeback.",
    )
    names = []
    for name, p in model.named_parameters():
        trainable = name.startswith(prefixes)
        p.requires_grad_(trainable)
        if trainable:
            names.append(name)
    model.eval()
    for module_name in ("forensic_stream", "dct_inject_adapters", "qtr", "hcv2", "dher", "decoder", "mask_head", "log_var_head", "aux_heads", "dher_visibility", "dher_calibrated_writeback"):
        module = getattr(model, module_name, None)
        if module is not None and any(n.startswith(module_name + ".") for n in names):
            module.train()
    if model.dct_branch is not None:
        model.dct_branch.eval()
        model.dct_branch.output_projections.train()
    return names


def optimizer_for_finetune(model: torch.nn.Module) -> torch.optim.Optimizer:
    specs = [
        ("forensic_stream.", 1e-5, 0.01),
        ("dct_branch.output_projections.", 5e-6, 0.01),
        ("dct_inject_adapters.", 5e-6, 0.01),
        ("qtr.", 5e-6, 0.01),
        ("hcv2.", 5e-6, 0.01),
        ("dher.", 5e-6, 0.01),
        ("decoder.", 5e-6, 0.01),
        ("mask_head.", 5e-6, 0.0),
        ("log_var_head.", 5e-6, 0.0),
        ("aux_heads.", 5e-6, 0.0),
        ("dher_visibility.", 2e-5, 0.01),
        ("dher_calibrated_writeback.", 2e-5, 0.01),
    ]
    groups = []
    for prefix, lr, wd in specs:
        params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": wd, "name": prefix})
    return torch.optim.AdamW(groups)


def compact_metrics(result: dict) -> dict:
    m = result["metrics"]
    out = {
        "fake_oracle_f1": m["fake"]["oracle"]["f1"],
        "fake_oracle_threshold": m["fake"]["oracle"].get("threshold"),
        "fake_fixed_f1": m["fake"]["fixed"]["f1"],
        "fake_fixed_precision": m["fake"]["fixed"]["precision"],
        "fake_fixed_recall": m["fake"]["fixed"]["recall"],
        "all_oracle_f1": m["all"]["oracle"]["f1"],
    }
    for key in ["Tp_S:lt_1", "Tp_S:1_5", "Tp_S:5_20", "Tp_S:gte_20", "Tp_D:gte_20"]:
        if key in m:
            out[key] = {"oracle_f1": m[key]["oracle"]["f1"], "fixed_f1": m[key]["fixed"]["f1"], "fixed_precision": m[key]["fixed"].get("precision"), "fixed_recall": m[key]["fixed"].get("recall")}
    return out


def finetune(model, cfg, outdir: Path, summary: dict, epochs: int) -> dict:
    names = set_finetuneable(model)
    opt = optimizer_for_finetune(model)
    seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
    loader = train_loader_for(cfg)
    val_records = e21.read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 4))
    best = {"value": -math.inf, "epoch": None}
    summary["finetune"] = {"epochs": epochs, "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_examples": names[:100], "optimizer_groups": [{"name": g.get("name"), "lr": g.get("lr"), "weight_decay": g.get("weight_decay"), "params": sum(p.numel() for p in g["params"])} for g in opt.param_groups], "history": []}
    for epoch in range(1, epochs + 1):
        started = time.time()
        totals: dict[str, float] = {}
        opt.zero_grad(set_to_none=True)
        summary.update(status="finetuning", current_epoch=epoch)
        dump(outdir / "summary.yaml", summary)
        for step, batch in enumerate(loader, 1):
            out = real_forward(model, batch)
            terms = objective(out, batch, seg, cfg)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
            (terms["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["training"].get("grad_clip", 0.5)), error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            if step == 1 or step % 50 == 0 or step == len(loader):
                elapsed = time.time() - started
                progress = {"stage": "finetune", "epoch": epoch, "epochs": epochs, "step": step, "steps": len(loader), "loss": totals["total"] / step, "elapsed_minutes": elapsed / 60, "eta_minutes": elapsed / step * (len(loader) - step) / 60}
                dump(outdir / "progress.yaml", progress)
                print("[finetune]", json.dumps(progress), flush=True)
        result = e21.evaluate(model, val_records, cfg, f"epoch_{epoch:02d}_original_val", outdir)
        score = result["metrics"]["fake"]["oracle"]["f1"]
        row = {"epoch": epoch, "selection_value": score, "delta_vs_source": score - SOURCE_BEST_F1, "train_losses": {k: v / len(loader) for k, v in totals.items()}, "validation_compact": compact_metrics(result), "validation": result, "elapsed_seconds": time.time() - started}
        if score > best["value"]:
            best = {"value": score, "epoch": epoch}
            atomic_torch_save({"epoch": epoch, "model": e21.copy_state(model), "optimizer": opt.state_dict(), "config": cfg, "eval_variant": "raw", "phase": "E25_finetune", "best_metric": score, "selection_threshold": result["metrics"]["fake"]["oracle"].get("threshold")}, outdir / "best.pt")
            row["is_best"] = True
            summary.update(best_F1=score, best_epoch=epoch, best_checkpoint=str(outdir / "best.pt"))
        else:
            row["is_best"] = False
        summary["finetune"]["history"].append(row)
        dump(outdir / "summary.yaml", summary)
        print("[epoch_done]", json.dumps({"epoch": epoch, "fake_F1": score, "delta_vs_source": score - SOURCE_BEST_F1, "best_F1": best["value"], "TpS_gte20": row["validation_compact"].get("Tp_S:gte_20")}, ensure_ascii=False), flush=True)
        if epoch == 1 and score < SOURCE_BEST_F1 - 0.006:
            summary.update(status="stopped", stop_reason="epoch1_drop_exceeded_0p006_after_compression_pretrain")
            dump(outdir / "summary.yaml", summary)
            print("[early_stop] epoch1_drop_exceeded_0p006_after_compression_pretrain", flush=True)
            return best
    return best


def main(args) -> None:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.yaml"
    if summary_path.exists():
        raise RuntimeError(f"refuse duplicate run: {summary_path}")
    summary = {"status": "initializing", "experiment": EXPERIMENT, "source": "E24 best", "test_run": False, "selection": "val_original_grid_fake_pooled_F1_best", "stop_reason": None}
    dump(summary_path, summary)
    try:
        set_seed(args.seed)
        source = torch.load(args.checkpoint, map_location="cpu")
        cfg = copy.deepcopy(source["config"])
        cfg["experiment"] = EXPERIMENT
        cfg["training"].update(ema=False, amp=False, batch_size=1, gradient_accumulation_steps=4, grad_clip=0.5, selection_metric="val_original_grid_fake_pooled_F1_best")
        dump(outdir / "config.yaml", cfg)
        model = build_final_model(cfg["model"]).cuda()
        model.load_state_dict(source["model"], strict=True)
        summary.update(source_checkpoint=args.checkpoint, source_best_F1=float(source.get("best_metric", SOURCE_BEST_F1)), source_state_for_alignment=e21.copy_state(model), protocol="original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA")
        dump(summary_path, {k: v for k, v in summary.items() if k != "source_state_for_alignment"})
        pre = pretrain(model, cfg, outdir, summary, args.pretrain_epochs, args.pretrain_steps, args.pretrain_batch_size, args.pretrain_lr, args.align_weight)
        summary.update(status="pretrain_completed", pretrain_result=pre)
        dump(summary_path, summary)
        best = finetune(model, cfg, outdir, summary, args.finetune_epochs)
        if (outdir / "best.pt").exists():
            val_records = e21.read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
            best_state = torch.load(outdir / "best.pt", map_location="cpu")
            model.load_state_dict(best_state["model"], strict=True)
            full = e21.evaluate(model, val_records, cfg, "best_original_val", outdir)
            summary.update(final_full_metrics=full, final_compact=compact_metrics(full), acceptance_pass=bool(full["metrics"]["fake"]["oracle"]["f1"] >= 0.800))
        if summary.get("status") != "stopped":
            summary.update(status="completed", stop_reason=f"fixed_pretrain_{args.pretrain_epochs}_finetune_{args.finetune_epochs}_budget_completed")
        summary.update(best_F1=best["value"], best_epoch=best["epoch"])
        dump(summary_path, summary)
        print("[E25_COMPLETE]", json.dumps({"status": summary["status"], "best_F1": summary.get("best_F1"), "delta_vs_source": (summary.get("best_F1") or 0.0) - SOURCE_BEST_F1, "acceptance_pass": summary.get("acceptance_pass"), "stop_reason": summary.get("stop_reason")}, ensure_ascii=False), flush=True)
    except BaseException:
        summary.pop("source_state_for_alignment", None)
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-steps", type=int, default=2500)
    parser.add_argument("--pretrain-batch-size", type=int, default=2)
    parser.add_argument("--pretrain-lr", type=float, default=1e-5)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    main(parser.parse_args())
