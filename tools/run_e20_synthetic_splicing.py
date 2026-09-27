"""E20: targeted synthetic splicing pretraining, then CASIA fine-tuning from E5."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrace.data import ManipulationDataset, pil_to_tensor, read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, dher_v2_aux_loss, train_loader_for, tversky_loss, weighted_deep_supervision_loss
from train_e15_hr_corrected import dump, evaluate
from e15_original_metrics import stratum
from run_e17_e5_pair import sha_file, state_hash


TRAINABLE = [
    "forensic_stream.stage3.",
    "forensic_stream.stage4.",
    "forensic_stream.mrfe.",
    "qtr.",
    "hcv2.",
    "dher.",
    "decoder.",
    "mask_head.",
    "log_var_head.",
    "aux_heads.",
]


AREA_BUCKETS = [
    ("lt_1", 0.003, 0.010),
    ("1_5", 0.010, 0.050),
    ("5_20", 0.050, 0.200),
    ("gte_20", 0.200, 0.380),
]


def _resize640(image: Image.Image, mask: Image.Image | None = None):
    image = image.resize((640, 640), Image.Resampling.BILINEAR)
    if mask is None:
        return image
    return image, mask.resize((640, 640), Image.Resampling.NEAREST)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _irregular_mask(width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = width / 2, height / 2
    points = []
    count = random.randint(9, 18)
    base = min(width, height) * random.uniform(0.38, 0.55)
    for i in range(count):
        angle = 2 * math.pi * i / count
        radius = base * random.uniform(0.55, 1.05)
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    draw.polygon(points, fill=255)
    if random.random() < 0.35:
        draw.ellipse(
            (
                random.randint(0, max(0, width // 3)),
                random.randint(0, max(0, height // 3)),
                random.randint(max(1, width // 2), width),
                random.randint(max(1, height // 2), height),
            ),
            fill=255,
        )
    return mask.filter(ImageFilter.GaussianBlur(radius=random.choice([0, 0, 1, 2])))


class SyntheticSplicingDataset(Dataset):
    def __init__(self, records, image_size=640, length=None):
        self.records = records
        self.image_size = int(image_size)
        self.length = int(length or len(records))
        self.authentic = [r for r in records if r.label == 0]
        self.manipulated = [r for r in records if r.label == 1]
        assert len(records) >= 2

    def __len__(self):
        return self.length

    def _record(self, index):
        return self.records[index % len(self.records)]

    def _open_rgb(self, path: Path) -> Image.Image:
        with Image.open(path) as handle:
            return handle.convert("RGB")

    def __getitem__(self, index):
        seed = int(torch.initial_seed() + index) % (2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        bucket_name, lo, hi = AREA_BUCKETS[index % len(AREA_BUCKETS)]
        bg_record = self._record(index)
        donor_record = self._record(index * 7919 + random.randint(1, len(self.records) - 1))
        if donor_record.image == bg_record.image:
            donor_record = self._record(index + 1)
        background = _resize640(self._open_rgb(bg_record.image))
        donor = self._open_rgb(donor_record.image)
        donor_scale = random.uniform(0.75, 1.65)
        donor_short = max(1, min(donor.size))
        donor = donor.resize(
            (
                max(32, round(donor.width * self.image_size * donor_scale / donor_short)),
                max(32, round(donor.height * self.image_size * donor_scale / donor_short)),
            ),
            Image.Resampling.BILINEAR,
        )
        target_area = random.uniform(lo, hi) * self.image_size * self.image_size
        side = int(max(16, min(self.image_size, math.sqrt(target_area / random.uniform(0.45, 0.85)))))
        width = int(max(16, min(self.image_size, side * random.uniform(0.75, 1.45))))
        height = int(max(16, min(self.image_size, target_area / max(1, width) * random.uniform(0.9, 1.3))))
        donor_w, donor_h = donor.size
        if donor_w < width or donor_h < height:
            donor = donor.resize((max(width, donor_w), max(height, donor_h)), Image.Resampling.BILINEAR)
            donor_w, donor_h = donor.size
        left = random.randint(0, donor_w - width)
        top = random.randint(0, donor_h - height)
        patch = donor.crop((left, top, left + width, top + height))
        if random.random() < 0.85:
            patch = _jpeg_roundtrip(patch, random.randint(45, 95))
        if random.random() < 0.7:
            amount = random.uniform(0.06, 0.22)
            patch = ImageEnhance.Brightness(patch).enhance(random.uniform(1 - amount, 1 + amount))
            patch = ImageEnhance.Contrast(patch).enhance(random.uniform(1 - amount, 1 + amount))
            patch = ImageEnhance.Color(patch).enhance(random.uniform(1 - amount, 1 + amount))
        if random.random() < 0.35:
            patch = patch.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
        alpha = _irregular_mask(width, height)
        if random.random() < 0.65:
            alpha = alpha.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 2.4)))
        max_left = self.image_size - width
        max_top = self.image_size - height
        paste_left = random.randint(0, max(0, max_left))
        paste_top = random.randint(0, max(0, max_top))
        forged = background.copy()
        forged.paste(patch, (paste_left, paste_top), alpha)
        mask = Image.new("L", (self.image_size, self.image_size), 0)
        hard_alpha = alpha.point(lambda value: 255 if value > 16 else 0)
        mask.paste(hard_alpha, (paste_left, paste_top))
        if random.random() < 0.65:
            forged = _jpeg_roundtrip(forged, random.randint(55, 98))
        mask_tensor = (pil_to_tensor(mask) > 0.5).float()
        return {
            "image": pil_to_tensor(forged),
            "mask": mask_tensor,
            "label": torch.tensor([1.0], dtype=torch.float32),
            "path": f"synthetic_Tp_S_{bucket_name}_{index:08d}.png",
            "dataset": "synthetic_splicing",
            "synthetic_bucket": bucket_name,
            "post_mask_area_ratio": float(mask_tensor.mean()),
        }


class CasiaBucketDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        mask = sample["mask"].numpy()
        sample["post_geometry_bucket"] = stratum(sample["path"], mask) if mask.any() else "authentic_or_empty"
        return sample


def casia_loader(cfg):
    base = train_loader_for(cfg)
    return DataLoader(
        CasiaBucketDataset(base.dataset),
        batch_size=1,
        sampler=base.sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )


def optimizer_for(model, lr: float):
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in TRAINABLE))
    model.eval()
    groups = []
    for prefix in TRAINABLE:
        params = [param for name, param in model.named_parameters() if name.startswith(prefix) and param.requires_grad]
        if params:
            groups.append({"params": params, "name": prefix, "lr": lr})
    assert not any(param.requires_grad for param in model.content_encoder.parameters())
    return torch.optim.AdamW(groups, weight_decay=0.01)


def objective(out, batch, seg, cfg):
    target = batch["mask"].cuda()
    label = batch["label"].cuda()
    terms = seg(out, target, label)
    aux = dher_v2_aux_loss(out, target, cfg)
    terms["total"] = terms["total"] + aux["dher_aux_total"]
    ds = weighted_deep_supervision_loss(out, target, cfg)
    tv = tversky_loss(
        out["mask_logits"],
        target,
        float(cfg["loss"].get("tversky_alpha_fp", 0.3)),
        float(cfg["loss"].get("tversky_beta_fn", 0.7)),
    )
    terms.update(weighted_deep_supervision=ds, tversky=tv)
    terms["total"] = terms["total"] + ds + float(cfg["loss"]["weights"].get("tversky", 0.0)) * tv
    return terms


def train_epoch(model, opt, loader, seg, cfg, outdir, phase, epoch, total_epochs):
    model.eval()
    opt.zero_grad(set_to_none=True)
    totals = {}
    sampled = {}
    started = time.time()
    count = len(loader)
    print(f"[train] phase={phase} epoch={epoch}/{total_epochs} step=0/{count}", flush=True)
    for bi, batch in enumerate(loader):
        bucket = batch.get("synthetic_bucket") or batch.get("post_geometry_bucket") or ["unknown"]
        bucket = bucket[0] if isinstance(bucket, list) or isinstance(bucket, tuple) else str(bucket)
        sampled[bucket] = sampled.get(bucket, 0) + 1
        terms = objective(model(batch["image"].cuda()), batch, seg, cfg)
        assert torch.isfinite(terms["total"])
        accumulation = min(4, count - (bi // 4) * 4)
        (terms["total"] / accumulation).backward()
        if (bi + 1) % 4 == 0 or bi + 1 == count:
            torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 0.5, error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        if bi == 0 or (bi + 1) % 50 == 0 or bi + 1 == count:
            elapsed = time.time() - started
            progress = {
                "phase": phase,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": bi + 1,
                "steps": count,
                "loss": totals["total"] / (bi + 1),
                "elapsed_minutes": elapsed / 60,
                "eta_minutes": elapsed / (bi + 1) * (count - bi - 1) / 60,
                "sampled": sampled,
            }
            dump(outdir / "progress.yaml", progress)
            print("[train]", json.dumps(progress), flush=True)
    return {key: value / count for key, value in totals.items()}, sampled, time.time() - started


def audit(model, opt, loader, seg, cfg, source_state):
    batch = next(iter(loader))
    before = state_hash(model)
    frozen = state_hash(model, True)
    out = model(batch["image"].cuda())
    terms = objective(out, batch, seg, cfg)
    terms["total"].backward()
    groups = {}
    for group in opt.param_groups:
        named = [(name, param) for name, param in model.named_parameters() if name.startswith(group["name"])]
        grads = [param.grad for _, param in named if param.grad is not None]
        norm = sum(float(grad.detach().square().sum()) for grad in grads) ** 0.5
        assert math.isfinite(norm) and norm > 0, (group["name"], norm)
        groups[group["name"]] = {"parameters": sum(param.numel() for _, param in named), "with_grad": len(grads), "grad_norm": norm, "lr": group["lr"]}
    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 0.5, error_if_nonfinite=True)
    opt.step()
    assert frozen == state_hash(model, True)
    model.load_state_dict(source_state, strict=True)
    opt.state.clear()
    opt.zero_grad(set_to_none=True)
    assert state_hash(model) == before
    return {"status": "PASS", "groups": groups, "frozen_state_unchanged": True, "audit_step_restored": True}


def main(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.yaml"
    assert not summary_path.exists(), "Refuse duplicate run"
    summary = {
        "status": "initializing",
        "experiment": "E20_synthetic_splicing_pretrain",
        "history": [],
        "test_run": False,
        "selection": "val_original_grid_fake_pooled_F1_best",
        "auc_used_for_selection": False,
        "single_change": "targeted synthetic Tp_S pretraining before CASIA fine-tune; architecture unchanged",
        "stop_reason": None,
    }
    dump(summary_path, summary)
    try:
        set_seed(3407)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        cfg = checkpoint["config"]
        cfg["training"]["batch_size"] = 1
        source_state = checkpoint["model"]
        train_records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "train")
        val_records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
        e17 = yaml.safe_load(Path(args.e17_summary).read_text())
        seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
        model = build_final_model(cfg["model"]).cuda()
        model.load_state_dict(source_state, strict=True)
        opt = optimizer_for(model, float(args.lr))
        synth = SyntheticSplicingDataset(train_records, image_size=640, length=len(train_records))
        synth_loader = DataLoader(synth, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
        summary.update(
            source_checkpoint=args.checkpoint,
            source_checkpoint_sha256=sha_file(args.checkpoint),
            source_checkpoint_epoch=checkpoint["epoch"],
            e5_reference=e17["baseline"]["metrics"],
            trainable_prefixes=TRAINABLE,
            optimizer={"type": "AdamW", "lr": float(args.lr), "weight_decay": 0.01, "constant_lr": True},
            synthetic_policy={
                "target": "Tp_S",
                "area_buckets": [name for name, _, _ in AREA_BUCKETS],
                "foreign_patch": True,
                "jpeg_mismatch": True,
                "resize_mismatch": True,
                "color_mismatch": True,
                "boundary_blending": True,
            },
            phase_plan={
                "synthetic_epochs": int(args.synthetic_epochs),
                "casia_finetune_epochs": int(args.finetune_epochs),
                "casia_finetune_geometry": "original_E5_training_pipeline",
                "test": False,
            },
        )
        summary["preflight"] = audit(model, opt, synth_loader, seg, cfg, source_state)
        dump(outdir / "preflight.yaml", summary["preflight"])
        dump(summary_path, summary)
        print("[preflight]", json.dumps(summary["preflight"]), flush=True)
        best = -math.inf
        for epoch in range(1, int(args.synthetic_epochs) + 1):
            summary.update(status="training", phase="synthetic_pretrain", current_epoch=epoch)
            dump(summary_path, summary)
            losses, sampled, elapsed = train_epoch(model, opt, synth_loader, seg, cfg, outdir, "synthetic_pretrain", epoch, int(args.synthetic_epochs))
            summary.update(status="validation")
            dump(summary_path, summary)
            result = evaluate(model, val_records, f"synthetic_epoch_{epoch:02d}_original_val", outdir)
            current = result["metrics"]["fake"]["oracle"]["f1"]
            row = {"phase": "synthetic_pretrain", "epoch": epoch, "selection_value": current, "delta_vs_E5": current - e17["baseline"]["metrics"]["fake"]["oracle"]["f1"], "train_losses": losses, "sampled": sampled, "validation": result, "elapsed_seconds": elapsed}
            summary["history"].append(row)
            if current > best:
                best = current
                atomic_torch_save({"epoch": epoch - 1, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "config": cfg, "eval_variant": "raw", "phase": "E20_synthetic_pretrain", "best_metric": best}, outdir / "best.pt")
                summary.update(best_F1=best, best_phase="synthetic_pretrain", best_epoch=epoch, best_checkpoint=str(outdir / "best.pt"))
            dump(summary_path, summary)
            print("[epoch_done]", "synthetic_pretrain", epoch, "F1", current, flush=True)
        casia = casia_loader(cfg)
        for group in opt.param_groups:
            group["lr"] = float(args.lr) * 0.5
        for epoch in range(1, int(args.finetune_epochs) + 1):
            summary.update(status="training", phase="casia_finetune", current_epoch=epoch)
            dump(summary_path, summary)
            losses, sampled, elapsed = train_epoch(model, opt, casia, seg, cfg, outdir, "casia_finetune", epoch, int(args.finetune_epochs))
            summary.update(status="validation")
            dump(summary_path, summary)
            result = evaluate(model, val_records, f"casia_epoch_{epoch:02d}_original_val", outdir)
            current = result["metrics"]["fake"]["oracle"]["f1"]
            row = {"phase": "casia_finetune", "epoch": epoch, "selection_value": current, "delta_vs_E5": current - e17["baseline"]["metrics"]["fake"]["oracle"]["f1"], "train_losses": losses, "sampled": sampled, "validation": result, "elapsed_seconds": elapsed}
            summary["history"].append(row)
            atomic_torch_save({"epoch": epoch - 1, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "config": cfg, "eval_variant": "raw", "phase": "E20_casia_finetune", "best_metric": max(best, current)}, outdir / "last.pt")
            if current > best:
                best = current
                atomic_torch_save({"epoch": epoch - 1, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "config": cfg, "eval_variant": "raw", "phase": "E20_casia_finetune", "best_metric": best}, outdir / "best.pt")
                summary.update(best_F1=best, best_phase="casia_finetune", best_epoch=epoch, best_checkpoint=str(outdir / "best.pt"))
            dump(summary_path, summary)
            print("[epoch_done]", "casia_finetune", epoch, "F1", current, flush=True)
        summary.update(
            status="completed",
            stop_reason="synthetic_pretrain_then_casia_finetune_completed",
            final_F1=summary["history"][-1]["selection_value"],
            gates={"best_at_least_0p800": best >= 0.800, "auc_not_used": True},
        )
        assert sha_file(args.checkpoint) == summary["source_checkpoint_sha256"]
        dump(summary_path, summary)
        print("[COMPLETE]", best, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--e17-summary", required=True)
    parser.add_argument("--synthetic-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-6)
    main(parser.parse_args())
