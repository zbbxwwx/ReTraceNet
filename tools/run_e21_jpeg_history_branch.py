"""E21: parallel original-JPEG compression history branch injected before QTR."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import ManipulationDataset, pil_to_tensor, read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import (
    clean_loss_config,
    dher_v2_aux_loss,
    train_loader_for,
    tversky_loss,
    weighted_deep_supervision_loss,
)
from e15_original_metrics import ValidationStats, stratum


PHASE_A = "A_evidence_space_adapt"
PHASE_B = "B_mechanism_decoder_adapt"
E5_F1 = 0.7954329136683185


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model: torch.nn.Module, frozen_only: bool = False) -> str:
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    h = hashlib.sha256()
    for n, t in model.state_dict().items():
        if frozen_only and n in trained:
            continue
        h.update(n.encode())
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_e5_with_dct(model: torch.nn.Module, source_state: dict[str, torch.Tensor]) -> dict:
    target = model.state_dict()
    loaded = {}
    skipped_shape = []
    for key, value in source_state.items():
        if key in target and tuple(value.shape) == tuple(target[key].shape):
            loaded[key] = value
        elif key in target:
            skipped_shape.append(key)
    result = model.load_state_dict(loaded, strict=False)
    missing = list(result.missing_keys)
    unexpected = [k for k in source_state if k not in target]
    bad_missing = [
        k for k in missing
        if not (k.startswith("dct_branch.") or k.startswith("dct_inject_adapters."))
    ]
    if bad_missing or unexpected or skipped_shape:
        raise RuntimeError(
            f"E21 init mismatch bad_missing={bad_missing[:20]} unexpected={unexpected[:20]} skipped_shape={skipped_shape[:20]}"
        )
    return {
        "loaded": len(loaded),
        "missing": missing,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "skipped_shape": skipped_shape,
        "missing_prefix_counts": prefix_counts(missing),
    }


def prefix_counts(keys: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def make_e21_config(e5_cfg: dict, dct_pretrained: str) -> dict:
    cfg = json.loads(json.dumps(e5_cfg))
    cfg["experiment"] = "e21_jpeg_history_branch"
    cfg["data"].update(
        dct_forensic=True,
        dct_source_mode="original_jpeg",
        dct_qtable_from_source=False,
        dct_jpeg_quality=100,
    )
    cfg["model"].update(
        forensic_stream_type="residual",
        enable_dct_parallel=True,
        dct_injection_zero_init=True,
        dct_pretrained_path=dct_pretrained,
        freeze_forensic_bn=True,
    )
    cfg["training"].update(
        ema=False,
        amp=False,
        batch_size=1,
        gradient_accumulation_steps=4,
        grad_clip=0.5,
        selection_metric="val_original_grid_fake_pooled_F1_best",
    )
    return cfg


def forward(model: torch.nn.Module, batch: dict) -> dict:
    kwargs = {
        "dct": batch["dct"].cuda(non_blocking=True),
        "qtable": batch["qtable"].cuda(non_blocking=True),
        "dct_available": batch["dct_available"].cuda(non_blocking=True),
    }
    return model(batch["image"].cuda(non_blocking=True), **kwargs)


def objective(out: dict, batch: dict, seg: ReTraceLoss, cfg: dict) -> dict[str, torch.Tensor]:
    y, lab = batch["mask"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True)
    terms = seg(out, y, lab)
    aux = dher_v2_aux_loss(out, y, cfg)
    terms["total"] = terms["total"] + aux["dher_aux_total"]
    ds = weighted_deep_supervision_loss(out, y, cfg)
    tv = tversky_loss(
        out["mask_logits"],
        y,
        float(cfg["loss"].get("tversky_alpha_fp", 0.3)),
        float(cfg["loss"].get("tversky_beta_fn", 0.7)),
    )
    terms.update(weighted_deep_supervision=ds, tversky=tv)
    terms["total"] = terms["total"] + ds + float(cfg["loss"]["weights"].get("tversky", 0.0)) * tv
    return terms


def set_phase_trainable(model: torch.nn.Module, phase: str) -> list[str]:
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    if phase == PHASE_A:
        prefixes = ("dct_branch.output_projections.", "dct_inject_adapters.")
    elif phase == PHASE_B:
        prefixes = (
            "dct_branch.output_projections.",
            "dct_inject_adapters.",
            "qtr.",
            "hcv2.",
            "dher.",
            "decoder.",
            "mask_head.",
            "log_var_head.",
            "aux_heads.",
        )
    else:
        raise ValueError(phase)
    names = []
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            p.requires_grad_(True)
            names.append(name)
    if model.dct_branch is not None:
        model.dct_branch.eval()
    model.eval()
    for module_name in ("dct_inject_adapters", "qtr", "hcv2", "dher", "decoder", "mask_head", "log_var_head", "aux_heads"):
        module = getattr(model, module_name, None)
        if module is not None and any(name.startswith(module_name + ".") for name in names):
            module.train()
    if model.dct_branch is not None:
        model.dct_branch.output_projections.train()
    assert not any(p.requires_grad for p in model.content_encoder.parameters())
    assert not any(p.requires_grad for p in model.forensic_stream.parameters())
    frozen_dct_base = [
        p.requires_grad for n, p in model.named_parameters()
        if n.startswith("dct_branch.") and not n.startswith("dct_branch.output_projections.")
    ]
    assert not any(frozen_dct_base)
    return names


def optimizer_for(model: torch.nn.Module, phase: str) -> torch.optim.Optimizer:
    if phase == PHASE_A:
        specs = [
            ("dct_branch.output_projections.", 2e-5),
            ("dct_inject_adapters.", 2e-5),
        ]
    else:
        specs = [
            ("dct_branch.output_projections.", 3e-6),
            ("dct_inject_adapters.", 5e-6),
            ("qtr.", 2e-6),
            ("hcv2.", 2e-6),
            ("dher.", 2e-6),
            ("decoder.", 5e-6),
            ("mask_head.", 5e-6),
            ("log_var_head.", 5e-6),
            ("aux_heads.", 5e-6),
        ]
    groups = []
    for prefix, lr in specs:
        params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01, "name": prefix})
    if not groups:
        raise RuntimeError(f"no trainable groups for {phase}")
    return torch.optim.AdamW(groups)


def audit_step(model, opt, batch, seg, cfg, phase, source_state) -> dict:
    before = state_hash(model)
    frozen_before = state_hash(model, frozen_only=True)
    trainable_before = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    out = forward(model, batch)
    diag = out.get("dct_evidence", {})
    terms = objective(out, batch, seg, cfg)
    assert torch.isfinite(terms["total"])
    terms["total"].backward()
    groups = {}
    for group in opt.param_groups:
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(group["name"]) and p.requires_grad]
        grads = [p.grad for _, p in named if p.grad is not None]
        norm = sum(float(g.detach().square().sum()) for g in grads) ** 0.5
        zero_init_blocks_first_step = group["name"] == "dct_branch.output_projections."
        if zero_init_blocks_first_step:
            assert math.isfinite(norm), (phase, group["name"], norm)
        else:
            assert math.isfinite(norm) and norm > 0, (phase, group["name"], norm)
        groups[group["name"]] = {
            "parameters": sum(p.numel() for _, p in named),
            "with_grad": len(grads),
            "grad_norm": norm,
            "lr": group["lr"],
            "zero_init_adapter_blocks_first_step_grad": zero_init_blocks_first_step and norm == 0,
        }
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
    opt.step()
    for prefix, row in groups.items():
        delta = sum(
            float((p.detach().cpu() - trainable_before[n]).abs().sum())
            for n, p in model.named_parameters()
            if n.startswith(prefix) and n in trainable_before
        )
        if prefix == "dct_branch.output_projections.":
            assert math.isfinite(delta), (prefix, delta)
        else:
            assert delta > 0, (prefix, delta)
        row["parameter_delta_abs_sum"] = delta
    assert state_hash(model, frozen_only=True) == frozen_before
    model.load_state_dict(source_state, strict=False)
    opt.state.clear()
    opt.zero_grad(set_to_none=True)
    assert state_hash(model) == before
    return {
        "status": "PASS",
        "phase": phase,
        "loss": float(terms["total"].detach()),
        "dct_available_mean": float(diag.get("available_mean", torch.tensor(-1.0)).detach().cpu()),
        "dct_write_rms": [float(x.detach().cpu()) for x in diag.get("write_rms", [])],
        "groups": groups,
        "frozen_state_unchanged": True,
        "audit_step_restored": True,
    }


@torch.inference_mode()
def evaluate(model, records, cfg, label, directory, ablate_dct: bool = False) -> dict:
    dataset = ManipulationDataset(
        records,
        image_size=int(cfg["data"].get("image_size", 640)),
        training=False,
        dct_forensic=True,
        dct_source_mode="original_jpeg",
    )
    model.eval()
    model.disable_dct_evidence = bool(ablate_dct)
    stats = ValidationStats()
    started = time.time()
    available_sum = 0.0
    per_image = directory / (label + ("_dct_ablate" if ablate_dct else "") + "_per_image.jsonl")
    with per_image.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            sample = dataset[index]
            with Image.open(record.image) as im:
                size = im.size
            if record.mask is None:
                target = np.zeros((size[1], size[0]), dtype=np.int16)
            else:
                with Image.open(record.mask) as mask:
                    target = (np.asarray(mask.convert("L")) > 127).astype(np.int16)
            batch = {
                "image": sample["image"][None],
                "dct": sample["dct"][None],
                "qtable": sample["qtable"][None],
                "dct_available": sample["dct_available"][None],
            }
            out = forward(model, batch)
            prob = F.interpolate(
                out["mask_logits"].float().sigmoid(),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
            stats.add(prob, target, record.label, record.image)
            available = float(sample["dct_available"].mean())
            available_sum += available
            handle.write(json.dumps({
                "path": str(record.image),
                "label": record.label,
                "bucket": stratum(record.image, target),
                "dct_available": available,
            }) + "\n")
            del out
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(records):
                print(f"[val:{label}{':ablate' if ablate_dct else ''}] {index+1}/{len(records)} elapsed={(time.time()-started)/60:.1f}m", flush=True)
    model.disable_dct_evidence = False
    result = {
        "protocol": "original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA",
        "split": "val",
        "ablate_dct": bool(ablate_dct),
        "dct_available_image_fraction": available_sum / max(1, len(records)),
        "metrics": stats.result(),
        "elapsed_seconds": time.time() - started,
    }
    dump(directory / (label + ("_dct_ablate" if ablate_dct else "") + ".yaml"), result)
    return result


def train_phase(model, cfg, source_state, phase, epochs, outdir, val_records, summary, summary_path, best):
    names = set_phase_trainable(model, phase)
    opt = optimizer_for(model, phase)
    seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
    frozen_hash = state_hash(model, frozen_only=True)
    loader = train_loader_for(cfg)
    steps = len(loader)
    phase_dir = outdir / phase
    phase_dir.mkdir(exist_ok=True)
    summary["phases"][phase] = {
        "epochs": epochs,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_examples": names[:40],
        "optimizer": [{"name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in opt.param_groups],
        "history": [],
    }
    for epoch in range(1, epochs + 1):
        model.eval()
        for module_name in ("dct_inject_adapters", "qtr", "hcv2", "dher", "decoder", "mask_head", "log_var_head", "aux_heads"):
            module = getattr(model, module_name, None)
            if module is not None and any(n.startswith(module_name + ".") for n in names):
                module.train()
        if model.dct_branch is not None:
            model.dct_branch.eval()
            model.dct_branch.output_projections.train()
        opt.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        dct_seen = 0.0
        started = time.time()
        summary.update(status="training", current_phase=phase, current_epoch=epoch)
        dump(summary_path, summary)
        print(f"[train] {phase} epoch={epoch}/{epochs} step=0/{steps}", flush=True)
        for bi, batch in enumerate(loader):
            out = forward(model, batch)
            terms = objective(out, batch, seg, cfg)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"nonfinite total loss at {phase} epoch={epoch} batch={bi}")
            accumulation = min(4, steps - (bi // 4) * 4)
            (terms["total"] / accumulation).backward()
            if (bi + 1) % 4 == 0 or bi + 1 == steps:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            dct_seen += float(batch["dct_available"].float().mean())
            if bi == 0 or (bi + 1) % 50 == 0 or bi + 1 == steps:
                elapsed = time.time() - started
                progress = {
                    "phase": phase,
                    "epoch": epoch,
                    "step": bi + 1,
                    "steps": steps,
                    "loss": totals["total"] / (bi + 1),
                    "dct_available_batch_mean": dct_seen / (bi + 1),
                    "elapsed_minutes": elapsed / 60,
                    "eta_minutes": elapsed / (bi + 1) * (steps - bi - 1) / 60,
                }
                dump(outdir / "progress.yaml", progress)
                print("[train]", json.dumps(progress), flush=True)
            del out, terms
        assert state_hash(model, frozen_only=True) == frozen_hash
        result = evaluate(model, val_records, cfg, f"{phase}_epoch_{epoch:02d}_original_val", phase_dir)
        current = result["metrics"]["fake"]["oracle"]["f1"]
        row = {
            "epoch": epoch,
            "selection_value": current,
            "delta_vs_E5": current - E5_F1,
            "train_losses": {k: v / steps for k, v in totals.items()},
            "dct_available_train_batch_mean": dct_seen / steps,
            "validation": result,
            "elapsed_seconds": time.time() - started,
        }
        is_best = current > best["value"]
        if is_best:
            best.update(value=current, phase=phase, epoch=epoch)
            state = {
                "epoch": epoch,
                "model": copy_state(model),
                "optimizer": opt.state_dict(),
                "config": cfg,
                "eval_variant": "raw",
                "phase": phase,
                "best_metric": current,
                "selection_threshold": result["metrics"]["fake"]["oracle"]["threshold"],
            }
            atomic_torch_save(state, outdir / "best.pt")
            atomic_torch_save(state, phase_dir / "best.pt")
            summary.update(best_F1=current, best_phase=phase, best_epoch=epoch, best_checkpoint=str(outdir / "best.pt"))
        atomic_torch_save({
            "epoch": epoch,
            "model": copy_state(model),
            "optimizer": opt.state_dict(),
            "config": cfg,
            "eval_variant": "raw",
            "phase": phase,
            "best_metric": best["value"],
            "selection_threshold": result["metrics"]["fake"]["oracle"]["threshold"],
        }, outdir / "last.pt")
        row["is_best"] = is_best
        summary["phases"][phase]["history"].append(row)
        dump(summary_path, summary)
        print("[epoch_done]", phase, epoch, "F1", current, "delta_E5", current - E5_F1, flush=True)
    return best


def find_preflight_batch(cfg: dict) -> tuple[dict, dict]:
    loader = train_loader_for(cfg)
    scanned = 0
    available = 0
    positives = 0
    for batch in loader:
        scanned += 1
        is_available = float(batch["dct_available"].float().mean()) > 0
        is_positive = float(batch["mask"].float().sum()) > 0
        available += int(is_available)
        positives += int(is_positive)
        if is_available and is_positive:
            return batch, {"scanned": scanned, "available": available, "positive": positives}
        if scanned >= 512:
            break
    raise RuntimeError(f"could not find a positive JPEG-available preflight batch in {scanned} samples; available={available} positive={positives}")


def main(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.yaml"
    if summary_path.exists():
        raise RuntimeError(f"refuse duplicate run: {summary_path}")
    summary = {
        "status": "initializing",
        "experiment": "E21_original_jpeg_history_parallel_branch",
        "single_change": "add parallel original-JPEG DCT/qtable evidence before QTR through zero-init injection; keep E5 residual forensic stream",
        "test_run": False,
        "phases": {},
        "stop_reason": None,
    }
    dump(summary_path, summary)
    try:
        set_seed(3407)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        source_state = checkpoint["model"]
        e5_cfg = checkpoint["config"]
        assert checkpoint.get("eval_variant", "raw") == "raw"
        cfg = make_e21_config(e5_cfg, args.dct_pretrained)
        dump(outdir / "config.yaml", cfg)
        val_records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")

        baseline = build_final_model(e5_cfg["model"]).cuda()
        baseline.load_state_dict(source_state, strict=True)
        model = build_final_model(cfg["model"]).cuda()
        init_report = load_e5_with_dct(model, source_state)

        train_probe, probe_search = find_preflight_batch(cfg)
        baseline.eval()
        model.eval()
        with torch.no_grad():
            e5_out = baseline(train_probe["image"].cuda())
            e21_out = forward(model, train_probe)
            identity = float((e21_out["mask_logits"] - e5_out["mask_logits"]).abs().max())
            dct_diag = e21_out["dct_evidence"]
        if identity >= 1e-6:
            raise AssertionError(f"zero-init DCT identity failed: {identity}")

        source_with_dct = copy_state(model)
        seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
        audits = {}
        for phase in (PHASE_A, PHASE_B):
            names = set_phase_trainable(model, phase)
            opt = optimizer_for(model, phase)
            audits[phase] = audit_step(model, opt, train_probe, seg, cfg, phase, source_with_dct)
            audits[phase]["trainable_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            audits[phase]["trainable_examples"] = names[:30]
            model.load_state_dict(source_with_dct, strict=True)
            del opt

        summary.update(
            source_checkpoint=args.checkpoint,
            source_checkpoint_sha256=sha_file(args.checkpoint),
            source_checkpoint_epoch=checkpoint.get("epoch"),
            dct_pretrained_path=args.dct_pretrained,
            init_report=init_report,
            protocol="original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA",
            zero_init_identity_max_abs_diff=identity,
            initial_dct_available_mean=float(dct_diag["available_mean"].detach().cpu()),
            preflight_batch_search=probe_search,
            initial_dct_write_rms=[float(x.detach().cpu()) for x in dct_diag["write_rms"]],
            preflight=audits,
            baseline_E5_fake_F1=E5_F1,
        )
        dump(outdir / "preflight.yaml", summary["preflight"])
        dump(summary_path, summary)
        print("[preflight]", json.dumps({
            "identity_max_abs_diff": identity,
            "init_missing_prefix_counts": init_report["missing_prefix_counts"],
            "initial_dct_write_rms": summary["initial_dct_write_rms"],
            "audits": {k: v["status"] for k, v in audits.items()},
        }), flush=True)

        del baseline, e5_out, e21_out
        gc.collect()
        torch.cuda.empty_cache()

        best = {"value": -math.inf, "phase": None, "epoch": None}
        best = train_phase(model, cfg, source_with_dct, PHASE_A, args.phase_a_epochs, outdir, val_records, summary, summary_path, best)
        best = train_phase(model, cfg, copy_state(model), PHASE_B, args.phase_b_epochs, outdir, val_records, summary, summary_path, best)

        best_state = torch.load(outdir / "best.pt", map_location="cpu")
        model.load_state_dict(best_state["model"], strict=True)
        ablation = evaluate(model, val_records, cfg, "best_original_val", outdir, ablate_dct=True)
        full = evaluate(model, val_records, cfg, "best_original_val", outdir, ablate_dct=False)
        summary.update(
            status="completed",
            stop_reason="fixed_E21_two_phase_completed",
            best_F1=full["metrics"]["fake"]["oracle"]["f1"],
            best_delta_vs_E5=full["metrics"]["fake"]["oracle"]["f1"] - E5_F1,
            dct_ablation_fake_F1=ablation["metrics"]["fake"]["oracle"]["f1"],
            dct_contribution_fake_F1=full["metrics"]["fake"]["oracle"]["f1"] - ablation["metrics"]["fake"]["oracle"]["f1"],
            dct_contribution_TpS_lt1=full["metrics"].get("Tp_S:lt_1", {}).get("oracle", {}).get("f1", 0.0) - ablation["metrics"].get("Tp_S:lt_1", {}).get("oracle", {}).get("f1", 0.0),
            dct_contribution_TpS_gte20=full["metrics"].get("Tp_S:gte_20", {}).get("oracle", {}).get("f1", 0.0) - ablation["metrics"].get("Tp_S:gte_20", {}).get("oracle", {}).get("f1", 0.0),
            final_full_metrics=full,
            final_dct_ablation_metrics=ablation,
        )
        dump(summary_path, summary)
        print("[E21_COMPLETE]", json.dumps({
            "fake_F1": summary["best_F1"],
            "delta_vs_E5": summary["best_delta_vs_E5"],
            "dct_contribution_fake_F1": summary["dct_contribution_fake_F1"],
            "TpS_lt1_contribution": summary["dct_contribution_TpS_lt1"],
            "TpS_gte20_contribution": summary["dct_contribution_TpS_gte20"],
        }), flush=True)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dct-pretrained", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase-a-epochs", type=int, default=2)
    parser.add_argument("--phase-b-epochs", type=int, default=6)
    main(parser.parse_args())
