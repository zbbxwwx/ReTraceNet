"""Targeted E21 DHER visibility training.

Single-variable follow-up after the DHER visibility audit:
train the zero-init visibility head plus a tiny mask_head LR, with losses that
push missed foreground evidence while discouraging broad positive residuals.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_e21_jpeg_history_branch as e21
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, dher_v2_aux_loss, tversky_loss, train_loader_for, weighted_deep_supervision_loss


EXPERIMENT = "E21_DHER_visibility_targeted"
E21_BEST_F1 = 0.7977327505234914


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def make_config(source_cfg: dict) -> dict:
    cfg = copy.deepcopy(source_cfg)
    cfg["experiment"] = EXPERIMENT
    cfg["model"]["enable_dher_visibility"] = True
    cfg["model"]["enable_splicing_region_affinity"] = False
    cfg["model"]["enable_region_consistency"] = False
    cfg["training"].update(
        ema=False,
        amp=False,
        batch_size=1,
        gradient_accumulation_steps=4,
        grad_clip=0.5,
        selection_metric="val_original_grid_fake_pooled_F1_best",
        visibility_pos_weight=0.5,
        visibility_sparse_weight=0.05,
        visibility_bg_weight=0.05,
        visibility_threshold_source="source_checkpoint_selection_threshold",
    )
    return cfg


def load_source(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict:
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad = [key for key in missing if not key.startswith("dher_visibility.")]
    if bad or unexpected:
        raise RuntimeError(f"source init mismatch bad_missing={bad[:30]} unexpected={unexpected[:30]}")
    return {"missing": missing, "missing_count": len(missing), "unexpected": unexpected}


def forward(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        batch["image"].cuda(non_blocking=True),
        dct=batch["dct"].cuda(non_blocking=True),
        qtable=batch["qtable"].cuda(non_blocking=True),
        dct_available=batch["dct_available"].cuda(non_blocking=True),
    )


def _at(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def evidence_mask(out: dict, ref: torch.Tensor) -> torch.Tensor:
    hcv = out["hcv2"]
    dher = out["dher"]
    terms = []
    for index in range(2):
        proposal = _at(hcv["proposals"][index], ref)
        delta = _at(hcv["deltas"][index], ref)
        support = _at(dher["support"][index], ref)
        conflict = _at(dher["conflict"][index], ref)
        terms.extend([
            proposal.clamp(0, 1),
            delta.abs(),
            F.relu(delta),
            torch.log1p(support.clamp_min(0)),
            conflict.clamp_min(0),
        ])
    raw = torch.stack([t.float() for t in terms], dim=0).mean(0)
    mean = raw.mean((2, 3), keepdim=True)
    std = raw.std((2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
    return torch.sigmoid((raw - mean) / std)


def original_objective(out: dict, batch: dict, seg: ReTraceLoss, cfg: dict) -> dict[str, torch.Tensor]:
    target = batch["mask"].cuda(non_blocking=True)
    label = batch["label"].cuda(non_blocking=True)
    terms = seg(out, target, label)
    aux = dher_v2_aux_loss(out, target, cfg)
    deep = weighted_deep_supervision_loss(out, target, cfg)
    tv = tversky_loss(
        out["mask_logits"],
        target,
        float(cfg["loss"].get("tversky_alpha_fp", 0.3)),
        float(cfg["loss"].get("tversky_beta_fn", 0.7)),
    )
    terms["total"] = terms["total"] + aux["dher_aux_total"] + deep + float(cfg["loss"]["weights"].get("tversky", 0.0)) * tv
    terms.update(weighted_deep_supervision=deep, tversky=tv)
    return terms


def targeted_visibility_losses(out: dict, baseline_out: dict, batch: dict, threshold: float, cfg: dict) -> dict[str, torch.Tensor]:
    target = batch["mask"].cuda(non_blocking=True).float()
    base_prob = baseline_out["mask_logits"].detach().float().sigmoid()
    final_logits = out["mask_logits"]
    missed_fg = ((target > 0.5) & (base_prob < float(threshold))).float()
    if missed_fg.sum() > 0:
        pos_loss = (F.binary_cross_entropy_with_logits(final_logits, target, reduction="none") * missed_fg).sum() / missed_fg.sum().clamp_min(1.0)
    else:
        pos_loss = final_logits.sum() * 0.0

    residual = out["dher_visibility"]["residual"]
    target_low = _at(target, residual, mode="nearest")
    e_mask = evidence_mask(out, residual).detach()
    sparse_loss = (residual.abs() * (1.0 - e_mask)).mean()
    bg_loss = (F.relu(residual) * (target_low <= 0.5).float()).sum() / (target_low <= 0.5).float().sum().clamp_min(1.0)
    residual_rms = residual.detach().float().square().mean().sqrt()
    return {
        "visibility_pos": pos_loss,
        "visibility_sparse": sparse_loss,
        "visibility_bg": bg_loss,
        "missed_fg_fraction": missed_fg.detach().float().mean(),
        "visibility_evidence_mask_mean": e_mask.detach().float().mean(),
        "visibility_residual_rms": residual_rms,
    }


def objective(out: dict, baseline_out: dict, batch: dict, seg: ReTraceLoss, cfg: dict, threshold: float) -> dict[str, torch.Tensor]:
    terms = original_objective(out, batch, seg, cfg)
    vis = targeted_visibility_losses(out, baseline_out, batch, threshold, cfg)
    pos_w = float(cfg["training"].get("visibility_pos_weight", 0.5))
    sparse_w = float(cfg["training"].get("visibility_sparse_weight", 0.05))
    bg_w = float(cfg["training"].get("visibility_bg_weight", 0.05))
    terms.update(vis)
    terms["total"] = terms["total"] + pos_w * vis["visibility_pos"] + sparse_w * vis["visibility_sparse"] + bg_w * vis["visibility_bg"]
    return terms


def set_trainable(model: torch.nn.Module) -> list[str]:
    prefixes = ("dher_visibility.", "mask_head.")
    names = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    model.eval()
    model.dher_visibility.train()
    model.mask_head.train()
    return names


def optimizer_for(model: torch.nn.Module, visibility_lr: float, mask_lr: float) -> torch.optim.Optimizer:
    specs = [("dher_visibility.", visibility_lr), ("mask_head.", mask_lr)]
    groups = []
    for prefix, lr in specs:
        params = [p for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01, "name": prefix})
    if not groups:
        raise RuntimeError("no trainable parameters")
    return torch.optim.AdamW(groups)


def find_probe_batch(cfg: dict) -> tuple[dict, int]:
    for index, batch in enumerate(train_loader_for(cfg), 1):
        if any("tp_s" in str(path).lower() for path in batch["path"]) and bool(batch["mask"].any()):
            return batch, index
        if index >= 1024:
            break
    raise RuntimeError("no positive Tp_S batch found in first 1024 training batches")


def audit(model, baseline, batch, seg, cfg, threshold: float, pristine_state) -> dict:
    names = set_trainable(model)
    opt = optimizer_for(model, visibility_lr=1e-3, mask_lr=1e-5)
    with torch.no_grad():
        old = forward(baseline, batch)["mask_logits"]
        new = forward(model, batch)
        identity = float((new["mask_logits"] - old).abs().max())
        residual_rms = float(new["dher_visibility"]["residual_rms"])
    if identity >= 1e-6 or residual_rms != 0.0:
        raise AssertionError((identity, residual_rms))
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        base = forward(baseline, batch)
    out = forward(model, batch)
    terms = objective(out, base, batch, seg, cfg, threshold)
    terms["total"].backward()
    grad_groups = {}
    for group in opt.param_groups:
        vals = [float(p.grad.detach().square().sum()) for p in group["params"] if p.grad is not None]
        grad_groups[group["name"]] = sum(vals) ** 0.5
        if not math.isfinite(grad_groups[group["name"]]) or grad_groups[group["name"]] <= 0:
            raise AssertionError((group["name"], grad_groups[group["name"]]))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
    opt.step()
    updates = {}
    for group in opt.param_groups:
        prefix = group["name"]
        updates[prefix] = sum(float((p.detach().cpu() - before[n]).abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and n in before)
        if updates[prefix] <= 0:
            raise AssertionError((prefix, updates[prefix]))
    model.load_state_dict(pristine_state, strict=True)
    opt.zero_grad(set_to_none=True)
    return {
        "status": "PASS",
        "identity_max_abs_diff": identity,
        "initial_residual_rms": residual_rms,
        "loss": float(terms["total"].detach()),
        "visibility_pos": float(terms["visibility_pos"].detach()),
        "visibility_sparse": float(terms["visibility_sparse"].detach()),
        "visibility_bg": float(terms["visibility_bg"].detach()),
        "missed_fg_fraction": float(terms["missed_fg_fraction"].detach()),
        "grad_norm": grad_groups,
        "parameter_delta_abs_sum": updates,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_examples": names,
    }


def train(model, baseline, cfg, val_records, outdir: Path, summary: dict, epochs: int, threshold: float, visibility_lr: float, mask_lr: float) -> dict:
    names = set_trainable(model)
    baseline.eval()
    opt = optimizer_for(model, visibility_lr, mask_lr)
    seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
    loader = train_loader_for(cfg)
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 4))
    best = {"value": -math.inf, "epoch": None}
    summary["phase"] = {
        "name": "targeted_visibility",
        "epochs": epochs,
        "visibility_lr": visibility_lr,
        "mask_head_lr": mask_lr,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_examples": names,
        "history": [],
    }
    for epoch in range(1, epochs + 1):
        model.eval()
        model.dher_visibility.train()
        model.mask_head.train()
        opt.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        started = time.time()
        summary.update(status="training", current_epoch=epoch)
        dump(outdir / "summary.yaml", summary)
        for step, batch in enumerate(loader, 1):
            with torch.no_grad():
                base = forward(baseline, batch)
            out = forward(model, batch)
            terms = objective(out, base, batch, seg, cfg, threshold)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
            (terms["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            if step == 1 or step % 50 == 0 or step == len(loader):
                elapsed = time.time() - started
                progress = {
                    "epoch": epoch,
                    "epochs": epochs,
                    "step": step,
                    "steps": len(loader),
                    "loss": totals["total"] / step,
                    "visibility_pos": totals.get("visibility_pos", 0.0) / step,
                    "visibility_sparse": totals.get("visibility_sparse", 0.0) / step,
                    "visibility_bg": totals.get("visibility_bg", 0.0) / step,
                    "missed_fg_fraction": totals.get("missed_fg_fraction", 0.0) / step,
                    "visibility_residual_rms": totals.get("visibility_residual_rms", 0.0) / step,
                    "elapsed_minutes": elapsed / 60,
                    "eta_minutes": elapsed / step * (len(loader) - step) / 60,
                }
                dump(outdir / "progress.yaml", progress)
                print("[train]", json.dumps(progress), flush=True)
        result = e21.evaluate(model, val_records, cfg, f"epoch_{epoch:02d}_original_val", outdir)
        score = result["metrics"]["fake"]["oracle"]["f1"]
        row = {
            "epoch": epoch,
            "selection_value": score,
            "delta_vs_E21": score - E21_BEST_F1,
            "train_losses": {k: v / len(loader) for k, v in totals.items()},
            "validation": result,
            "elapsed_seconds": time.time() - started,
        }
        if score > best["value"]:
            best = {"value": score, "epoch": epoch}
            checkpoint = {
                "epoch": epoch,
                "model": e21.copy_state(model),
                "optimizer": opt.state_dict(),
                "config": cfg,
                "eval_variant": "raw",
                "phase": "targeted_visibility",
                "best_metric": score,
                "selection_threshold": result["metrics"]["fake"]["oracle"]["threshold"],
            }
            atomic_torch_save(checkpoint, outdir / "best.pt")
            summary.update(best_F1=score, best_epoch=epoch, best_checkpoint=str(outdir / "best.pt"))
            row["is_best"] = True
        else:
            row["is_best"] = False
        summary["phase"]["history"].append(row)
        dump(outdir / "summary.yaml", summary)
        print("[epoch_done]", json.dumps({"epoch": epoch, "fake_F1": score, "delta_vs_E21": score - E21_BEST_F1, "best_F1": best["value"]}), flush=True)
    return best


def main(args) -> None:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.yaml"
    if summary_path.exists():
        raise RuntimeError(f"refuse duplicate run: {summary_path}")
    summary = {
        "status": "initializing",
        "experiment": EXPERIMENT,
        "source": "E21 best",
        "single_change": "targeted DHER visibility loss plus tiny mask_head LR",
        "test_run": False,
        "selection": "val_original_grid_fake_pooled_F1_best",
        "stop_reason": None,
    }
    dump(summary_path, summary)
    try:
        set_seed(3407)
        source = torch.load(args.checkpoint, map_location="cpu")
        source_cfg = source["config"]
        threshold = float(source.get("selection_threshold", source_cfg.get("evaluation", {}).get("threshold", 0.5)))
        cfg = make_config(source_cfg)
        dump(outdir / "config.yaml", cfg)
        baseline = build_final_model(source_cfg["model"]).cuda()
        baseline.load_state_dict(source["model"], strict=True)
        baseline.eval()
        model = build_final_model(cfg["model"]).cuda()
        init_report = load_source(model, source["model"])
        pristine = e21.copy_state(model)
        probe, scanned = find_probe_batch(cfg)
        seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
        preflight = audit(model, baseline, probe, seg, cfg, threshold, pristine)
        summary.update(
            source_checkpoint=args.checkpoint,
            source_best_F1=float(source.get("best_metric", E21_BEST_F1)),
            source_selection_threshold=threshold,
            init_report=init_report,
            protocol="original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA",
            preflight_batch_index=scanned,
            preflight=preflight,
        )
        dump(outdir / "preflight.yaml", summary)
        dump(summary_path, summary)
        print("[preflight]", json.dumps({
            "identity": preflight["identity_max_abs_diff"],
            "initial_residual_rms": preflight["initial_residual_rms"],
            "missed_fg_fraction": preflight["missed_fg_fraction"],
            "audit": preflight["status"],
            "trainable_count": preflight["trainable_count"],
        }), flush=True)
        val_records = e21.read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
        best = train(model, baseline, cfg, val_records, outdir, summary, args.epochs, threshold, args.visibility_lr, args.mask_lr)
        best_state = torch.load(outdir / "best.pt", map_location="cpu")
        model.load_state_dict(best_state["model"], strict=True)
        full = e21.evaluate(model, val_records, cfg, "best_original_val", outdir)
        model.disable_dher_visibility = True
        ablation = e21.evaluate(model, val_records, cfg, "best_visibility_ablation_original_val", outdir)
        model.disable_dher_visibility = False
        summary.update(
            status="completed",
            stop_reason=f"fixed_{args.epochs}_epoch_targeted_visibility_budget_completed",
            best_F1=full["metrics"]["fake"]["oracle"]["f1"],
            best_epoch=best["epoch"],
            final_full_metrics=full,
            final_visibility_ablation_metrics=ablation,
            visibility_contribution_fake_F1=full["metrics"]["fake"]["oracle"]["f1"] - ablation["metrics"]["fake"]["oracle"]["f1"],
            visibility_contribution_TpS_gte20_fixed_F1=full["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1", 0.0) - ablation["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1", 0.0),
            acceptance_pass=bool(
                full["metrics"]["fake"]["oracle"]["f1"] >= 0.800
                and full["metrics"].get("Tp_S:gte_20", {}).get("oracle", {}).get("f1", 0.0) >= ablation["metrics"].get("Tp_S:gte_20", {}).get("oracle", {}).get("f1", 0.0)
            ),
        )
        dump(summary_path, summary)
        print("[TARGETED_VISIBILITY_COMPLETE]", json.dumps({
            "fake_F1": summary["best_F1"],
            "delta_vs_E21": summary["best_F1"] - E21_BEST_F1,
            "visibility_contribution_fake_F1": summary["visibility_contribution_fake_F1"],
            "TpS_gte20_fixed_F1": full["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1"),
            "TpS_gte20_oracle_F1": full["metrics"].get("Tp_S:gte_20", {}).get("oracle", {}).get("f1"),
            "visibility_contribution_TpS_gte20_fixed_F1": summary["visibility_contribution_TpS_gte20_fixed_F1"],
            "acceptance_pass": summary["acceptance_pass"],
        }), flush=True)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--visibility-lr", type=float, default=2e-4)
    parser.add_argument("--mask-lr", type=float, default=2e-6)
    main(parser.parse_args())
