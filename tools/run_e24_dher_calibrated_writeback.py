"""E24: evidence-calibrated DHER write-back from E21 visibility-only best.

Single hypothesis: previous DHER visibility helped a little, but targeted loss made
logit residuals too large. Keep the E21 visibility-only checkpoint as the exact
step0 function and add a bounded, confidence-gated D1/logit write-back on top.
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


EXPERIMENT = "E24_DHER_calibrated_writeback"
SOURCE_BEST_F1 = 0.7980785749644633
E21_DCT_F1 = 0.7977327505234914


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def make_config(source_cfg: dict) -> dict:
    cfg = copy.deepcopy(source_cfg)
    cfg["experiment"] = EXPERIMENT
    cfg["model"]["enable_dher_visibility"] = True
    cfg["model"]["enable_dher_calibrated_writeback"] = True
    cfg["model"]["enable_splicing_region_affinity"] = False
    cfg["model"]["enable_region_consistency"] = False
    cfg["training"].update(
        ema=False,
        amp=False,
        batch_size=1,
        gradient_accumulation_steps=4,
        grad_clip=0.5,
        selection_metric="val_original_grid_fake_pooled_F1_best",
    )
    cfg.setdefault("e24", {})
    cfg["e24"].update(
        residual_l1=0.01,
        bg_positive=0.02,
        feature_l1=0.005,
        source_best_f1=SOURCE_BEST_F1,
        catastrophic_drop=0.003,
        max_safe_logit_residual_rms=0.35,
    )
    return cfg


def load_source(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict:
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad_missing = [key for key in missing if not key.startswith("dher_calibrated_writeback.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"source init mismatch bad_missing={bad_missing[:50]} unexpected={unexpected[:50]}")
    return {"missing": missing, "missing_count": len(missing), "unexpected": unexpected}


def forward(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        batch["image"].cuda(non_blocking=True),
        dct=batch["dct"].cuda(non_blocking=True),
        qtable=batch["qtable"].cuda(non_blocking=True),
        dct_available=batch["dct_available"].cuda(non_blocking=True),
    )


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


def objective(out: dict, batch: dict, seg: ReTraceLoss, cfg: dict) -> dict[str, torch.Tensor]:
    terms = original_objective(out, batch, seg, cfg)
    calib = out["dher_calibrated_writeback"]
    target = batch["mask"].cuda(non_blocking=True)
    low_target = F.interpolate(target, size=calib["logit_residual"].shape[-2:], mode="nearest")
    bg = (low_target < 0.5).float()
    logit_residual = calib["logit_residual"]
    feature_residual = calib["feature_residual"]
    residual_l1 = logit_residual.abs().mean()
    bg_positive = (F.relu(logit_residual) * bg).sum() / bg.sum().clamp_min(1.0)
    feature_l1 = feature_residual.abs().mean()
    e24 = cfg.get("e24", {})
    calibrated_regularization = (
        float(e24.get("residual_l1", 0.01)) * residual_l1
        + float(e24.get("bg_positive", 0.02)) * bg_positive
        + float(e24.get("feature_l1", 0.005)) * feature_l1
    )
    terms["calibrated_regularization"] = calibrated_regularization
    terms["calib_residual_l1"] = residual_l1
    terms["calib_bg_positive"] = bg_positive
    terms["calib_feature_l1"] = feature_l1
    terms["total"] = terms["total"] + calibrated_regularization
    return terms


def set_trainable(model: torch.nn.Module) -> list[str]:
    train_prefixes = (
        "dher_calibrated_writeback.",
        "mask_head.",
        "decoder.stage1.fuse.3.",
        "decoder.stage1.fuse.4.",
        "decoder.stage1.fuse.5.",
    )
    names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(train_prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    model.eval()
    model.dher_calibrated_writeback.train()
    model.mask_head.train()
    model.decoder.stage1.fuse[3].train()
    model.decoder.stage1.fuse[4].train()
    model.decoder.stage1.fuse[5].train()
    if model.dher_visibility is not None:
        model.dher_visibility.eval()
    return names


def optimizer_for(model: torch.nn.Module, calib_lr: float, decoder_lr: float) -> torch.optim.Optimizer:
    groups = []
    calib = [p for n, p in model.named_parameters() if n.startswith("dher_calibrated_writeback.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if n.startswith("mask_head.") and p.requires_grad]
    dec = [
        p for n, p in model.named_parameters()
        if (n.startswith("decoder.stage1.fuse.3.") or n.startswith("decoder.stage1.fuse.4.") or n.startswith("decoder.stage1.fuse.5.")) and p.requires_grad
    ]
    if not calib:
        raise RuntimeError("no dher_calibrated_writeback trainable parameters")
    groups.append({"params": calib, "lr": calib_lr, "weight_decay": 0.01, "name": "dher_calibrated_writeback."})
    if head:
        groups.append({"params": head, "lr": decoder_lr, "weight_decay": 0.0, "name": "mask_head."})
    if dec:
        groups.append({"params": dec, "lr": decoder_lr, "weight_decay": 0.01, "name": "decoder.stage1.last."})
    return torch.optim.AdamW(groups)


def find_probe_batch(cfg: dict) -> tuple[dict, int]:
    for index, batch in enumerate(train_loader_for(cfg), 1):
        if any("tp_s" in str(path).lower() for path in batch["path"]) and bool(batch["mask"].any()):
            return batch, index
        if index >= 1024:
            break
    raise RuntimeError("no positive Tp_S batch found in first 1024 training batches")


def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    return {
        n: float(p.grad.detach().square().sum().sqrt()) if p.grad is not None else 0.0
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def delta_abs_sum(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    return {n: float((p.detach().cpu() - before[n]).abs().sum()) for n, p in model.named_parameters() if n in before}


def audit(model, baseline, batch, seg, cfg, pristine_state) -> dict:
    names = set_trainable(model)
    opt = optimizer_for(model, calib_lr=1e-3, decoder_lr=1e-4)
    with torch.no_grad():
        old = forward(baseline, batch)["mask_logits"]
        new = forward(model, batch)
        identity = float((new["mask_logits"] - old).abs().max())
        diag = new["dher_calibrated_writeback"]
        initial = {
            "feature_residual_rms": float(diag["feature_residual_rms"]),
            "logit_residual_rms": float(diag["logit_residual_rms"]),
            "feature_scale": float(diag["feature_scale"]),
            "logit_scale": float(diag["logit_scale"]),
            "gate_mean": float(diag["gate_mean"]),
        }
    if identity >= 1e-6 or initial["feature_residual_rms"] != 0.0 or initial["logit_residual_rms"] != 0.0:
        raise AssertionError({"identity": identity, "initial": initial})

    before1 = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    out = forward(model, batch)
    terms = objective(out, batch, seg, cfg)
    terms["total"].backward()
    grads1 = grad_norms(model)
    scale_grad_ok = grads1.get("dher_calibrated_writeback.logit_scale_raw", 0.0) > 0 or grads1.get("dher_calibrated_writeback.feature_scale_raw", 0.0) > 0
    if not scale_grad_ok:
        raise AssertionError(f"calibrated scales have no first-step grad: {grads1}")
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
    opt.step(); opt.zero_grad(set_to_none=True)
    updates1 = delta_abs_sum(model, before1)

    before2 = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    out = forward(model, batch)
    terms2 = objective(out, batch, seg, cfg)
    terms2["total"].backward()
    grads2 = grad_norms(model)
    write_grad_ok = grads2.get("dher_calibrated_writeback.logit_write.weight", 0.0) > 0 or grads2.get("dher_calibrated_writeback.feature_write.weight", 0.0) > 0
    if not write_grad_ok:
        raise AssertionError(f"calibrated write heads have no second-step grad: {grads2}")
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
    opt.step(); opt.zero_grad(set_to_none=True)
    updates2 = delta_abs_sum(model, before2)
    model.load_state_dict(pristine_state, strict=True)
    return {
        "status": "PASS",
        "identity_max_abs_diff": identity,
        "initial": initial,
        "loss_step1": float(terms["total"].detach()),
        "loss_step2": float(terms2["total"].detach()),
        "grad_norm_step1": grads1,
        "grad_norm_step2": grads2,
        "parameter_delta_abs_sum_step1": updates1,
        "parameter_delta_abs_sum_step2": updates2,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_examples": names,
    }


def compact_metrics(result: dict) -> dict:
    metrics = result["metrics"]
    out = {
        "fake_oracle_f1": metrics["fake"]["oracle"]["f1"],
        "fake_oracle_threshold": metrics["fake"]["oracle"].get("threshold"),
        "fake_fixed_f1": metrics["fake"]["fixed"]["f1"],
        "fake_fixed_precision": metrics["fake"]["fixed"]["precision"],
        "fake_fixed_recall": metrics["fake"]["fixed"]["recall"],
        "all_oracle_f1": metrics["all"]["oracle"]["f1"] if "all" in metrics else None,
    }
    for key in ["Tp_S:lt_1", "Tp_S:gte_20", "Tp_S:5_20", "Tp_D:gte_20"]:
        if key in metrics:
            out[key] = {
                "oracle_f1": metrics[key]["oracle"]["f1"],
                "fixed_f1": metrics[key]["fixed"]["f1"],
                "fixed_precision": metrics[key]["fixed"].get("precision"),
                "fixed_recall": metrics[key]["fixed"].get("recall"),
            }
    return out


def train(model, cfg, val_records, outdir: Path, summary: dict, epochs: int, calib_lr: float, decoder_lr: float) -> dict:
    names = set_trainable(model)
    opt = optimizer_for(model, calib_lr, decoder_lr)
    seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
    loader = train_loader_for(cfg)
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 4))
    best = {"value": -math.inf, "epoch": None}
    summary["phase"] = {
        "name": "evidence_calibrated_writeback",
        "epochs": epochs,
        "calib_lr": calib_lr,
        "decoder_lr": decoder_lr,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_examples": names,
        "history": [],
    }
    for epoch in range(1, epochs + 1):
        model.eval()
        model.dher_calibrated_writeback.train()
        model.mask_head.train()
        model.decoder.stage1.fuse[3].train()
        model.decoder.stage1.fuse[4].train()
        model.decoder.stage1.fuse[5].train()
        opt.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        diag_sums: dict[str, float] = {}
        started = time.time()
        summary.update(status="training", current_epoch=epoch)
        dump(outdir / "summary.yaml", summary)
        for step, batch in enumerate(loader, 1):
            out = forward(model, batch)
            terms = objective(out, batch, seg, cfg)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
            (terms["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            diag = out["dher_calibrated_writeback"]
            for key in ["logit_residual_rms", "feature_residual_rms", "gate_mean", "logit_scale", "feature_scale"]:
                diag_sums[key] = diag_sums.get(key, 0.0) + float(diag[key])
            if step == 1 or step % 50 == 0 or step == len(loader):
                elapsed = time.time() - started
                progress = {
                    "epoch": epoch,
                    "epochs": epochs,
                    "step": step,
                    "steps": len(loader),
                    "loss": totals["total"] / step,
                    "calib_logit_residual_rms": diag_sums["logit_residual_rms"] / step,
                    "calib_feature_residual_rms": diag_sums["feature_residual_rms"] / step,
                    "gate_mean": diag_sums["gate_mean"] / step,
                    "logit_scale": diag_sums["logit_scale"] / step,
                    "feature_scale": diag_sums["feature_scale"] / step,
                    "elapsed_minutes": elapsed / 60,
                    "eta_minutes": elapsed / step * (len(loader) - step) / 60,
                }
                dump(outdir / "progress.yaml", progress)
                print("[train]", json.dumps(progress), flush=True)
        result = e21.evaluate(model, val_records, cfg, f"epoch_{epoch:02d}_original_val", outdir)
        score = result["metrics"]["fake"]["oracle"]["f1"]
        diag_avg = {k: v / len(loader) for k, v in diag_sums.items()}
        row = {
            "epoch": epoch,
            "selection_value": score,
            "delta_vs_source_visibility": score - SOURCE_BEST_F1,
            "delta_vs_E21_DCT": score - E21_DCT_F1,
            "train_losses": {k: v / len(loader) for k, v in totals.items()},
            "calibration_diag": diag_avg,
            "validation_compact": compact_metrics(result),
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
                "phase": "evidence_calibrated_writeback",
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
        print("[epoch_done]", json.dumps({"epoch": epoch, "fake_F1": score, "delta_vs_source_visibility": score - SOURCE_BEST_F1, "best_F1": best["value"], **diag_avg}), flush=True)
        e24 = cfg.get("e24", {})
        if epoch == 1 and score < SOURCE_BEST_F1 - float(e24.get("catastrophic_drop", 0.003)) and diag_avg.get("logit_residual_rms", 0.0) > float(e24.get("max_safe_logit_residual_rms", 0.35)):
            summary.update(status="stopped", stop_reason="catastrophic_epoch1_drop_with_high_calibrated_residual")
            dump(outdir / "summary.yaml", summary)
            print("[early_stop] catastrophic_epoch1_drop_with_high_calibrated_residual", flush=True)
            return best
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
        "source": "E21 visibility-only best",
        "single_change": "bounded confidence-gated calibrated evidence write-back on top of E21 visibility-only checkpoint",
        "test_run": False,
        "selection": "val_original_grid_fake_pooled_F1_best",
        "stop_reason": None,
    }
    dump(summary_path, summary)
    try:
        set_seed(args.seed)
        source = torch.load(args.checkpoint, map_location="cpu")
        source_cfg = source["config"]
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
        preflight = audit(model, baseline, probe, seg, cfg, pristine)
        summary.update(
            source_checkpoint=args.checkpoint,
            source_best_F1=float(source.get("best_metric", SOURCE_BEST_F1)),
            init_report=init_report,
            protocol="original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA",
            preflight_batch_index=scanned,
            preflight=preflight,
        )
        dump(outdir / "preflight.yaml", summary)
        dump(summary_path, summary)
        print("[preflight]", json.dumps({
            "identity": preflight["identity_max_abs_diff"],
            "initial": preflight["initial"],
            "audit": preflight["status"],
            "trainable_count": preflight["trainable_count"],
        }), flush=True)
        val_records = e21.read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
        best = train(model, cfg, val_records, outdir, summary, args.epochs, args.calib_lr, args.decoder_lr)
        if (outdir / "best.pt").exists():
            best_state = torch.load(outdir / "best.pt", map_location="cpu")
            model.load_state_dict(best_state["model"], strict=True)
            full = e21.evaluate(model, val_records, cfg, "best_original_val", outdir)
            model.disable_dher_calibrated_writeback = True
            ablation = e21.evaluate(model, val_records, cfg, "best_calibrated_ablation_original_val", outdir)
            model.disable_dher_calibrated_writeback = False
            summary.update(
                final_full_metrics=full,
                final_calibrated_ablation_metrics=ablation,
                calibrated_contribution_fake_F1=full["metrics"]["fake"]["oracle"]["f1"] - ablation["metrics"]["fake"]["oracle"]["f1"],
                calibrated_contribution_TpS_gte20_fixed_F1=full["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1", 0.0) - ablation["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1", 0.0),
                acceptance_pass=bool(full["metrics"]["fake"]["oracle"]["f1"] >= 0.800),
            )
        if summary.get("status") != "stopped":
            summary.update(status="completed", stop_reason=f"fixed_{args.epochs}_epoch_e24_budget_completed")
        summary.update(best_F1=best["value"], best_epoch=best["epoch"])
        dump(summary_path, summary)
        print("[E24_COMPLETE]", json.dumps({
            "status": summary["status"],
            "fake_F1": summary.get("best_F1"),
            "delta_vs_source_visibility": (summary.get("best_F1") or 0.0) - SOURCE_BEST_F1,
            "acceptance_pass": summary.get("acceptance_pass"),
            "stop_reason": summary.get("stop_reason"),
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
    parser.add_argument("--calib-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    main(parser.parse_args())
