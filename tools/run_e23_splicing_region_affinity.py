"""E23: donor/host affinity supervision plus edge-aware decoder propagation."""
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
from train_v2_final_pipeline import clean_loss_config, dher_v2_aux_loss, train_loader_for, tversky_loss, weighted_deep_supervision_loss


EXPERIMENT = "E23_splicing_donor_host_affinity"
PHASE = "donor_host_affinity_adapt"
AFFINITY_WEIGHT = 0.10


def dump(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def make_config(source: dict) -> dict:
    cfg = copy.deepcopy(source)
    cfg["experiment"] = EXPERIMENT
    cfg["model"].update(
        enable_dct_parallel=True,
        enable_dct_gate=False,
        enable_dct_evidence_head=False,
        enable_splicing_region_affinity=True,
        splicing_affinity_channels=32,
    )
    cfg["training"].update(
        ema=False,
        amp=False,
        batch_size=1,
        gradient_accumulation_steps=4,
        grad_clip=0.5,
        splicing_affinity_loss_weight=AFFINITY_WEIGHT,
        selection_metric="val_original_grid_fake_pooled_F1_best",
    )
    return cfg


def load_source(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict:
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    bad = [key for key in missing if not key.startswith("splicing_region_affinity.")]
    if bad or unexpected:
        raise RuntimeError(f"source init mismatch bad_missing={bad[:20]} unexpected={unexpected[:20]}")
    return {"missing": missing, "missing_count": len(missing), "unexpected": unexpected}


def forward(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        batch["image"].cuda(non_blocking=True),
        dct=batch["dct"].cuda(non_blocking=True),
        qtable=batch["qtable"].cuda(non_blocking=True),
        dct_available=batch["dct_available"].cuda(non_blocking=True),
    )


def _is_splicing(path: str) -> bool:
    name = Path(path).name.lower()
    return "tp_s" in name or "splicing" in name


def affinity_loss(out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    diag = out["splicing_region"]
    horizontal_logits = diag["horizontal_logits"]
    vertical_logits = diag["vertical_logits"]
    target = F.interpolate(batch["mask"].cuda(non_blocking=True).float(), size=(horizontal_logits.shape[-2], vertical_logits.shape[-1]), mode="nearest") > 0.5
    paths = list(batch["path"])
    losses = []
    crossing_edges = 0
    same_edges = 0
    used = 0
    for index, path in enumerate(paths):
        if not _is_splicing(str(path)) or not bool(target[index].any()):
            continue
        used += 1
        ht = (target[index:index + 1, :, :, 1:] == target[index:index + 1, :, :, :-1]).float()
        vt = (target[index:index + 1, :, 1:, :] == target[index:index + 1, :, :-1, :]).float()
        hl = F.binary_cross_entropy_with_logits(horizontal_logits[index:index + 1], ht, reduction="none")
        vl = F.binary_cross_entropy_with_logits(vertical_logits[index:index + 1], vt, reduction="none")
        same_h, cross_h = ht > 0.5, ht <= 0.5
        same_v, cross_v = vt > 0.5, vt <= 0.5
        same = torch.cat([hl[same_h], vl[same_v]])
        cross = torch.cat([hl[cross_h], vl[cross_v]])
        crossing_edges += int(cross.numel())
        same_edges += int(same.numel())
        if cross.numel() > 0:
            losses.append(0.5 * same.mean() + 0.5 * cross.mean())
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = horizontal_logits.sum() * 0.0 + vertical_logits.sum() * 0.0
    return loss, {"splicing_samples": used, "crossing_edges": crossing_edges, "same_edges": same_edges}


def objective(out: dict, batch: dict, seg: ReTraceLoss, cfg: dict) -> tuple[dict[str, torch.Tensor], dict]:
    y = batch["mask"].cuda(non_blocking=True)
    label = batch["label"].cuda(non_blocking=True)
    terms = seg(out, y, label)
    aux = dher_v2_aux_loss(out, y, cfg)
    deep = weighted_deep_supervision_loss(out, y, cfg)
    tv = tversky_loss(out["mask_logits"], y, float(cfg["loss"].get("tversky_alpha_fp", 0.3)), float(cfg["loss"].get("tversky_beta_fn", 0.7)))
    affinity, affinity_counts = affinity_loss(out, batch)
    terms["total"] = terms["total"] + aux["dher_aux_total"] + deep + float(cfg["loss"]["weights"].get("tversky", 0.0)) * tv + AFFINITY_WEIGHT * affinity
    terms.update(weighted_deep_supervision=deep, tversky=tv, donor_host_affinity=affinity)
    return terms, affinity_counts


def set_trainable(model: torch.nn.Module) -> list[str]:
    prefixes = ("splicing_region_affinity.", "decoder.", "mask_head.", "log_var_head.")
    names = []
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    model.eval()
    for module_name in ("splicing_region_affinity", "decoder", "mask_head", "log_var_head"):
        getattr(model, module_name).train()
    if model.dct_branch is not None:
        model.dct_branch.eval()
    return names


def optimizer_for(model: torch.nn.Module) -> torch.optim.Optimizer:
    specs = [
        ("splicing_region_affinity.", 2e-5),
        ("decoder.", 2e-6),
        ("mask_head.", 2e-6),
        ("log_var_head.", 2e-6),
    ]
    groups = []
    for prefix, lr in specs:
        params = [p for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01, "name": prefix})
    return torch.optim.AdamW(groups)


def find_splicing_batch(cfg: dict) -> tuple[dict, int]:
    for index, batch in enumerate(train_loader_for(cfg), 1):
        if any(_is_splicing(str(path)) for path in batch["path"]) and bool(batch["mask"].any()):
            return batch, index
        if index >= 1024:
            break
    raise RuntimeError("no positive Tp_S batch found in first 1024 training batches")


def audit(model, batch, seg, cfg, pristine_state) -> dict:
    names = set_trainable(model)
    optimizer = optimizer_for(model)
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    out = forward(model, batch)
    terms, counts = objective(out, batch, seg, cfg)
    terms["total"].backward()
    groups = {}
    for group in optimizer.param_groups:
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(group["name"]) and p.requires_grad]
        grad_norm = sum(float(p.grad.detach().square().sum()) for _, p in named if p.grad is not None) ** 0.5
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise AssertionError((group["name"], grad_norm))
        groups[group["name"]] = {"lr": group["lr"], "grad_norm": grad_norm}
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
    optimizer.step()
    for prefix, row in groups.items():
        delta = sum(float((p.detach().cpu() - before[n]).abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix))
        if delta <= 0:
            raise AssertionError((prefix, delta))
        row["parameter_delta_abs_sum"] = delta
    model.load_state_dict(pristine_state, strict=True)
    optimizer.zero_grad(set_to_none=True)
    return {"status": "PASS", "loss": float(terms["total"].detach()), "affinity_counts": counts, "groups": groups, "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_examples": names[:30]}


def train(model, cfg, val_records, outdir: Path, summary: dict, epochs: int) -> dict:
    names = set_trainable(model)
    optimizer = optimizer_for(model)
    seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
    loader = train_loader_for(cfg)
    accumulation = int(cfg["training"].get("gradient_accumulation_steps", 1))
    best = {"value": -math.inf, "epoch": None}
    summary["phase"] = {"name": PHASE, "epochs": epochs, "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_examples": names[:40], "optimizer": [{"name": g["name"], "lr": g["lr"]} for g in optimizer.param_groups], "history": []}
    for epoch in range(1, epochs + 1):
        model.eval()
        for module_name in ("splicing_region_affinity", "decoder", "mask_head", "log_var_head"):
            getattr(model, module_name).train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        counts = {"splicing_samples": 0, "crossing_edges": 0, "same_edges": 0}
        started = time.time()
        summary.update(status="training", current_epoch=epoch)
        dump(outdir / "summary.yaml", summary)
        for step, batch in enumerate(loader, 1):
            out = forward(model, batch)
            terms, batch_counts = objective(out, batch, seg, cfg)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
            (terms["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            for key in counts:
                counts[key] += int(batch_counts[key])
            if step == 1 or step % 50 == 0 or step == len(loader):
                elapsed = time.time() - started
                progress = {"epoch": epoch, "epochs": epochs, "step": step, "steps": len(loader), "loss": totals["total"] / step, "affinity_loss": totals.get("donor_host_affinity", 0.0) / step, "splicing_samples": counts["splicing_samples"], "elapsed_minutes": elapsed / 60, "eta_minutes": elapsed / step * (len(loader) - step) / 60}
                dump(outdir / "progress.yaml", progress)
                print("[train]", json.dumps(progress), flush=True)
        result = e21.evaluate(model, val_records, cfg, f"epoch_{epoch:02d}_original_val", outdir)
        score = result["metrics"]["fake"]["oracle"]["f1"]
        row = {"epoch": epoch, "selection_value": score, "train_losses": {k: v / len(loader) for k, v in totals.items()}, "affinity_counts": counts, "validation": result, "elapsed_seconds": time.time() - started}
        if score > best["value"]:
            best = {"value": score, "epoch": epoch}
            checkpoint = {"epoch": epoch, "model": e21.copy_state(model), "optimizer": optimizer.state_dict(), "config": cfg, "eval_variant": "raw", "phase": PHASE, "best_metric": score, "selection_threshold": result["metrics"]["fake"]["oracle"]["threshold"]}
            atomic_torch_save(checkpoint, outdir / "best.pt")
            summary.update(best_F1=score, best_epoch=epoch, best_checkpoint=str(outdir / "best.pt"))
            row["is_best"] = True
        else:
            row["is_best"] = False
        summary["phase"]["history"].append(row)
        dump(outdir / "summary.yaml", summary)
        print("[epoch_done]", json.dumps({"epoch": epoch, "fake_F1": score, "best_F1": best["value"]}), flush=True)
    return best


def main(args) -> None:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.yaml"
    if summary_path.exists():
        raise RuntimeError(f"refuse duplicate run: {summary_path}")
    summary = {"status": "initializing", "experiment": EXPERIMENT, "single_change": "learn Tp_S donor/host adjacency affinity from residual+DCT evidence and use zero-init edge-aware propagation on D1", "source": "E21 best", "test_run": False, "acceptance": {"fake_F1_min": 0.800, "TpS_gte20_fixed_F1_gain_min": 0.02}, "stop_reason": None}
    dump(summary_path, summary)
    try:
        set_seed(3407)
        source = torch.load(args.checkpoint, map_location="cpu")
        source_cfg = source["config"]
        cfg = make_config(source_cfg)
        dump(outdir / "config.yaml", cfg)
        baseline = build_final_model(source_cfg["model"]).cuda()
        baseline.load_state_dict(source["model"], strict=True)
        model = build_final_model(cfg["model"]).cuda()
        init_report = load_source(model, source["model"])
        probe, scanned = find_splicing_batch(cfg)
        baseline.eval(); model.eval()
        with torch.no_grad():
            old = forward(baseline, probe)["mask_logits"]
            new_out = forward(model, probe)
            identity = float((new_out["mask_logits"] - old).abs().max())
            write_rms = float(new_out["splicing_region"]["write_rms"])
            write_weight_rms = float(new_out["splicing_region"]["write_weight_rms"])
        if identity >= 1e-6 or write_rms != 0.0 or write_weight_rms != 0.0:
            raise AssertionError((identity, write_rms, write_weight_rms))
        pristine = e21.copy_state(model)
        seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
        audit_result = audit(model, probe, seg, cfg, pristine)
        summary.update(source_checkpoint=args.checkpoint, source_best_F1=float(source.get("best_metric", -1)), init_report=init_report, protocol="original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA", preflight_batch_index=scanned, zero_init_identity_max_abs_diff=identity, initial_region_write_rms=write_rms, initial_write_weight_rms=write_weight_rms, preflight=audit_result)
        dump(outdir / "preflight.yaml", summary)
        dump(summary_path, summary)
        print("[preflight]", json.dumps({"identity": identity, "write_rms": write_rms, "audit": audit_result["status"], "affinity_counts": audit_result["affinity_counts"]}), flush=True)
        val_records = e21.read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
        best = train(model, cfg, val_records, outdir, summary, args.epochs)
        best_state = torch.load(outdir / "best.pt", map_location="cpu")
        model.load_state_dict(best_state["model"], strict=True)
        full = e21.evaluate(model, val_records, cfg, "best_original_val", outdir)
        model.disable_splicing_region = True
        ablation = e21.evaluate(model, val_records, cfg, "best_region_ablation_original_val", outdir)
        model.disable_splicing_region = False
        full_fake = full["metrics"]["fake"]["oracle"]["f1"]
        full_large = full["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1", 0.0)
        ablate_large = ablation["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("f1", 0.0)
        summary.update(status="completed", stop_reason="fixed_4_epoch_budget_completed", best_F1=full_fake, best_epoch=best["epoch"], final_full_metrics=full, final_region_ablation_metrics=ablation, region_contribution_fake_F1=full_fake - ablation["metrics"]["fake"]["oracle"]["f1"], region_contribution_TpS_gte20_fixed_F1=full_large - ablate_large, acceptance_pass=bool(full_fake >= 0.800 and full_large - ablate_large >= 0.02))
        dump(summary_path, summary)
        print("[E23_COMPLETE]", json.dumps({"fake_F1": full_fake, "TpS_gte20_fixed_F1": full_large, "region_contribution_fake_F1": summary["region_contribution_fake_F1"], "region_contribution_TpS_gte20_fixed_F1": summary["region_contribution_TpS_gte20_fixed_F1"], "acceptance_pass": summary["acceptance_pass"]}), flush=True)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    main(parser.parse_args())
