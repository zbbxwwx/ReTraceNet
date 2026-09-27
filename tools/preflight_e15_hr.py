"""Preflight checks for the E15-HR structural experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import (
    build_optimizer,
    clean_loss_config,
    e15_hr_loss,
    forward_from_batch,
    load_initial_weights_repaired,
    set_trainable,
)


AUDIT_PREFIXES = [
    "content_encoder",
    "e15_anchor_content_encoder",
    "forensic_stream",
    "qtr.",
    "hcv2.",
    "dher.",
    "decoder.stage1",
    "decoder.stage2",
    "decoder.stage3",
    "decoder.stage4",
    "e15_decoder.stage1",
    "e15_decoder.stage2",
    "e15_decoder.stage3",
    "e15_decoder.stage4",
    "e15_decoder.evidence_adapters",
    "e15_decoder.spatial_stem",
    "e15_decoder.up_to_half",
    "e15_decoder.r2_",
    "e15_decoder.up_to_full",
    "e15_decoder.r1_",
    "e15_decoder.region_head",
    "e15_decoder.boundary_head",
    "e15_decoder.aux_",
    "mask_head",
    "log_var_head",
    "aux_heads",
    "image_head",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--decoder-config")
    p.add_argument("--device")
    p.add_argument("--out")
    return p.parse_args()


def prefix_for(name: str) -> str:
    for prefix in AUDIT_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return "other"


def count_by_prefix(model: torch.nn.Module) -> dict[str, int]:
    out = {p: 0 for p in AUDIT_PREFIXES}
    out["other"] = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            out[prefix_for(name)] += param.numel()
    return {k: v for k, v in out.items() if v}


def optimizer_lrs_by_prefix(model: torch.nn.Module, opt: torch.optim.Optimizer) -> dict[str, list[float]]:
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, set[float]] = {}
    for group in opt.param_groups:
        lr = float(group["lr"])
        for p in group["params"]:
            name = id_to_name.get(id(p))
            if name is None:
                continue
            out.setdefault(prefix_for(name), set()).add(lr)
    return {k: sorted(v) for k, v in out.items()}


def shape_list(xs) -> list[list[int]]:
    return [list(x.shape) for x in xs]


def grouped_keys(keys: list[str]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for key in keys:
        parts = key.split(".")
        group = parts[0] if parts else "other"
        entry = groups.setdefault(group, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 6:
            entry["examples"].append(key)
    return groups


def init_group(key: str) -> str:
    mappings = [
        ("e15_anchor_content_encoder.", "e5_content_to_anchor"),
        ("content_encoder.", "hrnet_content"),
        ("forensic_stream.", "forensic_stream"),
        ("qtr.", "qtr"),
        ("hcv2.", "hcv2"),
        ("dher.", "dher"),
        ("decoder.", "old_decoder"),
        ("e15_decoder.", "e15_decoder"),
        ("mask_head.", "old_heads"),
        ("log_var_head.", "old_heads"),
        ("aux_heads.", "old_heads"),
        ("image_head.", "old_heads"),
    ]
    for prefix, group in mappings:
        if key.startswith(prefix):
            return group
    return "other"


def grouped_init_keys(keys: list[str]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for key in keys:
        group = init_group(key)
        entry = groups.setdefault(group, {"count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 6:
            entry["examples"].append(key)
    return groups


def tensor_rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().cpu())


def aggregate_audit(entries: dict[str, dict], prefixes: list[str], count_key: str) -> dict:
    selected = [v for k, v in entries.items() if any(k.startswith(prefix) for prefix in prefixes)]
    return {
        count_key: sum(int(v.get(count_key, 0)) for v in selected),
        "abs_sum": sum(float(v.get("abs_sum", 0.0)) for v in selected),
    }


def grad_and_step_audit(model: torch.nn.Module, cfg: dict, phase: str, batch: dict, device: torch.device) -> dict:
    model = copy.deepcopy(model).to(device)
    names = set_trainable(model, phase)
    opt = build_optimizer(model, cfg, phase)
    seg_loss = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).to(device)
    im = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt.zero_grad(set_to_none=True)
    out = forward_from_batch(model, batch, device, im)
    losses = e15_hr_loss(out, mask, label, seg_loss, cfg)
    losses["total"].backward()
    grad = {p: {"nonzero": 0, "abs_sum": 0.0} for p in AUDIT_PREFIXES + ["other"]}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        key = prefix_for(n)
        if g is not None:
            grad[key]["abs_sum"] += float(g.detach().abs().sum().cpu())
            if float(g.detach().abs().sum().cpu()) > 0:
                grad[key]["nonzero"] += 1
    opt.step()
    changed = {p: {"changed": 0, "abs_sum": 0.0} for p in AUDIT_PREFIXES + ["other"]}
    for n, p in model.named_parameters():
        if n not in before:
            continue
        diff = (p.detach() - before[n]).abs().sum()
        key = prefix_for(n)
        changed[key]["abs_sum"] += float(diff.cpu())
        if float(diff.cpu()) > 0:
            changed[key]["changed"] += 1
    return {
        "phase": phase,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_by_prefix": count_by_prefix(model),
        "optimizer_lrs_by_prefix": optimizer_lrs_by_prefix(model, opt),
        "trainable_examples": names[:40],
        "loss_total": float(losses["total"].detach().cpu()),
        "grad_by_prefix": {k: v for k, v in grad.items() if v["nonzero"] or v["abs_sum"]},
        "changed_by_prefix": {k: v for k, v in changed.items() if v["changed"] or v["abs_sum"]},
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    decoder_cfg = load_config(args.decoder_config) if args.decoder_config else cfg
    set_seed(int(cfg.get("seed", 3407)))
    device = choose_device(args.device)
    model = build_final_model(cfg["model"]).to(device)
    init_report = load_initial_weights_repaired(
        model,
        cfg["training"].get("init_checkpoint"),
        device,
        cfg["training"].get("init_load_patterns") or [],
        cfg["training"].get("reinitialize_prefixes") or [],
        cfg["training"].get("reinitialize_keys") or [],
        cfg["training"].get("init_key_remaps") or [],
    )
    model.eval()
    loader = build_loader({**cfg["data"], "batch_size": 1, "num_workers": 0}, "train", shuffle=False)
    batch = next(iter(loader))
    with torch.inference_mode():
        out = forward_from_batch(model, batch, device)
    identity = (out["mask_logits"] - out["e15_old_logits"]).abs()
    decoded = out["e15_decoded_features"]
    hcv = out.get("hcv2") or {}
    proposals = hcv.get("proposals", [])
    deltas = hcv.get("deltas", [])
    proposal_shapes = {}
    for i, ref in enumerate(decoded):
        source_index = min(i, max(0, len(proposals) - 1))
        if proposals:
            proposal_shapes[f"P{i+1}"] = list(model.e15_decoder._cue_at(proposals[source_index], ref).shape)
    beta_values = [float(adapter.beta.detach().cpu()) for adapter in model.e15_decoder.evidence_adapters]
    region_last = model.e15_decoder.region_head[-1]
    hrnet_load = getattr(model.content_encoder, "pretrained_load_report", {})
    shape_report = {
        "C1_C4_hrnet": shape_list(out["content_features"]),
        "N1_N4_forensic": shape_list(out["e15_detail_features"]),
        "P1_P4_adapter_inputs": proposal_shapes,
        "P4_source": "P3 resized to D4 (HCV defines three proposal scales)",
        "Delta1_Delta3": shape_list(deltas),
        "D1_D4": shape_list(decoded),
        "mask_logits": list(out["mask_logits"].shape),
        "old_logits": list(out["e15_old_logits"].shape),
        "residual": list(out["e15_residual_full"].shape),
        "boundary": list(out["e15_boundary_logits"].shape),
        "d1": list(out["e15_d1_logits"].shape),
        "d2": list(out["e15_d2_logits"].shape),
        "d3": list(out["e15_d3_logits"].shape),
        "r2": list(out["e15_highres"]["r2"].shape),
        "r1": list(out["e15_highres"]["r1"].shape),
    }
    allowed = tuple(cfg["training"].get("allowed_missing_prefixes") or [])
    allowed_keys = set(cfg["training"].get("allowed_missing_keys") or [])
    bad_missing = [k for k in init_report["missing"] if k not in allowed_keys and not any(k.startswith(p) for p in allowed)]
    warmup_audit = grad_and_step_audit(model, cfg, "e15_hr_warmup", batch, device)
    decoder_audit = grad_and_step_audit(model, decoder_cfg, "e15_hr_decoder_adapt", batch, device)
    required_prefixes = {
        "evidence_adapters": ["e15_decoder.evidence_adapters"],
        "spatial_stem": ["e15_decoder.spatial_stem"],
        "high_res_blocks": ["e15_decoder.up_to_half", "e15_decoder.r2_", "e15_decoder.up_to_full", "e15_decoder.r1_"],
        "region_head": ["e15_decoder.region_head"],
        "boundary_head": ["e15_decoder.boundary_head"],
    }
    required_updates = {}
    for name, prefixes in required_prefixes.items():
        grad_result = aggregate_audit(warmup_audit["grad_by_prefix"], prefixes, "nonzero")
        step_result = aggregate_audit(warmup_audit["changed_by_prefix"], prefixes, "changed")
        required_updates[name] = {
            "grad_nonzero_tensors": grad_result["nonzero"],
            "grad_abs_sum": grad_result["abs_sum"],
            "changed_tensors": step_result["changed"],
            "step_abs_change": step_result["abs_sum"],
            "pass": grad_result["nonzero"] > 0 and grad_result["abs_sum"] > 0 and step_result["changed"] > 0 and step_result["abs_sum"] > 0,
        }
    report = {
        "status": "PASS",
        "architecture": cfg["model"].get("architecture"),
        "parent": cfg["training"].get("parent", "E5"),
        "init": {
            "loaded": init_report["loaded"],
            "loaded_by_group": grouped_init_keys(init_report.get("loaded_keys", [])),
            "missing_count": len(init_report["missing"]),
            "missing_by_group": grouped_init_keys(init_report["missing"]),
            "missing_examples": init_report["missing"][:30],
            "unexpected_count": len(init_report["unexpected"]),
            "unexpected_examples": init_report["unexpected"][:30],
            "skipped_shape_count": len(init_report["skipped_shape"]),
            "skipped_shape_examples": init_report["skipped_shape"][:10],
            "bad_missing_count": len(bad_missing),
            "bad_missing_examples": bad_missing[:30],
        },
        "hrnet_imagenet_pretrained": {
            "path": hrnet_load.get("path", ""),
            "checkpoint_keys": hrnet_load.get("checkpoint_keys", 0),
            "matched_keys": hrnet_load.get("matched_keys", 0),
            "coverage_of_checkpoint": (
                float(hrnet_load.get("matched_keys", 0)) / max(1, int(hrnet_load.get("checkpoint_keys", 0)))
            ),
            "missing_count": len(hrnet_load.get("missing_keys", [])),
            "missing_by_structure": grouped_keys(hrnet_load.get("missing_keys", [])),
            "unexpected_count": len(hrnet_load.get("unexpected_keys", [])),
            "unexpected_by_structure": grouped_keys(hrnet_load.get("unexpected_keys", [])),
            "note": "Missing/unexpected keys are reported before acceptance; projection adapters are E15-specific and initialized separately.",
        },
        "shapes": shape_report,
        "zero_initialization": {
            "evidence_adapter_betas": beta_values,
            "all_betas_zero": all(v == 0.0 for v in beta_values),
            "region_residual_last_weight_rms": tensor_rms(region_last.weight),
            "region_residual_last_bias_rms": tensor_rms(region_last.bias),
        },
        "identity": {
            "max_abs_diff": float(identity.max().detach().cpu()),
            "mean_abs_diff": float(identity.mean().detach().cpu()),
            "residual_abs_max": float(out["e15_residual_full"].abs().max().detach().cpu()),
        },
        "warmup_required_update_audit": required_updates,
        "audits": [warmup_audit, decoder_audit],
    }
    if (
        report["identity"]["max_abs_diff"] >= 1e-6
        or not report["zero_initialization"]["all_betas_zero"]
        or report["zero_initialization"]["region_residual_last_weight_rms"] != 0.0
        or report["zero_initialization"]["region_residual_last_bias_rms"] != 0.0
        or not all(item["pass"] for item in required_updates.values())
        or report["hrnet_imagenet_pretrained"]["matched_keys"] <= 0
        or bad_missing
        or init_report["unexpected"]
        or init_report["skipped_shape"]
    ):
        report["status"] = "FAIL"
    text = json.dumps(report, indent=2, sort_keys=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
