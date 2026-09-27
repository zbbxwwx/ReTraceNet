"""One-batch, non-persistent forward/gradient wiring audit for E15-HR Phase B."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--out")
    return parser.parse_args()


def rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().cpu())


def mean_rms(tensor: torch.Tensor) -> dict[str, float]:
    return {"mean": float(tensor.detach().float().mean().cpu()), "rms": rms(tensor)}


def dher_group(name: str) -> str:
    if name in {"alpha_support", "alpha_conflict"}:
        return name
    return name.split(".", 1)[0]


def grad_norms(module: torch.nn.Module) -> tuple[dict[str, dict], dict[str, float | None]]:
    grouped: dict[str, dict] = {}
    exact: dict[str, float | None] = {}
    critical = {
        "alpha_support",
        "alpha_conflict",
        "support_fuse.4.weight",
        "support_fuse.4.bias",
        "reexam.4.weight",
        "reexam.4.bias",
    }
    for name, param in module.named_parameters():
        grad = param.grad
        grad_norm = float(grad.detach().float().norm().cpu()) if grad is not None else None
        if name in critical:
            exact[name] = grad_norm
        group = dher_group(name)
        item = grouped.setdefault(group, {"parameters": 0, "with_grad": 0, "nonzero_grad": 0, "grad_l2": 0.0})
        item["parameters"] += 1
        if grad is not None:
            item["with_grad"] += 1
            item["grad_l2"] = math.sqrt(item["grad_l2"] ** 2 + grad_norm ** 2)
            if grad_norm > 0:
                item["nonzero_grad"] += 1
    return grouped, exact


def parameter_deltas(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> tuple[dict[str, dict], dict[str, float]]:
    grouped: dict[str, dict] = {}
    exact: dict[str, float] = {}
    critical = {
        "alpha_support",
        "alpha_conflict",
        "support_fuse.4.weight",
        "support_fuse.4.bias",
        "reexam.4.weight",
        "reexam.4.bias",
    }
    for name, param in module.named_parameters():
        delta = param.detach() - before[name]
        value = rms(delta)
        if name in critical:
            exact[name] = value
        group = dher_group(name)
        item = grouped.setdefault(group, {"parameters": 0, "changed": 0, "delta_rms_l2": 0.0})
        item["parameters"] += 1
        item["delta_rms_l2"] = math.sqrt(item["delta_rms_l2"] ** 2 + value ** 2)
        if value > 0:
            item["changed"] += 1
    return grouped, exact


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 3407)))
    device = choose_device(args.device)
    model = build_final_model(cfg["model"]).to(device)
    init = load_initial_weights_repaired(
        model,
        args.checkpoint,
        device,
        cfg["training"].get("init_load_patterns") or [],
        cfg["training"].get("reinitialize_prefixes") or [],
        cfg["training"].get("reinitialize_keys") or [],
        cfg["training"].get("init_key_remaps") or [],
    )
    bad_missing = list(init["missing"])
    if bad_missing or init["unexpected"] or init["skipped_shape"]:
        raise RuntimeError(
            f"checkpoint load is not exact: missing={len(bad_missing)} "
            f"unexpected={len(init['unexpected'])} skipped={len(init['skipped_shape'])}"
        )
    trainable = set_trainable(model, "e15_hr_decoder_adapt")
    if any(name.startswith("forensic_stream.") for name in trainable):
        raise RuntimeError("forensic_stream unexpectedly trainable")
    optimizer = build_optimizer(model, cfg, "e15_hr_decoder_adapt")
    criterion = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).to(device)
    loader = build_loader({**cfg["data"], "batch_size": 1, "num_workers": 0}, "train", shuffle=False)
    batch = next(iter(loader))

    call_counter = {"dher": 0}
    hook = model.dher.register_forward_hook(lambda *_: call_counter.__setitem__("dher", call_counter["dher"] + 1))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = forward_from_batch(model, batch, device)
    hook.remove()

    d1_pre = out["e15_d1_pre_feature"]
    d1_refined = out["e15_d1_refined_feature"]
    highres_input = out["e15_highres"]["d1_input"]
    dher = out.get("dher")
    if dher is None:
        raise RuntimeError("E15 forward returned no DHER diagnostics")
    hcv = out.get("hcv2")
    if hcv is None:
        raise RuntimeError("E15 forward returned no HCV outputs")

    before = {name: param.detach().clone() for name, param in model.dher.named_parameters()}
    alpha_before = {
        "alpha_support": float(model.dher.alpha_support.detach().cpu()),
        "alpha_conflict": float(model.dher.alpha_conflict.detach().cpu()),
    }
    losses = e15_hr_loss(
        out,
        batch["mask"].to(device),
        batch["label"].to(device),
        criterion,
        cfg,
    )
    losses["total"].backward()
    grouped_grad, exact_grad = grad_norms(model.dher)
    beta_before = [float(adapter.beta.detach().cpu()) for adapter in model.e15_decoder.evidence_adapters]
    beta_grad = [
        None if adapter.beta.grad is None else float(adapter.beta.grad.detach().float().cpu())
        for adapter in model.e15_decoder.evidence_adapters
    ]
    optimizer.step()
    grouped_delta, exact_delta = parameter_deltas(model.dher, before)
    beta_after = [float(adapter.beta.detach().cpu()) for adapter in model.e15_decoder.evidence_adapters]
    alpha_after = {
        "alpha_support": float(model.dher.alpha_support.detach().cpu()),
        "alpha_conflict": float(model.dher.alpha_conflict.detach().cpu()),
    }

    d1_change = d1_refined - d1_pre
    old_rms = rms(out["e15_old_logits"])
    residual_rms = rms(out["e15_residual_full"])
    same_storage = highres_input.data_ptr() == d1_refined.data_ptr()
    highres_diff = float((highres_input - d1_refined).detach().abs().max().cpu())
    report = {
        "status": "PASS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_load": {
            "loaded": init["loaded"],
            "missing": len(init["missing"]),
            "unexpected": len(init["unexpected"]),
            "skipped_shape": len(init["skipped_shape"]),
        },
        "forensic_stream_trainable": False,
        "forward": {
            "D1_pre_rms": rms(d1_pre),
            "dher_forward_call_count": call_counter["dher"],
            "dher_executed_for_e15": call_counter["dher"] >= 2 and dher is not None,
            "note_on_call_count": "E15 executes shared DHER once for the E5 anchor path and once for the HR path.",
            "D1_refined_rms": rms(d1_refined),
            "D1_refined_minus_pre_rms": rms(d1_change),
            "alpha_support": alpha_before["alpha_support"],
            "alpha_conflict": alpha_before["alpha_conflict"],
            "support_writer_rms": rms(dher["support_write"]),
            "conflict_writer_rms": rms(dher["conflict_write"]),
            "support_maps": [mean_rms(value) for value in dher["support"]],
            "refutation_maps": [mean_rms(value) for value in dher["refutation"]],
            "conflict_maps": [mean_rms(value) for value in dher["conflict"]],
            "hcv_proposals_P1_P3": [mean_rms(value) for value in hcv["proposals"]],
            "hcv_delta1_delta3": [mean_rms(value) for value in hcv["deltas"]],
            "evidence_adapter_beta_before_step": beta_before,
            "R_full_rms": residual_rms,
            "old_logit_rms": old_rms,
            "R_full_rms_over_old_logit_rms": residual_rms / max(old_rms, 1e-12),
        },
        "backward": {
            "loss_total": float(losses["total"].detach().cpu()),
            "dher_grad_by_group": grouped_grad,
            "dher_critical_parameter_grad_norm": exact_grad,
            "evidence_adapter_beta_grad": beta_grad,
        },
        "optimizer_step": {
            "dher_delta_by_group": grouped_delta,
            "dher_critical_parameter_delta_rms": exact_delta,
            "alpha_after_step": alpha_after,
            "evidence_adapter_beta_after_step": beta_after,
            "evidence_adapter_beta_delta": [after - before for before, after in zip(beta_before, beta_after)],
        },
        "highres_input_wiring": {
            "uses_D1_refined": same_storage and highres_diff == 0.0,
            "same_tensor_storage": same_storage,
            "max_abs_diff_highres_input_vs_D1_refined": highres_diff,
            "max_abs_diff_highres_input_vs_D1_pre": float((highres_input - d1_pre).detach().abs().max().cpu()),
        },
    }
    if (
        not report["forward"]["dher_executed_for_e15"]
        or report["forward"]["D1_refined_minus_pre_rms"] == 0.0
        or report["forward"]["support_writer_rms"] == 0.0
        or report["forward"]["conflict_writer_rms"] == 0.0
        or not any(abs(value or 0.0) > 0 for value in beta_grad)
        or not any(item["nonzero_grad"] > 0 for item in grouped_grad.values())
        or not any(item["changed"] > 0 for item in grouped_delta.values())
        or not report["highres_input_wiring"]["uses_D1_refined"]
    ):
        report["status"] = "FAIL"
    text = yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
    print(text, flush=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
