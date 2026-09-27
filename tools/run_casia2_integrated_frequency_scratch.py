"""Train the two-stream ReTraceNet with an integrated local-frequency forensic front end.

This entry point deliberately removes the CAT-Net DCT encoder and its pretrained
weights.  The forensic stream is a single residual encoder whose input is
augmented by fixed, compact local-DCT statistics.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

import run_imd_mit_full_v3 as mit
from retracenet.integrated_frequency_forensic import (
    IntegratedFrequencyForensicStream,
)


stable = mit.stable
_mit_builder = stable.v3.e.build_final_model


def build_integrated_frequency_model(config: dict):
    cfg = copy.deepcopy(config)

    # Keep exactly one forensic stream.  Do not construct or load CAT-Net.
    cfg["forensic_stream_type"] = "residual"
    cfg["enable_dct_parallel"] = False
    cfg["dct_pretrained_path"] = None
    cfg["enable_dct_gate"] = False
    cfg["enable_dct_evidence_head"] = False
    cfg["integrated_frequency_frontend"] = True

    # Persist the real architecture in checkpoints.  The training driver keeps
    # the same config object and serializes it after each epoch.
    config.clear()
    config.update(copy.deepcopy(cfg))

    model = _mit_builder(cfg)
    if getattr(model, "dct_branch", None) is not None:
        raise RuntimeError("CAT-Net DCT branch was unexpectedly constructed")

    model.forensic_stream = IntegratedFrequencyForensicStream(
        model.forensic_stream,
        residual_kernels=int(cfg.get("residual_kernels", 8)),
    )
    model.integrated_frequency_no_catnet = True
    return model


stable.v3.e.build_final_model = build_integrated_frequency_model


def smoke_test(template: Path) -> None:
    checkpoint = torch.load(template, map_location="cpu")
    cfg = copy.deepcopy(checkpoint["config"])
    model = build_integrated_frequency_model(cfg["model"]).cuda().train()

    image = torch.rand(2, 3, 256, 256, device="cuda")
    output = model(image)
    loss = output["mask_logits"].float().square().mean()
    loss.backward()

    freq_grad = 0.0
    residual_grad = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm())
        if "forensic_stream.freq_" in name:
            freq_grad += value
        if "forensic_stream.base.residual_bank" in name:
            residual_grad += value

    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke-test loss")
    if freq_grad <= 0.0:
        raise RuntimeError("integrated frequency front end received no gradient")
    if residual_grad <= 0.0:
        raise RuntimeError("residual bank received no gradient")

    forbidden = [
        name
        for name, _ in model.named_parameters()
        if "dct_branch" in name or "cat_dct" in name
    ]
    if forbidden:
        raise RuntimeError(f"unexpected CAT-Net parameters: {forbidden[:5]}")

    print(
        {
            "status": "smoke_passed",
            "mask_shape": tuple(output["mask_logits"].shape),
            "loss": float(loss.detach()),
            "frequency_grad_norm_sum": freq_grad,
            "residual_grad_norm_sum": residual_grad,
            "catnet_parameter_count": 0,
        },
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CASIA2")
    parser.add_argument("--split", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_only:
        smoke_test(Path(args.template))
        return

    # Preserve the established CASIA2 protocol and objective.  Only the
    # forensic input representation changes.
    for key, value in {
        "positive_crop_prob": 0.0,
        "positive_crop_retries": 20,
        "hcv_aux_scale": 1.0,
        "qtr_aux_scale": 0.15,
        "dher_aux_scale": 0.05,
        "forensic_mode": "full",
        "hcv_semantic_bypass": False,
        "verified_residual_decoder": False,
        "decoder_ds_targeted": False,
        "final_mask_only": False,
        "splicing_region_affinity": False,
        "region_affinity_weight": 0.1,
        "early_gate_epoch": 0,
        "early_gate_min_f1": 0.5,
    }.items():
        setattr(args, key, value)

    stable.main(args)


if __name__ == "__main__":
    main()
