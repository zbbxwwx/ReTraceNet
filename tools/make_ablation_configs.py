#!/usr/bin/env python
"""Generate controlled ablation YAMLs from the default configuration."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def main() -> None:
    source = Path("configs/default.yaml")
    with source.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    variants = {
        "a0_baseline": {
            "use_quality_calibration": False,
            "use_content_orthogonalization": False,
            "use_counterfactual_intervention": False,
            "use_evidence_ledger": False,
            "recycle_steps": 0,
            "use_uncertainty_refinement": False,
        },
        "a1_qrc": {
            "use_quality_calibration": True,
            "use_content_orthogonalization": False,
            "use_counterfactual_intervention": False,
            "use_evidence_ledger": False,
            "recycle_steps": 0,
            "use_uncertainty_refinement": False,
        },
        "a2_qrc_icti": {
            "use_quality_calibration": True,
            "use_content_orthogonalization": True,
            "use_counterfactual_intervention": True,
            "use_evidence_ledger": False,
            "recycle_steps": 0,
            "use_uncertainty_refinement": False,
        },
        "a3_qrc_icti_dhel": {
            "use_quality_calibration": True,
            "use_content_orthogonalization": True,
            "use_counterfactual_intervention": True,
            "use_evidence_ledger": True,
            "recycle_steps": 2,
            "use_uncertainty_refinement": False,
        },
        "a4_full": {
            "use_quality_calibration": True,
            "use_content_orthogonalization": True,
            "use_counterfactual_intervention": True,
            "use_evidence_ledger": True,
            "recycle_steps": 2,
            "use_uncertainty_refinement": True,
        },
    }
    output = Path("configs/ablations")
    output.mkdir(parents=True, exist_ok=True)
    for name, changes in variants.items():
        config = copy.deepcopy(base)
        config["experiment"] = name
        config["output_dir"] = f"runs/ablations/{name}"
        config["model"].update(changes)
        if not changes["use_content_orthogonalization"]:
            config["loss"]["weights"]["orthogonality"] = 0.0
        if not changes["use_counterfactual_intervention"]:
            config["loss"]["weights"]["proposal"] = 0.0
            config["loss"]["weights"]["counterfactual"] = 0.0
            config["loss"]["weights"]["intervention_locality"] = 0.0
        if not changes["use_evidence_ledger"]:
            config["loss"]["weights"]["ledger"] = 0.0
        if not changes["use_uncertainty_refinement"]:
            config["loss"]["weights"]["variance_regularization"] = 0.0
        path = output / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
