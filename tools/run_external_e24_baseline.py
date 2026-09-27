"""Evaluate E24 on the exact external protocol used for Fair Training best."""
import copy
import traceback
from pathlib import Path

import torch
import yaml

import run_cross_dataset_after_chain as cross

e = cross.e
BASE = Path("/group4/zouwenxuan/iml_generalization")
CHECKPOINT = BASE / "outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt"
INDEX_ROOT = BASE / "paper_compare/mvss_five_indices"
OUT = BASE / "outputs/retracenet_v2/external_four_e24_same_protocol_20260912"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"status": "preparing", "test_run": True, "checkpoint": str(CHECKPOINT),
               "selection_policy": "E24 checkpoint and threshold selected only on CASIA2 validation",
               "comparison_protocol": "identical evaluator/input/tiling/masks to Fair-best external run",
               "datasets": {}}
    prior_path = OUT / "summary.yaml"
    if prior_path.exists():
        prior = yaml.safe_load(prior_path.read_text()) or {}
        summary["datasets"] = prior.get("datasets", {})
    cross.save_yaml(OUT / "summary.yaml", summary)
    try:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        cfg = copy.deepcopy(checkpoint["config"])
        model = e.build_final_model(cfg["model"]).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        threshold = float(checkpoint["selection_threshold"])
        summary.update(status="evaluating", casia2_selection_F1=float(checkpoint["best_metric"]),
                       locked_threshold=threshold)
        cross.save_yaml(OUT / "summary.yaml", summary)
        mapping = {"CASIA1": INDEX_ROOT / "CASIA1.txt", "Columbia": INDEX_ROOT / "Columbia.txt",
                   "NIST16": INDEX_ROOT / "NIST16.txt", "IMD2020": INDEX_ROOT / "IMD2020.txt"}
        for name, index_path in mapping.items():
            if name in summary["datasets"]:
                print(f"[resume] skip completed dataset {name}", flush=True)
                continue
            records = cross.parse_index(index_path, name)
            result = cross.evaluate_dataset(model, records, OUT / "jpeg_cache" / name,
                                            threshold, name, OUT)
            summary["datasets"][name] = {
                "num_images": result["num_images"], "locked": result["locked"],
                "auc": result["diagnostic_only"]["fake"]["auc"],
                "fixed_0p5": result["diagnostic_only"]["fake"]["fixed"],
                "oracle_diagnostic": result["diagnostic_only"]["fake"]["oracle"],
            }
            cross.save_yaml(OUT / "summary.yaml", summary)
        summary.update(status="completed", stop_reason="E24 four-dataset same-protocol baseline complete")
        cross.save_yaml(OUT / "summary.yaml", summary)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        cross.save_yaml(OUT / "summary.yaml", summary)
        raise


if __name__ == "__main__":
    main()
