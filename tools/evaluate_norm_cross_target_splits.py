import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

import run_imd_mit_full_v3_imagenet_norm  # noqa: F401; normalized MiT and 1024/768 patch
import run_cross_dataset_after_chain as cross
from retrace.data import Record


SPLITS = {
    "CASIA1": "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json",
    "IMD2020": "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json",
    "Columbia": "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json",
}


def add_all_pooled(result):
    fake = result["pixel_localization_fake_only"]["test_f1_locked"]
    authentic = result["authentic_pixel_fpr"]
    tp, fn = int(fake["tp"]), int(fake["fn"])
    fp = int(fake["fp"]) + int(authentic["false_positive_pixels"])
    tn = int(fake["tn"]) + int(authentic["authentic_pixels"]) - int(authentic["false_positive_pixels"])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    all_diag = result.get("diagnostic_only", {}).get("all", {})
    result["pixel_localization_all_images_pooled"] = {
        "test_f1_locked": {
            "threshold": result["pixel_threshold"], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "iou": tp / max(1, tp + fp + fn), "fpr": fp / max(1, fp + tn),
        },
        "auc": all_diag.get("auc"),
        "test_f1_oracle_diagnostic": all_diag.get("oracle"),
        "population": "all valid pixels from fake and authentic test images",
        "threshold_source": "FantasticReality_val",
    }
    return result


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = cross.e.build_final_model(copy.deepcopy(checkpoint["config"]["model"])).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    threshold = float(checkpoint["selection_threshold"])
    image_threshold = float(checkpoint.get("image_selection_threshold", 0.5))
    summary = {"status": "running", "checkpoint_epoch": int(checkpoint["epoch"]),
               "threshold": threshold, "threshold_source": "FantasticReality_val", "datasets": {}}
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    for dataset, split_path in SPLITS.items():
        rows = json.loads(Path(split_path).read_text())["test"]
        records = [Record(image=Path(r["image"]), mask=Path(r["mask"]) if r.get("mask") else None,
                          split="test", dataset=dataset, label=int(r["label"])) for r in rows]
        result = cross.evaluate_dataset(model, records, out / "jpeg_cache" / dataset,
                                        threshold, image_threshold, dataset, out,
                                        evaluation_mode="cross_domain_locked",
                                        threshold_source="FantasticReality_val")
        result["image_threshold_source"] = "FantasticReality_val"
        result["checkpoint"] = args.checkpoint
        result["checkpoint_epoch"] = int(checkpoint["epoch"])
        result = add_all_pooled(result)
        cross.save_yaml(out / f"{dataset}.yaml", result)
        summary["datasets"][dataset] = result
        (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    summary["status"] = "completed"
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
