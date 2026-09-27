"""Aggregate independently selected Columbia seed results as mean +/- sample std."""
import argparse
from pathlib import Path
import math
import statistics
import yaml


FIELDS = {
    "fake_only_f1": ("fake_only_pooled", "test_f1_locked", "f1"),
    "fake_only_precision": ("fake_only_pooled", "test_f1_locked", "precision"),
    "fake_only_recall": ("fake_only_pooled", "test_f1_locked", "recall"),
    "fake_only_auc": ("fake_only_pooled", "auc"),
    "authentic_pixel_fpr": ("authentic_pixel_fpr", "fpr"),
    "all_images_f1": ("all_images_pooled", "test_f1_locked", "f1"),
    "all_images_precision": ("all_images_pooled", "test_f1_locked", "precision"),
    "all_images_recall": ("all_images_pooled", "test_f1_locked", "recall"),
    "all_images_auc": ("all_images_pooled", "auc"),
}


def get(d, path):
    for key in path:
        d = d[key]
    return float(d)


def main(args):
    root = Path(args.root)
    seeds = [int(x) for x in args.seeds.split(",")]
    per_seed = []
    for seed in seeds:
        path = root / f"seed_{seed}" / "topk_ensemble_1024_768" / "result.yaml"
        result = yaml.safe_load(path.read_text())
        row = {
            "seed": seed,
            "result_path": str(path),
            "checkpoint_epochs": result["checkpoint_epochs"],
            "threshold": float(result["threshold"]),
        }
        row.update({name: get(result, path_keys) for name, path_keys in FIELDS.items()})
        per_seed.append(row)

    aggregate = {}
    for name in FIELDS:
        values = [r[name] for r in per_seed]
        aggregate[name] = {
            "mean": statistics.mean(values),
            "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    output = {
        "status": "completed",
        "dataset": "Columbia",
        "protocol": {
            "deduplicate_by": "rgb_sha256",
            "checkpoint_interval_epochs": 5,
            "checkpoint_selection": "top-6 saved candidates ranked by val F1 + 0.2*AUC; best 1-3 checkpoint combination selected from ensembled val probabilities",
            "ensemble": "equal probability average selected on val only",
            "threshold_source": "per-seed deduplicated Columbia val",
            "test_runs_per_seed": 1,
            "std": "sample standard deviation (ddof=1)",
            "inference": "1024 tile / 768 stride",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out = root / "three_seed_mean_std.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=False))
    print(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", default="3407,2026,42")
    main(p.parse_args())
