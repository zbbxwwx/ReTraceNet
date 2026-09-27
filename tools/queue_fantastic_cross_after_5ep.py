import subprocess
import sys
import time
from pathlib import Path

import yaml

TRAIN = Path("/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/fantastic_mitb2_norm_ctx1024_20ep_20260919")
OUT = Path("/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/fantastic_mitb2_norm_ctx1024_cross_after5_20260919")
WRAPPER = Path("/group4/zouwenxuan/iml_generalization/repos/ReTraceNet/tools/run_norm_mit_external.py")


def load_summary():
    path = TRAIN / "summary.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def enrich_all_images_pooled(path):
    """Add locked-threshold pooled pixel metrics over fake and authentic images."""
    path = Path(path)
    if not path.exists():
        return None
    result = yaml.safe_load(path.read_text()) or {}
    fake = result["pixel_localization_fake_only"]["test_f1_locked"]
    authentic = result["authentic_pixel_fpr"]
    tp = int(fake["tp"])
    fn = int(fake["fn"])
    fp = int(fake["fp"]) + int(authentic["false_positive_pixels"])
    authentic_tn = int(authentic["authentic_pixels"]) - int(authentic["false_positive_pixels"])
    tn = int(fake["tn"]) + authentic_tn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    fpr = fp / max(1, fp + tn)
    all_diag = result.get("diagnostic_only", {}).get("all", {})
    result["pixel_localization_all_images_pooled"] = {
        "test_f1_locked": {
            "threshold": float(result["pixel_threshold"]),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "iou": iou, "fpr": fpr,
        },
        "auc": all_diag.get("auc"),
        "test_f1_oracle_diagnostic": all_diag.get("oracle"),
        "population": "all valid pixels from fake and authentic images",
        "threshold_source": result.get("threshold_source"),
    }
    path.write_text(yaml.safe_dump(result, sort_keys=False))
    return result


while True:
    summary = load_summary()
    status = summary.get("status")
    if status == "completed":
        break
    if status == "failed":
        raise SystemExit(f"FantasticReality training/test failed: {summary.get('stop_reason')}")
    time.sleep(60)

checkpoint = Path(summary.get("best_checkpoint", TRAIN / "best.pt"))
if not checkpoint.exists():
    raise SystemExit(f"missing best checkpoint: {checkpoint}")

OUT.mkdir(parents=True, exist_ok=True)
fantastic_result = TRAIN / "test_once" / "FantasticReality.yaml"
fantastic = enrich_all_images_pooled(fantastic_result)
cmd = [
    sys.executable,
    str(WRAPPER),
    "--checkpoint", str(checkpoint),
    "--out", str(OUT),
    "--threshold-source", "FantasticReality_val",
    "--datasets", "CASIA1", "IMD2020", "Columbia",
]
with (OUT / "run.log").open("a") as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

combined = {"FantasticReality": fantastic}
for dataset in ("CASIA1", "IMD2020", "Columbia"):
    combined[dataset] = enrich_all_images_pooled(OUT / f"{dataset}.yaml")
(OUT / "metrics_with_fake_and_all_pooled.yaml").write_text(
    yaml.safe_dump(combined, sort_keys=False)
)
