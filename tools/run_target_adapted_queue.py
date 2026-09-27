"""Offline GPU0 queue for four target-adapted V3 experiments."""
import json, os, subprocess, time, traceback
from pathlib import Path
import yaml

BASE = Path("/group4/zouwenxuan/iml_generalization")
REPO = BASE / "repos/ReTraceNet"
PY = BASE / "envs/mesorch/bin/python"
PRETRAINED = BASE / "outputs/retracenet_v2/balanced_residual_v3_full30_20260912/best.pt"
INV = BASE / "outputs/retracenet_v2/final_experiment_summary_20260912/data_inventory"
ROOT = BASE / "outputs/retracenet_v2/v3_target_adapted_stable_20260913"
AUDIT = BASE / "outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/trainability_audit/summary.yaml"
SPECS = [
    ("CASIA1", .617, .821, 20),
    ("Columbia", .741, .687, 20),
    ("NIST16", .553, .772, 40),
    ("IMD2020", .619, .885, 40),
]


def save(value):
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = ROOT / "queue_summary.yaml.tmp"
    temp.write_text(yaml.safe_dump(value, sort_keys=False))
    os.replace(temp, ROOT / "queue_summary.yaml")


def main():
    summary = {"status": "waiting_for_trainability_audit", "protocol": "target-adapted V3",
               "pretrained": str(PRETRAINED), "datasets": {}}
    save(summary)
    while True:
        try:
            audit = yaml.safe_load(AUDIT.read_text()) or {}
        except FileNotFoundError:
            audit = {}
        if audit.get("status") in {"completed", "failed"}:
            break
        time.sleep(30)
    summary["trainability_audit_status"] = audit.get("status")
    summary["status"] = "running"
    save(summary)
    for name, target_f1, target_auc, max_epochs in SPECS:
        out = ROOT / name
        existing = {}
        try:
            existing = yaml.safe_load((out / "summary.yaml").read_text()) or {}
        except FileNotFoundError:
            pass
        if existing.get("status") == "completed":
            summary["datasets"][name] = existing.get("benchmark_comparison", {"status": "completed"})
            save(summary)
            continue
        cmd = [str(PY), "-u", str(REPO / "tools/run_v3_target_adapted.py"),
               "--dataset", name, "--inventory", str(INV / f"{name}.jsonl"),
               "--pretrained", str(PRETRAINED), "--out", str(out),
               "--epochs", str(max_epochs), "--patience", "10", "--base-lr", "0.0001",
               "--batch-size", "4", "--dct-dropout", "0.2", "--seed", "3407",
               "--target-f1", str(target_f1), "--target-auc", str(target_auc), "--resume"]
        cmd += ["--stable-epochs", "3"]
        summary["current_dataset"] = name
        save(summary)
        log = out / "run.log"; out.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            result = subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("TARGET_GPU", "1")})
        try:
            result_doc = yaml.safe_load((out / "summary.yaml").read_text()) or {}
        except FileNotFoundError:
            result_doc = {}
        summary["datasets"][name] = {
            "returncode": result.returncode, "status": result_doc.get("status"),
            "benchmark_comparison": result_doc.get("benchmark_comparison"),
            "best_F1": result_doc.get("best_F1"), "best_epoch": result_doc.get("best_epoch")}
        save(summary)
        if result.returncode != 0:
            # Preserve the failure and continue to the remaining independent dataset.
            continue
    summary.pop("current_dataset", None)
    summary["status"] = "completed_with_failures" if any(
        value.get("returncode", 0) != 0 for value in summary["datasets"].values()) else "completed"
    save(summary)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        ROOT.mkdir(parents=True, exist_ok=True)
        save({"status": "failed", "error": traceback.format_exc()})
        raise
