"""Run the single approved E15-HR A->B route and write one review summary."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python")
OUT = Path("/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e15_hr")
WARMUP_CONFIG = ROOT / "configs/casia2_retracenet_e15_hr_warmup.yaml"
DECODER_CONFIG = ROOT / "configs/casia2_retracenet_e15_hr_decoder_adapt.yaml"
SUMMARY = OUT / "overnight_summary.yaml"
PREFLIGHT = OUT / "preflight_warmup_full.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_summary(summary: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(SUMMARY)


def archive_phase_dir(path: Path, stamp: str) -> str | None:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return None
    if path.parent.resolve() != OUT.resolve():
        raise RuntimeError(f"refusing to archive directory outside E15-HR output root: {path}")
    archived = OUT / f"{path.name}_archived_{stamp}"
    if archived.exists():
        raise RuntimeError(f"archive target already exists: {archived}")
    shutil.move(str(path), str(archived))
    path.mkdir(parents=True, exist_ok=True)
    return str(archived)


def convert(value: str):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_metrics(outdir: Path) -> list[dict]:
    path = outdir / "history.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {key: convert(value) for key, value in raw.items() if key != "extra"}
            try:
                row["extra"] = json.loads(raw.get("extra") or "{}")
            except json.JSONDecodeError:
                row["extra"] = {"unparsed": raw.get("extra")}
            rows.append(row)
    return rows


def phase_record(config: dict, outdir: Path) -> dict:
    rows = read_metrics(outdir)
    status_path = outdir / "training_status.yaml"
    status = load_yaml(status_path) if status_path.exists() else {}
    best_row = max(rows, key=lambda row: float(row["selection_value"])) if rows else None
    return {
        "phase": status.get("phase"),
        "selection_metric": config["training"].get("selection_metric"),
        "configured_epochs": config["training"].get("epochs"),
        "completed_epochs": status.get("completed_epochs", len(rows)),
        "stop_reason": status.get("stop_reason", "training_process_failed_before_status"),
        "best_selection_value": status.get("best_selection_value"),
        "best_epoch": best_row.get("epoch") if best_row else None,
        "best_checkpoint": status.get("best_checkpoint", str(outdir / "best.pt")),
        "base_learning_rate": config["training"].get("learning_rate"),
        "parameter_group_lrs": config["training"].get("parameter_groups", []),
        "epoch_metrics": rows,
        "log": str(outdir / "train.log"),
    }


def run_phase(config_path: Path, phase: str, outdir: Path) -> int:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    command = [str(PYTHON), "train_v2_final_pipeline.py", "--config", str(config_path), "--phase", phase]
    log_path = outdir / "train.log"
    print(f"[overnight] starting {phase}; log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def main() -> int:
    warmup_cfg = load_yaml(WARMUP_CONFIG)
    decoder_cfg = load_yaml(DECODER_CONFIG)
    for name, cfg in [("warmup", warmup_cfg), ("decoder_adapt", decoder_cfg)]:
        if cfg["training"].get("selection_metric") != "val_tampered_pooled_f1_best":
            raise RuntimeError(f"{name} selection metric is not the E5-compatible tampered pooled F1-best")
        patterns = [p for group in cfg["training"].get("parameter_groups", []) for p in group.get("patterns", [])]
        if any("forensic_stream" in pattern for pattern in patterns):
            raise RuntimeError(f"{name} optimizer configuration includes forensic_stream")
    if int(warmup_cfg["training"].get("epochs", 0)) != 2:
        raise RuntimeError("Phase A must be exactly 2 epochs")
    if int(decoder_cfg["training"].get("epochs", 0)) > 6:
        raise RuntimeError("Phase B exceeds the six-epoch cap")
    if float(decoder_cfg["training"].get("selection_floor", -1)) != 0.785:
        raise RuntimeError("Phase B F1 floor must be 0.785")
    expected_init = OUT / "warmup/best.pt"
    if Path(decoder_cfg["training"].get("init_checkpoint", "")) != expected_init:
        raise RuntimeError(f"Phase B must initialize from {expected_init}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    warmup_dir = OUT / "warmup"
    decoder_dir = OUT / "decoder_adapt"
    archived = {
        "warmup": archive_phase_dir(warmup_dir, stamp),
        "decoder_adapt": archive_phase_dir(decoder_dir, stamp),
    }
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("full E15-HR preflight did not pass")
    for audit in preflight.get("audits", []):
        forensic_trainable = int(audit.get("trainable_by_prefix", {}).get("forensic_stream", 0))
        if forensic_trainable:
            raise RuntimeError(f"forensic_stream is trainable in {audit.get('phase')}: {forensic_trainable} parameters")

    summary = {
        "route": "E15-HR only",
        "status": "running_phase_a",
        "started_at": now(),
        "constraints": {
            "new_structure_variants": False,
            "forensic_stream_frozen_all_phases": True,
            "selection_metric": "val_tampered_pooled_f1_best",
            "auc_used_for_gate": False,
            "automatic_test": False,
            "phase_a_epochs": 2,
            "phase_b_max_epochs": 6,
            "phase_b_f1_floor": 0.785,
            "phase_b_floor_grace_epochs": 2,
            "phase_b_no_improvement_patience": decoder_cfg["training"].get("early_stopping"),
        },
        "preflight": {
            "path": str(PREFLIGHT),
            "status": preflight.get("status"),
            "identity_max_abs_diff": preflight.get("identity", {}).get("max_abs_diff"),
        },
        "archived_previous_partial_runs": archived,
        "phase_plan": {
            "phase_a": {
                "config": str(WARMUP_CONFIG),
                "output_dir": str(warmup_dir),
                "best_checkpoint": str(expected_init),
                "base_learning_rate": warmup_cfg["training"].get("learning_rate"),
                "parameter_group_lrs": warmup_cfg["training"].get("parameter_groups", []),
            },
            "phase_b": {
                "config": str(DECODER_CONFIG),
                "output_dir": str(decoder_dir),
                "init_checkpoint": str(expected_init),
                "best_checkpoint": str(decoder_dir / "best.pt"),
                "base_learning_rate": decoder_cfg["training"].get("learning_rate"),
                "parameter_group_lrs": decoder_cfg["training"].get("parameter_groups", []),
            },
        },
        "phases": {},
    }
    save_summary(summary)

    rc_a = run_phase(WARMUP_CONFIG, "e15_hr_warmup", warmup_dir)
    summary["phases"]["phase_a"] = phase_record(warmup_cfg, warmup_dir)
    if rc_a != 0 or not expected_init.exists():
        summary["status"] = "failed_phase_a"
        summary["stop_reason"] = f"phase_a_exit_code_{rc_a}; warmup_best_exists={expected_init.exists()}"
        summary["finished_at"] = now()
        save_summary(summary)
        return rc_a or 1
    summary["status"] = "running_phase_b"
    summary["phase_b_init_checkpoint"] = str(expected_init)
    save_summary(summary)

    rc_b = run_phase(DECODER_CONFIG, "e15_hr_decoder_adapt", decoder_dir)
    summary["phases"]["phase_b"] = phase_record(decoder_cfg, decoder_dir)
    summary["status"] = "complete" if rc_b == 0 else "failed_phase_b"
    summary["stop_reason"] = summary["phases"]["phase_b"]["stop_reason"]
    summary["final_best_checkpoint"] = summary["phases"]["phase_b"]["best_checkpoint"]
    summary["finished_at"] = now()
    save_summary(summary)
    print(f"[overnight] finished status={summary['status']} summary={SUMMARY}", flush=True)
    return rc_b


if __name__ == "__main__":
    sys.exit(main())
