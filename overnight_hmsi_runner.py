#!/usr/bin/env python
"""Independent overnight runner for ReTraceNet HMSI experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path("/group4/zouwenxuan/iml_generalization")
REPO = ROOT / "repos/ReTraceNet"
PYTHON = ROOT / "envs/mesorch/bin/python"
LOG_DIR = ROOT / "logs/domain_casia2"
OUTPUT_ROOT = ROOT / "outputs/retracenet"
PARENT_CKPT = OUTPUT_ROOT / "resnet50_poscrop70_stage3_featurefusion_adapter_lr5e5/best.pt"
PARENT_VAL = {"fixed": 0.748661, "oracle": 0.749622, "auc": 0.924916}
PARENT_TEST = {"fixed": 0.702694, "oracle": 0.703397, "auc": 0.907485}


@dataclass
class RunSpec:
    name: str
    key: str
    config: Path
    init_checkpoint: Path
    new_epochs: int
    gpu: int = 0
    continue_if_promising: bool = False
    continuation_new_epochs: int = 15
    output_dir: Path | None = None
    runtime_config: Path | None = None
    log_path: Path | None = None
    status: str = "PENDING"
    decision: str = "PENDING"
    best: dict[str, Any] = field(default_factory=dict)
    history_rows: list[dict[str, str]] = field(default_factory=list)
    checkpoint: Path | None = None
    error: str = ""
    pid: int | None = None
    started_at: str = ""
    finished_at: str = ""
    test_output: Path | None = None
    robustness_output: Path | None = None


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def gpu_idle(gpu: int, memory_limit_mb: int = 512) -> bool:
    query = run_cmd(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
    ).stdout.strip()
    gpu_query = run_cmd(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=False,
    ).stdout.strip()
    for line in gpu_query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(gpu):
            try:
                if int(parts[1]) > memory_limit_mb:
                    return False
            except ValueError:
                return False
    if not query:
        return True
    # If nvidia-smi reports compute apps, conservatively map via CUDA_VISIBLE_DEVICES
    # by also requiring the selected GPU memory to be below the idle threshold.
    return True


def wait_for_gpu(gpu: int, manifest: Path) -> None:
    while not gpu_idle(gpu):
        append(manifest, f"[{now()}] GPU{gpu} busy; waiting")
        time.sleep(300)


def checkpoint_epoch(path: Path) -> int:
    state = torch.load(path, map_location="cpu")
    return int(state["epoch"])


def make_runtime_config(spec: RunSpec, config_dir: Path, stamp: str, resume_epoch: int, reset_optimizer: bool) -> Path:
    config = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    spec.output_dir = OUTPUT_ROOT / f"hmsi_overnight_{stamp}" / spec.key
    if spec.output_dir.exists() and any(spec.output_dir.iterdir()):
        suffix = datetime.now(timezone.utc).strftime("%H%M%S")
        spec.output_dir = OUTPUT_ROOT / f"hmsi_overnight_{stamp}_{suffix}" / spec.key
    config["experiment"] = f"hmsi_overnight_{stamp}_{spec.key}"
    config["output_dir"] = str(spec.output_dir)
    config["training"]["epochs"] = int(resume_epoch + 1 + spec.new_epochs)
    config["training"]["reset_optimizer_on_resume"] = bool(reset_optimizer)
    config["training"]["strict_resume"] = False
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{spec.key}.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    spec.runtime_config = out
    return out


def make_continuation_config(spec: RunSpec, config_dir: Path, resume_epoch: int) -> Path:
    config = yaml.safe_load(spec.runtime_config.read_text(encoding="utf-8"))
    config["training"]["epochs"] = int(resume_epoch + 1 + spec.continuation_new_epochs)
    config["training"]["reset_optimizer_on_resume"] = False
    out = config_dir / f"{spec.key}_continue.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    spec.runtime_config = out
    return out


def parse_history(output_dir: Path) -> list[dict[str, str]]:
    history = output_dir / "history.csv"
    if not history.exists():
        return []
    with history.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_row(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}

    def value(row: dict[str, str], key: str, fallback: float = -1.0) -> float:
        try:
            return float(row.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    row = max(rows, key=lambda item: value(item, "val_f1_oracle", value(item, "selection_value")))
    return {
        "epoch": int(float(row["epoch"])),
        "fixed": value(row, "val_f1"),
        "oracle": value(row, "val_f1_oracle"),
        "auc": value(row, "val_auc"),
        "threshold": value(row, "oracle_threshold"),
        "row": row,
    }


def scalar_from_ckpt(ckpt: Path, names: list[str]) -> dict[str, Any]:
    if not ckpt.exists():
        return {}
    state = torch.load(ckpt, map_location="cpu")["model"]
    result = {}
    for name in names:
        if name in state:
            value = state[name].float()
            result[name] = {
                "mean": float(value.mean()),
                "min": float(value.min()),
                "max": float(value.max()),
            }
    return result


def fatal_log_reason(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")[-500000:].lower()
    patterns = ["cuda out of memory", "traceback", "floatingpointerror", "nan", " inf", "runtimeerror"]
    hits = [pattern for pattern in patterns if pattern in text]
    return ", ".join(hits)


def run_training(spec: RunSpec, manifest: Path, reset_optimizer: bool = True) -> None:
    assert spec.runtime_config is not None
    assert spec.output_dir is not None
    spec.log_path = LOG_DIR / f"hmsi_overnight_{spec.key}.log"
    if spec.log_path.exists():
        spec.log_path = LOG_DIR / f"hmsi_overnight_{spec.key}_{int(time.time())}.log"
    wait_for_gpu(spec.gpu, manifest)
    cmd = [
        str(PYTHON),
        "train.py",
        "--config",
        str(spec.runtime_config),
        "--device",
        f"cuda:{spec.gpu}",
        "--resume",
        str(spec.init_checkpoint),
    ]
    spec.started_at = now()
    append(manifest, f"[{spec.started_at}] START {spec.name} gpu={spec.gpu} config={spec.runtime_config} resume={spec.init_checkpoint} output={spec.output_dir} log={spec.log_path}")
    with spec.log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        spec.pid = proc.pid
        append(manifest, f"[{now()}] PID {spec.name} pid={proc.pid}")
        rc = proc.wait()
    spec.finished_at = now()
    spec.history_rows = parse_history(spec.output_dir)
    spec.best = best_row(spec.history_rows)
    spec.checkpoint = spec.output_dir / "best.pt"
    reason = fatal_log_reason(spec.log_path)
    if rc != 0:
        spec.status = "FAILED"
        spec.error = f"returncode={rc}; {reason}".strip("; ")
        append(manifest, f"[{now()}] FAILED {spec.name}: {spec.error}")
    elif reason:
        spec.status = "SUSPICIOUS"
        spec.error = reason
        append(manifest, f"[{now()}] SUSPICIOUS {spec.name}: {reason}")
    else:
        spec.status = "COMPLETED"
        append(manifest, f"[{now()}] DONE {spec.name}: best={spec.best}")


def decide_run_a(spec: RunSpec) -> str:
    b = spec.best
    if not b:
        return "FAIL"
    oracle, auc = b["oracle"], b["auc"]
    if oracle >= 0.7505:
        return "PASS"
    if auc >= 0.9260 and oracle >= PARENT_VAL["oracle"] - 0.0005:
        return "PROMISING"
    if oracle >= PARENT_VAL["oracle"] - 0.0002:
        return "NEUTRAL"
    if oracle < PARENT_VAL["oracle"] - 0.001 and auc < PARENT_VAL["auc"]:
        return "FAIL"
    return "NEUTRAL"


def decide_run_b(spec: RunSpec, run_a: RunSpec | None) -> str:
    b = spec.best
    if not b:
        return "NO-GO"
    reference = max(PARENT_VAL["oracle"], run_a.best.get("oracle", -1.0) if run_a else -1.0)
    if b["oracle"] >= 0.7510 or b["oracle"] > reference:
        return "PROMISING"
    if b["auc"] >= PARENT_VAL["auc"] and b["oracle"] >= PARENT_VAL["oracle"] - 0.0005:
        return "PROMISING"
    if b["oracle"] < PARENT_VAL["oracle"] - 0.001 and b["auc"] < PARENT_VAL["auc"]:
        return "NO-GO"
    return "NEUTRAL"


def last_five_rising(rows: list[dict[str, str]]) -> bool:
    if len(rows) < 5:
        return False
    vals = []
    for row in rows[-5:]:
        try:
            vals.append(float(row["val_f1_oracle"]))
        except Exception:
            return False
    return vals[-1] > vals[0] and sum(vals[i + 1] >= vals[i] for i in range(4)) >= 2


def decide_run_c(spec: RunSpec) -> str:
    b = spec.best
    if not b:
        return "NO-GO"
    if b["oracle"] >= 0.7510:
        return "BEST"
    if b["auc"] >= 0.9270 and b["oracle"] >= PARENT_VAL["oracle"] - 0.0005:
        return "PROMISING"
    if last_five_rising(spec.history_rows):
        return "PROMISING"
    if b["oracle"] < PARENT_VAL["oracle"] - 0.001 and b["auc"] < PARENT_VAL["auc"]:
        return "NO-GO"
    return "NEUTRAL"


def cbr_allowed(run_c: RunSpec) -> bool:
    b = run_c.best
    return bool(b and b["oracle"] >= PARENT_VAL["oracle"] - 0.0005 and b["auc"] >= PARENT_VAL["auc"])


def launch_eval(spec: RunSpec, manifest: Path, gpu: int = 1, robustness: bool = False) -> subprocess.Popen[str] | None:
    if spec.checkpoint is None or not spec.checkpoint.exists() or spec.runtime_config is None:
        return None
    if not gpu_idle(gpu):
        append(manifest, f"[{now()}] skip eval for {spec.name}: GPU{gpu} busy")
        return None
    kind = "robustness" if robustness else "test"
    output = LOG_DIR / f"hmsi_overnight_{spec.key}_{kind}.json"
    log = LOG_DIR / f"hmsi_overnight_{spec.key}_{kind}.log"
    if robustness:
        cmd = [
            str(PYTHON),
            "evaluate_robustness.py",
            "--config",
            str(spec.runtime_config),
            "--checkpoint",
            str(spec.checkpoint),
            "--split",
            "test",
            "--conditions",
            "clean",
            "jpeg90",
            "jpeg70",
            "scale075",
            "blur1",
            "--output",
            str(output),
            "--device",
            f"cuda:{gpu}",
        ]
        spec.robustness_output = output
    else:
        cmd = [
            str(PYTHON),
            "evaluate.py",
            "--config",
            str(spec.runtime_config),
            "--checkpoint",
            str(spec.checkpoint),
            "--split",
            "test",
            "--output",
            str(output),
            "--device",
            f"cuda:{gpu}",
        ]
        spec.test_output = output
    append(manifest, f"[{now()}] START {kind} eval {spec.name} gpu={gpu} output={output} log={log}")
    log_handle = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log_handle, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return proc


def read_test_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {
        "fixed": metrics.get("pixel_f1_fixed"),
        "oracle": metrics.get("pixel_f1_oracle"),
        "auc": metrics.get("pixel_auc"),
    }


def write_summary(path: Path, runs: list[RunSpec]) -> None:
    lines: list[str] = []
    lines += ["==================================================", "PARENT", "==================================================", ""]
    lines += [f"Val fixed F1: {PARENT_VAL['fixed']}", f"Val oracle F1: {PARENT_VAL['oracle']}", f"Val AUC: {PARENT_VAL['auc']}", ""]
    lines += [f"Test fixed F1: {PARENT_TEST['fixed']}", f"Test oracle F1: {PARENT_TEST['oracle']}", f"Test AUC: {PARENT_TEST['auc']}", ""]
    for spec in runs:
        b = spec.best
        lines += ["==================================================", spec.name.upper(), "==================================================", ""]
        lines += [f"status: {spec.status}", f"decision: {spec.decision}", f"epochs: {len(spec.history_rows)}", f"best epoch: {b.get('epoch', '')}"]
        lines += [f"best val fixed F1: {b.get('fixed', '')}", f"best val oracle F1: {b.get('oracle', '')}", f"best val AUC: {b.get('auc', '')}", f"threshold: {b.get('threshold', '')}"]
        if b:
            lines += [
                f"delta fixed vs parent: {b['fixed'] - PARENT_VAL['fixed']}",
                f"delta oracle vs parent: {b['oracle'] - PARENT_VAL['oracle']}",
                f"delta AUC vs parent: {b['auc'] - PARENT_VAL['auc']}",
            ]
        lines += [f"output_dir: {spec.output_dir}", f"log: {spec.log_path}", f"checkpoint: {spec.checkpoint}"]
        scalars = scalar_from_ckpt(
            spec.checkpoint or Path("/missing"),
            [
                "residual_stream_gain",
                "cross_scale_reliability.alpha",
                "coarse_to_fine_icvm.alpha",
                "conflict_cross_scale_reasoning.alpha",
                "multiscale_evidence_feedback.beta",
                "cbr.alpha",
                "dhel_feature_fusion_gain",
            ],
        )
        lines += ["scalars: " + json.dumps(scalars, sort_keys=True)]
        lines += ["test: " + json.dumps(read_test_metrics(spec.test_output), sort_keys=True)]
        if spec.error:
            lines += [f"error: {spec.error}"]
        lines += [""]
    best = max((r for r in runs if r.best), key=lambda r: (r.best.get("oracle", -1), r.best.get("auc", -1)), default=None)
    lines += ["==================================================", "FINAL RECOMMENDATION", "==================================================", ""]
    lines += [f"Tonight best checkpoint: {best.checkpoint if best else ''}"]
    lines += ["Next step: inspect trends in history and prioritize Run C continuation if it is still improving."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_only() -> None:
    required = [
        REPO / "train.py",
        REPO / "evaluate.py",
        REPO / "evaluate_robustness.py",
        REPO / "retrace/model.py",
        REPO / "retrace/hierarchy.py",
        PARENT_CKPT,
        PYTHON,
    ]
    configs = [
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_a_residual_source.yaml",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_b_icvm_cderf.yaml",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_c_full_hierarchy.yaml",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_c_plus_cbr.yaml",
    ]
    missing = [str(path) for path in required + configs if not path.exists()]
    if missing:
        raise SystemExit("missing paths:\n" + "\n".join(missing))
    for config in configs:
        yaml.safe_load(config.read_text(encoding="utf-8"))
    print("check_only=ok")
    print(f"parent_epoch={checkpoint_epoch(PARENT_CKPT)}")
    print(f"gpu0_idle={gpu_idle(0)} gpu1_idle={gpu_idle(1)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only()
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = LOG_DIR / "hmsi_overnight_manifest.txt"
    summary = LOG_DIR / "hmsi_overnight_summary.txt"
    config_dir = LOG_DIR / f"hmsi_overnight_{stamp}_configs"
    backup_dir = REPO / "backups" / f"hmsi_overnight_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    files_to_backup = [
        REPO / "retrace/model.py",
        REPO / "retrace/hierarchy.py",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_a_residual_source.yaml",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_b_icvm_cderf.yaml",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_c_full_hierarchy.yaml",
        REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_c_plus_cbr.yaml",
    ]
    append(manifest, f"[{now()}] OVERNIGHT RUNNER START stamp={stamp} pid={os.getpid()}")
    append(manifest, f"[{now()}] code_backup={backup_dir}")
    for path in files_to_backup:
        shutil.copy2(path, backup_dir / path.name)
        append(manifest, f"[{now()}] code_file {path} sha256={sha256(path)}")

    parent_epoch = checkpoint_epoch(PARENT_CKPT)
    specs = [
        RunSpec("Run A", "run_a_residual_source", REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_a_residual_source.yaml", PARENT_CKPT, 8, 0),
        RunSpec("Run B", "run_b_icvm_cderf", REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_b_icvm_cderf.yaml", PARENT_CKPT, 12, 0),
        RunSpec("Run C", "run_c_full_hierarchy", REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_c_full_hierarchy.yaml", PARENT_CKPT, 15, 0, True, 15),
    ]
    aux: list[subprocess.Popen[str]] = []
    completed: list[RunSpec] = []
    severe_failure = False
    for spec in specs:
        make_runtime_config(spec, config_dir, stamp, parent_epoch, reset_optimizer=True)
        run_training(spec, manifest, reset_optimizer=True)
        if spec.name == "Run A":
            spec.decision = decide_run_a(spec)
            severe_failure = spec.status == "FAILED"
        elif spec.name == "Run B":
            spec.decision = decide_run_b(spec, completed[0] if completed else None)
        elif spec.name == "Run C":
            spec.decision = decide_run_c(spec)
            if spec.decision in {"PROMISING", "BEST"} and spec.output_dir and (spec.output_dir / "last.pt").exists():
                last_epoch = checkpoint_epoch(spec.output_dir / "last.pt")
                spec.init_checkpoint = spec.output_dir / "last.pt"
                make_continuation_config(spec, config_dir, last_epoch)
                append(manifest, f"[{now()}] CONTINUE Run C from last.pt for {spec.continuation_new_epochs} more epochs")
                run_training(spec, manifest, reset_optimizer=False)
                spec.decision = decide_run_c(spec)
        completed.append(spec)
        if spec.checkpoint and spec.checkpoint.exists():
            eval_proc = launch_eval(spec, manifest, gpu=1, robustness=False)
            if eval_proc:
                aux.append(eval_proc)
        if severe_failure:
            append(manifest, f"[{now()}] severe Run A failure; skip B/C")
            break

    run_c = next((item for item in completed if item.name == "Run C"), None)
    if run_c and cbr_allowed(run_c):
        cbr_gpu = 1 if gpu_idle(1) else 0
        cbr = RunSpec("Run C+CBR", "run_c_plus_cbr", REPO / "configs/casia2_retrace_resnet50_poscrop70_hmsi_run_c_plus_cbr.yaml", run_c.checkpoint or PARENT_CKPT, 8, cbr_gpu)
        cbr_resume_epoch = checkpoint_epoch(cbr.init_checkpoint)
        make_runtime_config(cbr, config_dir, stamp, cbr_resume_epoch, reset_optimizer=True)
        run_training(cbr, manifest, reset_optimizer=True)
        if cbr.best and run_c.best:
            if cbr.best["oracle"] > run_c.best["oracle"] and cbr.best["auc"] >= run_c.best["auc"]:
                cbr.decision = "STRONG KEEP"
            elif cbr.best["oracle"] > run_c.best["oracle"] or cbr.best["fixed"] > run_c.best["fixed"]:
                cbr.decision = "KEEP"
            else:
                cbr.decision = "DROP"
        completed.append(cbr)
        eval_proc = launch_eval(cbr, manifest, gpu=1, robustness=False)
        if eval_proc:
            aux.append(eval_proc)
    else:
        append(manifest, f"[{now()}] CBR skipped; Run C acceptable={cbr_allowed(run_c) if run_c else False}")

    if run_c and run_c.decision in {"PROMISING", "BEST"}:
        robust_proc = launch_eval(run_c, manifest, gpu=1, robustness=True)
        if robust_proc:
            aux.append(robust_proc)

    for proc in aux:
        proc.wait()
    write_summary(summary, completed)
    append(manifest, f"[{now()}] OVERNIGHT RUNNER DONE summary={summary}")


if __name__ == "__main__":
    main()
