"""Post-run causal audit and one matched anti-shortcut experiment.

Waits for Fair Training v1, evaluates its best checkpoint with DCT enabled and
disabled, and only when DCT is demonstrably responsible for JPEG-small false
positives runs a matched control versus DCT-dropout continuation.  No test set.
"""
import argparse
import copy
import json
import math
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import run_fair_training_v1 as fair

e = fair.e
e24 = fair.e24


def load_yaml(path):
    import yaml
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def save_yaml(path, value):
    e.dump(Path(path), value)


def wait_for_parent(parent, poll_seconds):
    while True:
        state = load_yaml(parent / "summary.yaml")
        status = state.get("status")
        if status == "completed":
            return state
        if status == "failed":
            raise RuntimeError("Parent fair training failed; refusing post-training runs")
        print("[wait]", status, "epoch", state.get("epoch"), flush=True)
        time.sleep(poll_seconds)


def build_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = copy.deepcopy(checkpoint["config"])
    model = e.build_final_model(cfg["model"]).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, cfg, model


def fixed_counts(metric):
    f = metric["fixed"]
    return {k: int(f[k]) for k in ("tp", "fp", "fn", "tn")}


def add_counts(*items):
    result = {k: sum(x[k] for x in items) for k in ("tp", "fp", "fn", "tn")}
    tp, fp, fn = result["tp"], result["fp"], result["fn"]
    result.update(
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
    )
    return result


def jpeg_small(result):
    jpeg = result["formats"]["jpeg"]
    return add_counts(fixed_counts(jpeg["Tp_S:lt_1"]), fixed_counts(jpeg["Tp_S:1_5"]))


@torch.inference_mode()
def audit_dct(checkpoint_path, val_ds, out):
    _, cfg, model = build_from_checkpoint(checkpoint_path)
    model.disable_dct_evidence = False
    enabled = fair.evaluate(model, val_ds, cfg, out, "dct_on")
    model.disable_dct_evidence = True
    disabled = fair.evaluate(model, val_ds, cfg, out, "dct_off")
    on_small, off_small = jpeg_small(enabled), jpeg_small(disabled)
    decision = {
        "checkpoint": str(checkpoint_path),
        "test_run": False,
        "dct_on": enabled,
        "dct_off": disabled,
        "jpeg_Tp_S_lt5_fixed_combined": {"on": on_small, "off": off_small},
    }
    # Require a material precision gain and preserve at least 85% of recall.
    causal = (
        off_small["precision"] >= on_small["precision"] + 0.03
        and off_small["recall"] >= 0.85 * on_small["recall"]
        and off_small["f1"] > on_small["f1"]
    )
    decision["dct_fp_causal_gate"] = bool(causal)
    decision["gate_rule"] = (
        "DCT-off JPEG Tp_S<5 fixed precision >= on+0.03, recall >= 0.85*on, and F1 improves"
    )
    save_yaml(out / "dct_causal_audit.yaml", decision)
    del model
    torch.cuda.empty_cache()
    return causal, decision


def train_arm(name, checkpoint_path, train_ds, val_ds, plan, out, dropout_probability):
    checkpoint, cfg, model = build_from_checkpoint(checkpoint_path)
    seg = e24.ReTraceLoss.from_config(e24.clean_loss_config(cfg["loss"])).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    arm_out = out / name
    arm_out.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "training",
        "arm": name,
        "init": str(checkpoint_path),
        "epochs": len(plan),
        "learning_rate": 1e-4,
        "dct_dropout_probability": dropout_probability,
        "selection": "val fake pooled F1-best",
        "test_run": False,
        "history": [],
        "best_F1": -1.0,
    }
    save_yaml(arm_out / "summary.yaml", state)
    total_steps = sum(math.ceil(len(p) / 4) for p in plan)
    global_step = 0
    generator = torch.Generator(device="cpu").manual_seed(7719)
    for epoch, epoch_plan in enumerate(plan, 1):
        fair.mode_train(model)
        loader = DataLoader(train_ds, batch_size=4, sampler=epoch_plan, num_workers=2, pin_memory=True)
        total = 0.0
        for step, batch in enumerate(loader, 1):
            lr = 1e-4 * (1 - global_step / total_steps) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = lr
            if dropout_probability:
                keep = torch.rand((batch["image"].shape[0], 1), generator=generator) >= dropout_probability
                batch["dct_available"] = batch["dct_available"] * keep.to(batch["dct_available"].dtype)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                terms = fair.objective(fair.fwd(model, batch), batch, seg, cfg)
                loss = terms["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} nonfinite epoch={epoch} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            total += float(loss.detach())
            if step == 1 or step % 50 == 0 or step == len(loader):
                progress = {"arm": name, "epoch": epoch, "batch": step, "batches": len(loader),
                            "loss": total / step, "lr": lr, "global_step": global_step}
                save_yaml(arm_out / "progress.yaml", progress)
                print("[train]", json.dumps(progress), flush=True)
        result = fair.evaluate(model, val_ds, cfg, arm_out, f"epoch{epoch:02d}")
        score = result["metrics"]["fake"]["oracle"]["f1"]
        row = {"epoch": epoch, "train_loss": total / len(loader), "selection_value": score,
               "validation": result}
        state["history"].append(row)
        ck = {"model": e.copy_state(model), "optimizer": optimizer.state_dict(), "config": cfg,
              "epoch": epoch, "global_step": global_step, "eval_variant": "raw",
              "best_metric": score, "selection_threshold": result["metrics"]["fake"]["oracle"]["threshold"]}
        e24.atomic_torch_save(ck, arm_out / "last.pt")
        if score > state["best_F1"]:
            state.update(best_F1=score, best_epoch=epoch, best_checkpoint=str(arm_out / "best.pt"))
            e24.atomic_torch_save(ck, arm_out / "best.pt")
        save_yaml(arm_out / "summary.yaml", state)
    state.update(status="completed", stop_reason="fixed_5_epoch_matched_budget_complete_no_test")
    save_yaml(arm_out / "summary.yaml", state)
    del model, optimizer, seg
    torch.cuda.empty_cache()
    return state


def main(args):
    parent, out = Path(args.parent), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "waiting_for_parent", "parent": str(parent), "test_run": False}
    save_yaml(out / "summary.yaml", summary)
    try:
        parent_state = wait_for_parent(parent, args.poll_seconds)
        checkpoint_path = Path(parent_state["best_checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = checkpoint["config"]
        train_records = e.read_manifest(parent / "clean_manifest.csv", cfg["data"]["root"], "train")
        val_records = e.read_manifest(parent / "clean_manifest.csv", cfg["data"]["root"], "val")
        train_ds = fair.FairDataset(train_records, parent / "jpeg_cache", True)
        val_ds = fair.FairDataset(val_records, parent / "jpeg_cache", False)
        causal, audit = audit_dct(checkpoint_path, val_ds, out)
        summary.update(status="audited", parent_best=parent_state["best_F1"], audit_gate=causal)
        save_yaml(out / "summary.yaml", summary)
        if not causal:
            summary.update(status="completed", stop_reason="DCT causal gate failed; no speculative training launched")
            save_yaml(out / "summary.yaml", summary)
            return
        fake = [i for i, r in enumerate(train_records) if r.label]
        auth = [i for i, r in enumerate(train_records) if not r.label]
        plans = []
        for ep in range(5):
            gen = torch.Generator().manual_seed(8800 + ep)
            ids = fake + [auth[i] for i in torch.randperm(len(auth), generator=gen)[:len(fake)].tolist()]
            ids = [ids[i] for i in torch.randperm(len(ids), generator=gen).tolist()]
            plans.append([(idx, 880000000 + ep * 100000 + j) for j, idx in enumerate(ids)])
        control = train_arm("control_continue", checkpoint_path, train_ds, val_ds, plans, out, 0.0)
        dropout = train_arm("dct_dropout_p50", checkpoint_path, train_ds, val_ds, plans, out, 0.5)
        gain = dropout["best_F1"] - control["best_F1"]
        summary.update(status="completed", control_best=control["best_F1"],
                       dct_dropout_best=dropout["best_F1"], paired_gain=gain,
                       accepted=bool(dropout["best_F1"] > parent_state["best_F1"] and gain >= 0.003),
                       stop_reason="fixed paired audit and training budget complete; no test")
        save_yaml(out / "summary.yaml", summary)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        save_yaml(out / "summary.yaml", summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    main(parser.parse_args())
