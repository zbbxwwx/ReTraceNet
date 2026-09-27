"""Dataset-specific V3 fine-tuning from the CASIA2 V3 localization checkpoint."""
import argparse, copy, json, math, random, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader

from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from run_v3_indomain_721 import split_rows
from fair_training_data import FairDataset


SLOW_PREFIXES = ("content_encoder.", "forensic_stream.", "dct_branch.")

def freeze_pretrained_bn_stats(model):
    count = 0
    for branch in (model.content_encoder, model.dct_branch):
        for module in branch.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval(); count += 1
    return count

def mode_train_stable(model):
    v3.mode_train(model)
    count = freeze_pretrained_bn_stats(model)
    if count <= 0:
        raise RuntimeError("No pretrained BN modules found")
    return count

def evaluate_stable(model, *args, **kwargs):
    was_training = model.training
    model.eval()
    try:
        return cross.evaluate_dataset(model, *args, **kwargs)
    finally:
        if was_training:
            mode_train_stable(model)


def dump(path, value):
    v3.e.dump(Path(path), value)


def make_records(rows, split, dataset):
    return [Record(image=Path(r["image"]), mask=Path(r["mask"]) if r.get("mask") else None,
                   split=split, dataset=dataset, label=int(r["label"])) for r in rows]


def load_or_create_split(inventory, out, seed):
    path = out / "split_721.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = [json.loads(x) for x in Path(inventory).read_text().splitlines() if x.strip()]
    split = split_rows(rows, seed)
    doc = {key: [{"image": r["image"], "mask": r.get("mask"), "label": r["label"],
                  "rgb_sha256": r["rgb_sha256"]} for r in value]
           for key, value in split.items()}
    path.write_text(json.dumps(doc, indent=2))
    return doc


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "preparing", "dataset": a.dataset,
               "experiment": "dataset_specific_finetuning_from_CASIA2_pretrained_V3",
               "selection": "target val fake-only pooled F1-best",
               "test_policy": "target test exactly once after training",
               "target": {"f1": a.target_f1, "auc": a.target_auc}}
    dump(out / "summary.yaml", summary)
    try:
        split = load_or_create_split(a.inventory, out, a.seed)
        tr, va, te = (make_records(split[k], k, a.dataset) for k in ("train", "val", "test"))
        fake = [i for i, r in enumerate(tr) if r.label]
        auth = [i for i, r in enumerate(tr) if not r.label]
        if not fake or not auth or not va or not te:
            raise RuntimeError("train must contain fake+auth and val/test must be nonempty")

        source = torch.load(a.pretrained, map_location="cpu")
        cfg = copy.deepcopy(source["config"])
        cfg.setdefault("training", {})["dct_sample_dropout"] = a.dct_dropout
        model = v3.e.build_final_model(cfg["model"]).cuda()
        model.load_state_dict(source["model"], strict=True)
        bn_count = mode_train_stable(model)
        seg = v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg["loss"])).cuda()

        slow, fast = [], []
        for name, parameter in model.named_parameters():
            (slow if name.startswith(SLOW_PREFIXES) else fast).append(parameter)
        opt = torch.optim.SGD([
            {"params": slow, "lr": a.base_lr * .1, "name": "pretrained_encoders"},
            {"params": fast, "lr": a.base_lr, "name": "localization_chain"}],
            momentum=.9, weight_decay=5e-4)

        history, best, best_epoch, bad, global_step, stable_pass_epochs = [], -1., 0, 0, 0, 0
        start_epoch = 1
        last_path = out / "last.pt"
        if a.resume and last_path.exists():
            resume = torch.load(last_path, map_location="cpu")
            model.load_state_dict(resume["model"], strict=True)
            opt.load_state_dict(resume["optimizer"])
            history = resume.get("history", [])
            best = float(resume.get("best_metric", -1))
            best_epoch = int(resume.get("best_epoch", 0))
            bad = int(resume.get("bad_epochs", 0))
            global_step = int(resume.get("global_step", 0))
            stable_pass_epochs = int(resume.get("stable_pass_epochs", 0))
            start_epoch = int(resume["epoch"]) + 1

        nf = min(len(fake), len(auth))
        steps_per_epoch = math.ceil(2 * nf / a.batch_size)
        total_steps = a.epochs * steps_per_epoch
        summary.update(status="training", pretrained_checkpoint=a.pretrained,
                       counts={k: len(split[k]) for k in split},
                       label_counts={k: {str(z): sum(int(r["label"]) == z for r in split[k]) for z in (0, 1)} for k in split},
                       learning_rates={"pretrained_encoders": a.base_lr * .1, "localization_chain": a.base_lr},
                       max_epochs=a.epochs, early_stop_patience=a.patience,
                       dct_sample_dropout=a.dct_dropout, batch_size=a.batch_size,
                       pretrained_bn_stats_frozen=True, pretrained_bn_module_count=bn_count,
                       history=history)
        dump(out / "summary.yaml", summary)

        for epoch in range(start_epoch, a.epochs + 1):
            generator = torch.Generator().manual_seed(a.seed + epoch)
            ids = [fake[i] for i in torch.randperm(len(fake), generator=generator)[:nf]]
            ids += [auth[i] for i in torch.randperm(len(auth), generator=generator)[:nf]]
            ids = [ids[i] for i in torch.randperm(len(ids), generator=generator)]
            plan = [(idx, a.seed * 1000000 + epoch * 100000 + j) for j, idx in enumerate(ids)]
            ds = FairDataset(tr, out / "jpeg_cache_train", True)
            loader = DataLoader(ds, batch_size=a.batch_size, sampler=plan, num_workers=2, pin_memory=True)
            mode_train_stable(model); running = 0.
            for step, batch in enumerate(loader, 1):
                progress = min(global_step / max(total_steps, 1), 1.)
                factor = (1. - progress) ** .9
                opt.param_groups[0]["lr"] = a.base_lr * .1 * factor
                opt.param_groups[1]["lr"] = a.base_lr * factor
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    terms = v3.objective(v3.fwd(model, batch, cfg, True), batch, seg, cfg)
                    loss = terms["total"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss epoch={epoch} step={step}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); opt.step()
                global_step += 1; running += float(loss)
                if step == 1 or step % 25 == 0:
                    progress_doc = {"dataset": a.dataset, "epoch": epoch, "max_epochs": a.epochs,
                                    "batch": step, "batches": len(loader), "global_step": global_step,
                                    "loss": running / step, "lr_encoder": opt.param_groups[0]["lr"],
                                    "lr_chain": opt.param_groups[1]["lr"]}
                    dump(out / "progress.yaml", progress_doc)
                    print("[target-adapt]", a.dataset, epoch, step, len(loader), running / step, flush=True)

            val_result = evaluate_stable(model, va, out / "jpeg_cache_val", None, None,
                a.dataset, out / f"val_epoch{epoch:03d}", evaluation_mode="validation_select",
                threshold_source=a.dataset + "_val_current")
            oracle = val_result["pixel_localization_fake_only"]["test_f1_oracle_diagnostic"]
            score = float(oracle["f1"])
            row = {"epoch": epoch, "global_step": global_step, "train_loss": running / len(loader),
                   "val_fake_f1": score, "val_fake_auc": val_result["pixel_localization_fake_only"]["auc"],
                   "threshold": float(oracle["threshold"]),
                   "image_threshold": float(val_result["image_level_all"]["test_f1_oracle_diagnostic"]["threshold"])}
            val_auc = row["val_fake_auc"]
            passed = bool(score > a.target_f1 and val_auc is not None and float(val_auc) > a.target_auc)
            stable_pass_epochs = stable_pass_epochs + 1 if passed else 0
            row["benchmark_pass"] = passed
            row["stable_pass_epochs"] = stable_pass_epochs
            history.append(row)
            improved = score > best
            if improved:
                best, best_epoch, bad = score, epoch, 0
            else:
                bad += 1
            state = {"model": v3.e.copy_state(model), "optimizer": opt.state_dict(), "config": cfg,
                     "epoch": epoch, "global_step": global_step, "history": history,
                     "best_metric": best, "best_epoch": best_epoch, "bad_epochs": bad,
                     "stable_pass_epochs": stable_pass_epochs,
                     "selection_threshold": row["threshold"], "image_selection_threshold": row["image_threshold"],
                     "source_checkpoint": a.pretrained}
            v3.e24.atomic_torch_save(state, last_path)
            if improved:
                v3.e24.atomic_torch_save(state, out / "best.pt")
            summary.update(history=history, best_F1=best, best_epoch=best_epoch,
                           best_checkpoint=str(out / "best.pt"), bad_epochs=bad,
                           stable_pass_epochs=stable_pass_epochs)
            dump(out / "summary.yaml", summary)
            if stable_pass_epochs >= a.stable_epochs:
                break
            if bad >= a.patience:
                break

        best_state = torch.load(out / "best.pt", map_location="cpu")
        model.load_state_dict(best_state["model"], strict=True)
        test = evaluate_stable(model, te, out / "jpeg_cache_test",
            float(best_state["selection_threshold"]), float(best_state["image_selection_threshold"]),
            a.dataset, out / "test_once", evaluation_mode="target_adapted_in_domain_locked",
            threshold_source=a.dataset + "_val")
        locked = test["pixel_localization_fake_only"]["test_f1_locked"]
        test_auc = test["pixel_localization_fake_only"]["auc"]
        summary.update(status="completed", test_run=True, test_checkpoint_epoch=best_epoch,
            test_locked_threshold=float(best_state["selection_threshold"]), test=test,
            benchmark_comparison={"target_f1": a.target_f1, "actual_f1": locked["f1"],
                "f1_pass": bool(locked["f1"] > a.target_f1), "target_auc": a.target_auc,
                "actual_auc": test_auc, "auc_pass": bool(test_auc is not None and test_auc > a.target_auc)},
            stop_reason=("stable_validation_target_exceeded" if stable_pass_epochs >= a.stable_epochs
                         else "early_stop" if bad >= a.patience else "fixed_budget_complete"))
        dump(out / "summary.yaml", summary)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(out / "summary.yaml", summary)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--inventory", required=True)
    p.add_argument("--pretrained", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--base-lr", type=float, default=1e-4); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dct-dropout", type=float, default=.2); p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--target-f1", type=float, required=True); p.add_argument("--target-auc", type=float, required=True)
    p.add_argument("--stable-epochs", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    main(p.parse_args())
