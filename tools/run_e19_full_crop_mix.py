"""E19: matched E17-A continuation with only train geometry changed to 50/50 full/crop."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrace.data import ManipulationDataset, pil_to_tensor, read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import (
    clean_loss_config,
    dher_v2_aux_loss,
    train_loader_for,
    tversky_loss,
    weighted_deep_supervision_loss,
)
from train_e15_hr_corrected import dump, evaluate
from e15_original_metrics import stratum
from run_e17_e5_pair import sha_file, state_hash


TRAINABLE = ["decoder.", "mask_head.", "log_var_head.", "aux_heads."]


class FullCropMixDataset(ManipulationDataset):
    forced_geometry = "crop"

    def _joint_geometry(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.training and self.forced_geometry == "full":
            target = self.image_size
            self.last_geometry = "full"
            image = image.resize((target, target), Image.Resampling.BILINEAR)
            mask = mask.resize((target, target), Image.Resampling.NEAREST)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.1:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            return image, mask
        self.last_geometry = "crop"
        return super()._joint_geometry(image, mask)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample["geometry"] = self.last_geometry
        mask = sample["mask"].numpy()
        sample["post_mask_area_ratio"] = float((mask > 0.5).mean())
        sample["post_geometry_bucket"] = stratum(sample["path"], mask) if mask.any() else "authentic_or_empty"
        return sample


class GeometryReplayDataset(Dataset):
    def __init__(self, dataset: FullCropMixDataset, indices, seeds, flags):
        self.dataset = dataset
        self.indices = indices
        self.seeds = seeds
        self.flags = flags

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = int(self.indices[position])
        seed = int(self.seeds[position])
        geometry = "full" if int(self.flags[position]) else "crop"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset.forced_geometry = geometry
        sample = self.dataset[index]
        h = hashlib.sha256()
        h.update(sample["image"].numpy().tobytes())
        h.update(sample["mask"].numpy().tobytes())
        h.update(sample["label"].numpy().tobytes())
        sample.update(
            replay_index=index,
            replay_seed=seed,
            geometry_flag=geometry,
            input_sha256=h.hexdigest(),
        )
        return sample


def convert_dataset(dataset: ManipulationDataset) -> FullCropMixDataset:
    dataset.__class__ = FullCropMixDataset
    dataset.forced_geometry = "crop"
    dataset.last_geometry = "crop"
    return dataset


def optimizer_for(model):
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in TRAINABLE))
    model.eval()
    groups = []
    for prefix in TRAINABLE:
        params = [param for name, param in model.named_parameters() if name.startswith(prefix) and param.requires_grad]
        assert params, prefix
        groups.append({"params": params, "name": prefix, "lr": 5e-6})
    assert not any(param.requires_grad for param in model.content_encoder.parameters())
    assert not any(param.requires_grad for param in model.forensic_stream.parameters())
    assert not any(param.requires_grad for param in model.qtr.parameters())
    assert not any(param.requires_grad for param in model.hcv2.parameters())
    assert not any(param.requires_grad for param in model.dher.parameters())
    return torch.optim.AdamW(groups, weight_decay=0.01)


def objective(out, batch, seg, cfg):
    target = batch["mask"].cuda()
    label = batch["label"].cuda()
    terms = seg(out, target, label)
    aux = dher_v2_aux_loss(out, target, cfg)
    terms["total"] = terms["total"] + aux["dher_aux_total"]
    ds = weighted_deep_supervision_loss(out, target, cfg)
    tv = tversky_loss(
        out["mask_logits"],
        target,
        float(cfg["loss"].get("tversky_alpha_fp", 0.3)),
        float(cfg["loss"].get("tversky_beta_fn", 0.7)),
    )
    terms.update(weighted_deep_supervision=ds, tversky=tv)
    terms["total"] = terms["total"] + ds + float(cfg["loss"]["weights"].get("tversky", 0.0)) * tv
    return terms


def make_geometry_flags(count: int, epoch: int) -> np.ndarray:
    flags = np.zeros(count, dtype=np.int8)
    flags[: (count + 1) // 2] = 1
    rng = np.random.default_rng(203407 + epoch)
    rng.shuffle(flags)
    return flags


def summarize_sampled(sampled):
    total = max(1, sum(sampled["geometry"].values()))
    return {
        "geometry": dict(sampled["geometry"]),
        "geometry_fraction": {key: value / total for key, value in sampled["geometry"].items()},
        "post_geometry_bucket": dict(sampled["post_geometry_bucket"]),
    }


def audit(model, opt, batch, seg, cfg, source_state):
    before = state_hash(model)
    frozen = state_hash(model, True)
    out = model(batch["image"].cuda())
    pred_hash = hashlib.sha256(out["mask_logits"].detach().cpu().numpy().tobytes()).hexdigest()
    terms = objective(out, batch, seg, cfg)
    terms["total"].backward()
    groups = {}
    for group in opt.param_groups:
        named = [(name, param) for name, param in model.named_parameters() if name.startswith(group["name"])]
        grads = [param.grad for _, param in named if param.grad is not None]
        norm = sum(float(grad.detach().square().sum()) for grad in grads) ** 0.5
        assert math.isfinite(norm) and norm > 0, (group["name"], norm)
        groups[group["name"]] = {
            "parameters": sum(param.numel() for _, param in named),
            "with_grad": len(grads),
            "grad_norm": norm,
            "lr": group["lr"],
        }
    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 0.5, error_if_nonfinite=True)
    opt.step()
    for prefix, row in groups.items():
        delta = sum(
            float((param.detach().cpu() - source_state[name]).abs().sum())
            for name, param in model.named_parameters()
            if name.startswith(prefix)
        )
        assert delta > 0, (prefix, delta)
        row["parameter_update_abs_sum"] = delta
    assert frozen == state_hash(model, True)
    model.load_state_dict(source_state, strict=True)
    opt.state.clear()
    opt.zero_grad(set_to_none=True)
    assert state_hash(model) == before
    return {
        "status": "PASS",
        "initial_prediction_sha256": pred_hash,
        "batch_geometry": batch["geometry_flag"][0],
        "batch_post_bucket": batch["post_geometry_bucket"][0],
        "groups": groups,
        "frozen_state_unchanged": True,
        "audit_step_restored": True,
    }


def main(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.yaml"
    assert not summary_path.exists(), "Refuse duplicate run"
    summary = {
        "status": "initializing",
        "experiment": "E19_full_crop_mix",
        "history": [],
        "test_run": False,
        "max_epochs": 5,
        "single_change": "train geometry is 50 percent full-image resize and 50 percent original E5 crop pipeline",
        "selection": "val_original_grid_fake_pooled_F1_best",
        "auc_used_for_selection": False,
        "stop_reason": None,
        "automatic_second_seed": False,
    }
    dump(summary_path, summary)
    try:
        set_seed(3407)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        cfg = checkpoint["config"]
        source_state = checkpoint["model"]
        assert checkpoint.get("eval_variant", "raw") == "raw"
        assert cfg["model"]["architecture"] == "retracenet_b3_qtr_hcv2_dher_v21_mrfe"
        cfg["training"]["batch_size"] = 1
        e17 = yaml.safe_load(Path(args.e17_summary).read_text())
        assert e17["status"] == "completed"
        plan_path = Path(args.replay_plan)
        assert sha_file(plan_path) == e17["replay_plan_sha256"]
        plan = np.load(plan_path)
        base = train_loader_for(cfg)
        dataset = convert_dataset(base.dataset)
        count = len(base.sampler)
        assert count == e17["training_microbatches_per_epoch"]
        flags = {f"geometry_flags_{epoch}": make_geometry_flags(count, epoch) for epoch in range(5)}
        geometry_plan_path = outdir / "geometry_flags_plan.npz"
        np.savez_compressed(geometry_plan_path, **flags)
        summary.update(
            source_checkpoint=args.checkpoint,
            source_checkpoint_sha256=sha_file(args.checkpoint),
            source_checkpoint_epoch=checkpoint["epoch"],
            source_config=cfg,
            e17_A_reference={
                "best_F1": e17["arms"]["A_decoder_only"]["best_F1"],
                "best_epoch": e17["arms"]["A_decoder_only"]["best_epoch"],
                "final_F1": e17["history"]["A_decoder_only"][-1]["selection_value"],
                "history": [row["selection_value"] for row in e17["history"]["A_decoder_only"]],
            },
            e5_reference=e17["baseline"]["metrics"],
            replay_plan=str(plan_path),
            replay_plan_sha256=sha_file(plan_path),
            geometry_plan=str(geometry_plan_path),
            geometry_plan_sha256=sha_file(geometry_plan_path),
            training_microbatches_per_epoch=count,
            optimizer_steps_per_epoch=math.ceil(count / 4),
            optimizer_steps_total=5 * math.ceil(count / 4),
            optimizer={"type": "AdamW", "lr": 5e-6, "weight_decay": 0.01, "constant_lr": True},
            gradient_accumulation=4,
            gradient_clip=0.5,
            layers_mode="eval; selected E17-A decoder/localization parameters require gradients",
        )
        seg = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda()
        audit_replay = GeometryReplayDataset(dataset, plan["indices_0"][:32], plan["seeds_0"][:32], flags["geometry_flags_0"][:32])
        audit_batch = next(batch for batch in DataLoader(audit_replay, batch_size=1, num_workers=0) if batch["mask"].sum() > 0)
        model = build_final_model(cfg["model"]).cuda()
        model.load_state_dict(source_state, strict=True)
        opt = optimizer_for(model)
        summary["preflight"] = audit(model, opt, audit_batch, seg, cfg, source_state)
        dump(outdir / "preflight.yaml", summary["preflight"])
        dump(summary_path, summary)
        print("[preflight]", json.dumps(summary["preflight"]), flush=True)
        val = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "val")
        frozen_initial = state_hash(model, True)
        best = -math.inf
        for epoch in range(5):
            replay = GeometryReplayDataset(dataset, plan[f"indices_{epoch}"], plan[f"seeds_{epoch}"], flags[f"geometry_flags_{epoch}"])
            loader = DataLoader(replay, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
            model.eval()
            opt.zero_grad(set_to_none=True)
            totals = {}
            stream = hashlib.sha256()
            started = time.time()
            sampled = {"geometry": {"full": 0, "crop": 0}, "post_geometry_bucket": {}}
            summary.update(status="training", current_epoch=epoch + 1)
            dump(summary_path, summary)
            print(f"[train] epoch={epoch+1}/5 step=0/{count}", flush=True)
            with (outdir / f"inputs_epoch_{epoch+1:02d}.jsonl").open("w", encoding="utf-8") as handle:
                for bi, batch in enumerate(loader):
                    geometry = batch["geometry_flag"][0]
                    bucket = batch["post_geometry_bucket"][0]
                    sampled["geometry"][geometry] = sampled["geometry"].get(geometry, 0) + 1
                    sampled["post_geometry_bucket"][bucket] = sampled["post_geometry_bucket"].get(bucket, 0) + 1
                    ident = {
                        "position": bi,
                        "index": int(batch["replay_index"]),
                        "seed": int(batch["replay_seed"]),
                        "geometry": geometry,
                        "path": batch["path"][0],
                        "input_sha256": batch["input_sha256"][0],
                    }
                    line = json.dumps(ident, sort_keys=True)
                    stream.update(line.encode())
                    handle.write(line + "\n")
                    terms = objective(model(batch["image"].cuda()), batch, seg, cfg)
                    assert torch.isfinite(terms["total"])
                    accumulation = min(4, count - (bi // 4) * 4)
                    (terms["total"] / accumulation).backward()
                    if (bi + 1) % 4 == 0 or bi + 1 == count:
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            0.5,
                            error_if_nonfinite=True,
                        )
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    for key, value in terms.items():
                        totals[key] = totals.get(key, 0.0) + float(value.detach())
                    if bi == 0 or (bi + 1) % 50 == 0 or bi + 1 == count:
                        elapsed = time.time() - started
                        progress = {
                            "epoch": epoch + 1,
                            "step": bi + 1,
                            "steps": count,
                            "loss": totals["total"] / (bi + 1),
                            "full": sampled["geometry"].get("full", 0),
                            "crop": sampled["geometry"].get("crop", 0),
                            "elapsed_minutes": elapsed / 60,
                            "eta_minutes": elapsed / (bi + 1) * (count - bi - 1) / 60,
                        }
                        dump(outdir / "progress.yaml", progress)
                        print("[train]", json.dumps(progress), flush=True)
            assert state_hash(model, True) == frozen_initial
            summary["status"] = "validation"
            dump(summary_path, summary)
            result = evaluate(model, val, f"epoch_{epoch+1:02d}_original_val", outdir)
            current = result["metrics"]["fake"]["oracle"]["f1"]
            row = {
                "epoch": epoch + 1,
                "optimizer_steps": (epoch + 1) * math.ceil(count / 4),
                "selection_value": current,
                "delta_vs_E5": current - e17["baseline"]["metrics"]["fake"]["oracle"]["f1"],
                "delta_vs_matched_E17_A": current - e17["history"]["A_decoder_only"][epoch]["selection_value"],
                "train_losses": {key: value / count for key, value in totals.items()},
                "actual_input_stream_sha256": stream.hexdigest(),
                "sampled": summarize_sampled(sampled),
                "validation": result,
                "elapsed_seconds": time.time() - started,
            }
            is_best = current > best
            best = max(best, current)
            state = {
                "epoch": epoch,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "config": cfg,
                "eval_variant": "raw",
                "phase": "E19_full_crop_mix",
                "best_metric": best,
                "selection_threshold": result["metrics"]["fake"]["oracle"]["threshold"],
            }
            atomic_torch_save(state, outdir / "last.pt")
            if is_best:
                atomic_torch_save(state, outdir / "best.pt")
                summary.update(best_F1=best, best_epoch=epoch + 1, best_checkpoint=str(outdir / "best.pt"))
            summary["history"].append(row)
            e5 = e17["baseline"]["metrics"]
            big_recall = result["metrics"].get("Tp_S:gte_20", {}).get("fixed", {}).get("recall")
            tiny_f1 = result["metrics"].get("Tp_S:lt_1", {}).get("fixed", {}).get("f1")
            row["directional_signals"] = {
                "big_TpS_fixed_recall_delta_vs_E5": None if big_recall is None else big_recall - e5["Tp_S:gte_20"]["fixed"]["recall"],
                "tiny_TpS_fixed_F1_delta_vs_E5": None if tiny_f1 is None else tiny_f1 - e5["Tp_S:lt_1"]["fixed"]["f1"],
            }
            dump(summary_path, summary)
            print("[epoch_done]", epoch + 1, "F1", current, "delta_A", row["delta_vs_matched_E17_A"], flush=True)
            if epoch == 1:
                big_ok = row["directional_signals"]["big_TpS_fixed_recall_delta_vs_E5"]
                tiny_ok = row["directional_signals"]["tiny_TpS_fixed_F1_delta_vs_E5"]
                no_direction = (big_ok is None or big_ok <= 0.01) and (tiny_ok is None or tiny_ok <= 0.005)
                if best < 0.797 and no_direction:
                    summary.update(
                        status="completed",
                        stop_reason="ep2_gate_stop_best_below_0p797_and_no_big_or_tiny_directional_signal",
                        final_F1=current,
                        final_epoch=epoch + 1,
                        qualifies_for_extension=False,
                    )
                    assert sha_file(args.checkpoint) == summary["source_checkpoint_sha256"]
                    dump(summary_path, summary)
                    print("[E19_STOP]", summary["stop_reason"], flush=True)
                    return
        final = summary["history"][-1]["selection_value"]
        best_row = summary["history"][summary["best_epoch"] - 1]
        gates = {
            "best_at_least_0p800": best >= 0.800,
            "large_or_tiny_signal_present": (
                (best_row["directional_signals"]["big_TpS_fixed_recall_delta_vs_E5"] or 0.0) > 0.01
                or (best_row["directional_signals"]["tiny_TpS_fixed_F1_delta_vs_E5"] or 0.0) > 0.005
            ),
            "auc_not_used": True,
        }
        summary.update(
            status="completed",
            stop_reason="fixed_5_epoch_budget_completed",
            final_F1=final,
            final_epoch=5,
            gates=gates,
            qualifies_for_extension=bool(gates["best_at_least_0p800"] and gates["large_or_tiny_signal_present"]),
        )
        assert sha_file(args.checkpoint) == summary["source_checkpoint_sha256"]
        dump(summary_path, summary)
        print("[COMPLETE]", json.dumps(gates), flush=True)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--e17-summary", required=True)
    parser.add_argument("--replay-plan", required=True)
    main(parser.parse_args())
