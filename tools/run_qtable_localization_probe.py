"""Zero-training full-model qtable localization probe on CASIA2 val / Q90."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import jpegio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from run_e3_fullmodel_jpeg_adaptation import Fixed

PARENT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "mit_full_offline_20260918/CASIA2_mit_full100/best.pt"
)

E3 = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "e3_fullmodel_jpeg_adaptation1_20260922/qtr_localnorm_epoch01.pt"
)

MANIFEST = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "balanced_residual_v3_full30_20260912/clean_manifest.csv"
)

OUT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "qtable_localization_probe_20260922"
)


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False))


def canonical_qtable(quality):
    arr = np.full((16, 16, 3), 128, dtype=np.uint8)

    with tempfile.TemporaryDirectory(prefix="qtable_probe_") as folder:
        p = Path(folder) / "x.jpg"

        Image.fromarray(arr).save(
            p,
            "JPEG",
            quality=int(quality),
            subsampling=2,
        )

        jpg = jpegio.read(str(p))
        qt = jpg.quant_tables[
            jpg.comp_info[0].quant_tbl_no
        ].astype("float32")

    return torch.from_numpy(qt[None])


Q100 = canonical_qtable(100)
Q90 = canonical_qtable(90)


class ReplaceQtable(torch.utils.data.Dataset):
    def __init__(self, image_dataset, qtable_dataset=None, fixed_qtable=None):
        self.image_dataset = image_dataset
        self.qtable_dataset = qtable_dataset
        self.fixed_qtable = fixed_qtable
        self.records = image_dataset.records

        assert (qtable_dataset is None) ^ (fixed_qtable is None)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, key):
        s = self.image_dataset[key]
        out = dict(s)

        if self.qtable_dataset is not None:
            ref = self.qtable_dataset[key]
            out["qtable"] = ref["qtable"].clone()
        else:
            out["qtable"] = self.fixed_qtable.clone()

        return out


def summarize(z):
    m = z["metrics"]["fake"]

    out = {
        "f1_best": float(m["oracle"]["f1"]),
        "precision": float(m["oracle"]["precision"]),
        "recall": float(m["oracle"]["recall"]),
        "auc": float(m["auc"]),
        "threshold": float(m["oracle"]["threshold"]),
    }

    if "formats" in z:
        out["formats"] = {}

        for name in ("jpeg", "nonjpeg"):
            if name not in z["formats"]:
                continue

            f = z["formats"][name]["fake"]

            out["formats"][name] = {
                "count": int(f["count"]),
                "f1_best": float(f["oracle"]["f1"]),
                "precision": float(f["oracle"]["precision"]),
                "recall": float(f["oracle"]["recall"]),
                "auc": float(f["auc"]),
                "threshold": float(f["oracle"]["threshold"]),
            }

    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "preparing",
        "training": False,
        "experiment": "Qtable_Localization_Probe",
        "checkpoint": str(E3),
        "purpose": (
            "Keep Q90 RGB/DCT fixed and modify only qtable "
            "to test whether qtable conditioning causes localization collapse."
        ),
    }
    dump("summary.json", summary)

    parent = torch.load(PARENT, map_location="cpu")
    assert int(parent["epoch"]) == 76

    cfg = copy.deepcopy(parent["config"])

    e3 = torch.load(E3, map_location="cpu")

    model = fair.e.build_final_model(cfg["model"]).cuda()
    model.load_state_dict(e3["model"], strict=True)
    model.eval()

    va = fair.e.read_manifest(
        MANIFEST,
        cfg["data"]["root"],
        "val",
    )

    clean = FairDataset(
        va,
        OUT / "clean_cache",
        False,
    )

    q90 = Fixed(clean, 90)

    # Sanity check: re-encoded Q90 qtable should be the canonical Q90 table.
    first_q90 = q90[0]["qtable"].float()

    q90_qtable_diff = float(
        (first_q90 - Q90.float()).abs().max()
    )

    print("canonical Q90 qtable max diff =", q90_qtable_diff, flush=True)

    # ---------------------------------------------------------
    # A. Native Q90
    # RGB = Q90
    # DCT = Q90
    # qtable = Q90
    # ---------------------------------------------------------
    print("\n===== A: native Q90 =====", flush=True)

    z_native = fair.evaluate(
        model,
        q90,
        cfg,
        OUT,
        "q90_native",
    )

    r_native = summarize(z_native)
    dump("progress.json", {"A_q90_native": r_native})
    print(json.dumps(r_native, indent=2), flush=True)

    # ---------------------------------------------------------
    # B. ORACLE DIAGNOSTIC:
    # RGB = Q90
    # DCT = Q90
    # qtable = paired CLEAN qtable
    #
    # This is NOT a deployable method.
    # It tells us whether qtable mismatch causally hurts output.
    # ---------------------------------------------------------
    print("\n===== B: Q90 + paired clean qtable =====", flush=True)

    q90_clean_q = ReplaceQtable(
        image_dataset=q90,
        qtable_dataset=clean,
    )

    z_cleanq = fair.evaluate(
        model,
        q90_clean_q,
        cfg,
        OUT,
        "q90_paired_clean_qtable",
    )

    r_cleanq = summarize(z_cleanq)

    dump(
        "progress.json",
        {
            "A_q90_native": r_native,
            "B_q90_paired_clean_qtable": r_cleanq,
        },
    )

    print(json.dumps(r_cleanq, indent=2), flush=True)

    # ---------------------------------------------------------
    # C. DEPLOYABLE DIAGNOSTIC:
    # RGB = Q90
    # DCT = Q90
    # qtable = fixed canonical Q100
    #
    # Does not use paired clean/source information.
    # ---------------------------------------------------------
    print("\n===== C: Q90 + fixed canonical Q100 qtable =====", flush=True)

    q90_q100 = ReplaceQtable(
        image_dataset=q90,
        fixed_qtable=Q100,
    )

    z_q100 = fair.evaluate(
        model,
        q90_q100,
        cfg,
        OUT,
        "q90_fixed_q100_qtable",
    )

    r_q100 = summarize(z_q100)
    print(json.dumps(r_q100, indent=2), flush=True)

    native_f1 = r_native["f1_best"]
    cleanq_gain = r_cleanq["f1_best"] - native_f1
    q100_gain = r_q100["f1_best"] - native_f1

    # Only if fixed Q100 helps Q90 materially do we spend time
    # evaluating its clean-image cost.
    clean_q100_result = None

    if q100_gain >= 0.05:
        print(
            "\n===== D: clean + fixed canonical Q100 qtable =====",
            flush=True,
        )

        clean_q100 = ReplaceQtable(
            image_dataset=clean,
            fixed_qtable=Q100,
        )

        z_clean_q100 = fair.evaluate(
            model,
            clean_q100,
            cfg,
            OUT,
            "clean_fixed_q100_qtable",
        )

        clean_q100_result = summarize(z_clean_q100)
        print(json.dumps(clean_q100_result, indent=2), flush=True)

    if cleanq_gain >= 0.05 and q100_gain >= 0.05:
        decision = "QTABLE_CAUSAL_AND_FIXED_CANONICAL_PROMISING"
    elif cleanq_gain >= 0.05:
        decision = "QTABLE_CAUSAL_BUT_FIXED_Q100_NOT_ENOUGH"
    else:
        decision = "QTABLE_FEATURE_DRIFT_NOT_SUFFICIENT_TO_EXPLAIN_OUTPUT"

    summary.update(
        status="completed",
        canonical_q90_max_abs_diff=q90_qtable_diff,
        results={
            "A_q90_native": r_native,
            "B_q90_paired_clean_qtable": r_cleanq,
            "C_q90_fixed_q100_qtable": r_q100,
            "D_clean_fixed_q100_qtable": clean_q100_result,
        },
        gains_vs_native_q90={
            "paired_clean_qtable_f1": cleanq_gain,
            "fixed_q100_qtable_f1": q100_gain,
        },
        decision=decision,
    )

    dump("summary.json", summary)

    print("\n===== FINAL =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
