import copy
import json
import random
import sys
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from run_qtr_localnorm_jpeg_pilot import Fixed

PARENT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "mit_full_offline_20260918/CASIA2_mit_full100/best.pt"
)

MANIFEST = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "balanced_residual_v3_full30_20260912/clean_manifest.csv"
)

OUT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "dct_factorization_probe_20260922"
)

N_PER_GROUP = 64
SEED = 3407
EPS = 1e-8


def dump(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, indent=2, allow_nan=False)
    )


def pathof(r):
    for k in ("image", "image_path", "path", "rgb_path"):
        v = getattr(r, k, None)
        if v:
            return str(v)
    return str(r)


def group(r):
    return (
        "original_jpeg"
        if Path(pathof(r)).suffix.lower() in {".jpg", ".jpeg", ".jpe"}
        else "original_nonjpeg"
    )


def batch(s):
    return {
        k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else [v])
        for k, v in s.items()
    }


def metric(a, b):
    x = a.detach().double().reshape(-1)
    y = b.detach().double().reshape(-1)

    nx = torch.linalg.vector_norm(x)
    ny = torch.linalg.vector_norm(y)

    return {
        "cosine": float(torch.dot(x, y) / (nx * ny + EPS)),
        "relative_l2": float(
            torch.linalg.vector_norm(y - x) / (nx + EPS)
        ),
        "norm_ratio_q90_over_clean": float(ny / (nx + EPS)),
    }


def branch_forward(branch, dct, qtable):
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        out = branch(dct, qtable)

    return [x.detach().float().cpu() for x in out]


def stem_forward(branch, dct):
    """
    Feature immediately before qtable is introduced:
      dc_layer0_dil -> dc_layer1_tail
    """
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        x = branch.dc_layer0_dil(dct.float())
        x = branch.dc_layer1_tail(x)

    return x.detach().float().cpu()


def aggregate(rows):
    out = {}

    for g in ("original_jpeg", "original_nonjpeg"):
        rr = [r for r in rows if r["group"] == g]

        out[g] = {
            "n": len(rr),
            "dct_input": {},
            "pre_qtable_stem": {},
            "conditions": {},
            "qtable_only": {},
        }

        for field in ("dct_input", "pre_qtable_stem", "qtable_only"):
            vals = [r[field] for r in rr]
            for k in (
                "cosine",
                "relative_l2",
                "norm_ratio_q90_over_clean",
            ):
                z = np.asarray([v[k] for v in vals])
                out[g][field][k] = {
                    "mean": float(z.mean()),
                    "median": float(np.median(z)),
                    "std": float(z.std()),
                }

        for cond in (
            "A_native",
            "B_shared_q90",
            "C_shared_clean",
        ):
            out[g]["conditions"][cond] = {}

            for j in range(4):
                vals = [
                    r["conditions"][cond][f"D{j+1}"]
                    for r in rr
                ]

                out[g]["conditions"][cond][f"D{j+1}"] = {}

                for k in (
                    "cosine",
                    "relative_l2",
                    "norm_ratio_q90_over_clean",
                ):
                    z = np.asarray([v[k] for v in vals])

                    out[g]["conditions"][cond][f"D{j+1}"][k] = {
                        "mean": float(z.mean()),
                        "median": float(np.median(z)),
                        "std": float(z.std()),
                    }

    return out


def mean_cosine(agg, group_name, condition):
    return float(
        np.mean(
            [
                agg[group_name]["conditions"][condition][f"D{i}"]
                ["cosine"]["mean"]
                for i in range(1, 5)
            ]
        )
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "preparing",
        "training": False,
        "experiment": "DCT_Factorization_Probe",
        "n_per_group": N_PER_GROUP,
    }

    dump("summary.json", summary)

    try:
        ck = torch.load(PARENT, map_location="cpu")
        assert int(ck["epoch"]) == 76

        cfg = copy.deepcopy(ck["config"])

        model = fair.e.build_final_model(cfg["model"]).cuda()
        model.load_state_dict(ck["model"], strict=True)
        model.eval()

        branch = model.dct_branch

        va = fair.e.read_manifest(
            MANIFEST,
            cfg["data"]["root"],
            "val",
        )

        base = FairDataset(
            va,
            OUT / "clean_cache",
            False,
        )

        q90 = Fixed(base, 90)

        by = defaultdict(list)

        for i, r in enumerate(va):
            by[group(r)].append(i)

        rng = random.Random(SEED)
        selected = []

        for g in ("original_jpeg", "original_nonjpeg"):
            ids = by[g][:]
            rng.shuffle(ids)
            selected += [(g, i) for i in ids[:N_PER_GROUP]]

        rows = []

        for n, (g, i) in enumerate(selected, 1):
            clean_s = base[i]
            q90_s = q90[i]

            a = batch(clean_s)
            b = batch(q90_s)

            dc = a["dct"]
            dq = b["dct"]

            qc = a["qtable"]
            qq = b["qtable"]

            # --------------------------------------------------
            # 0. Raw 21-channel DCT volume drift
            # --------------------------------------------------
            dct_input = metric(
                dc.detach().float().cpu(),
                dq.detach().float().cpu(),
            )

            # --------------------------------------------------
            # 1. Drift before qtable enters the DCT encoder
            # --------------------------------------------------
            sc = stem_forward(branch, dc)
            sq = stem_forward(branch, dq)

            pre_qtable_stem = metric(sc, sq)

            # --------------------------------------------------
            # A. Native condition
            # clean: Enc(DCTclean, Qclean)
            # q90:   Enc(DCTq90,   Qq90)
            # --------------------------------------------------
            ac = branch_forward(branch, dc, qc)
            aq = branch_forward(branch, dq, qq)

            # --------------------------------------------------
            # B. Shared Q90 qtable
            # clean: Enc(DCTclean, Qq90)
            # q90:   Enc(DCTq90,   Qq90)
            # --------------------------------------------------
            bc = branch_forward(branch, dc, qq)
            bq = branch_forward(branch, dq, qq)

            # --------------------------------------------------
            # C. Shared clean qtable
            # clean: Enc(DCTclean, Qclean)
            # q90:   Enc(DCTq90,   Qclean)
            # --------------------------------------------------
            cc = branch_forward(branch, dc, qc)
            cq = branch_forward(branch, dq, qc)

            # --------------------------------------------------
            # D. qtable-only sensitivity
            # Same clean DCT volume, only qtable changes
            # --------------------------------------------------
            qc_only = branch_forward(branch, dc, qc)
            qq_only = branch_forward(branch, dc, qq)

            row = {
                "index": i,
                "path": pathof(va[i]),
                "group": g,
                "dct_input": dct_input,
                "pre_qtable_stem": pre_qtable_stem,
                "conditions": {
                    "A_native": {
                        f"D{j+1}": metric(ac[j], aq[j])
                        for j in range(4)
                    },
                    "B_shared_q90": {
                        f"D{j+1}": metric(bc[j], bq[j])
                        for j in range(4)
                    },
                    "C_shared_clean": {
                        f"D{j+1}": metric(cc[j], cq[j])
                        for j in range(4)
                    },
                },
                "qtable_only": {
                    # summarize qtable-only sensitivity by D1-D4 mean later
                    "cosine": float(
                        np.mean(
                            [
                                metric(qc_only[j], qq_only[j])["cosine"]
                                for j in range(4)
                            ]
                        )
                    ),
                    "relative_l2": float(
                        np.mean(
                            [
                                metric(qc_only[j], qq_only[j])["relative_l2"]
                                for j in range(4)
                            ]
                        )
                    ),
                    "norm_ratio_q90_over_clean": float(
                        np.mean(
                            [
                                metric(qc_only[j], qq_only[j])
                                ["norm_ratio_q90_over_clean"]
                                for j in range(4)
                            ]
                        )
                    ),
                },
            }

            rows.append(row)

            if n == 1 or n % 10 == 0 or n == len(selected):
                dump(
                    "progress.json",
                    {
                        "done": n,
                        "total": len(selected),
                        "group": g,
                    },
                )

                print(
                    f"[{n:03d}/{len(selected)}] {g}",
                    flush=True,
                )

        agg = aggregate(rows)

        a = mean_cosine(
            agg,
            "original_nonjpeg",
            "A_native",
        )

        b = mean_cosine(
            agg,
            "original_nonjpeg",
            "B_shared_q90",
        )

        c = mean_cosine(
            agg,
            "original_nonjpeg",
            "C_shared_clean",
        )

        best_shared = max(b, c)
        shared_gain = best_shared - a

        decision = (
            "QTABLE_MAJOR_CONTRIBUTOR"
            if shared_gain >= 0.15
            else "DCTVOL_MAJOR_CONTRIBUTOR"
        )

        summary.update(
            status="completed",
            aggregate=agg,
            original_nonjpeg_mean_cosine={
                "A_native": a,
                "B_shared_q90": b,
                "C_shared_clean": c,
                "best_shared": best_shared,
                "shared_gain_vs_native": shared_gain,
            },
            decision=decision,
            interpretation={
                "QTABLE_MAJOR_CONTRIBUTOR":
                    "shared qtable improves mean D1-D4 cosine by >=0.15",
                "DCTVOL_MAJOR_CONTRIBUTOR":
                    "shared qtable does not improve mean D1-D4 cosine by >=0.15",
            }[decision],
            artifacts={
                "per_image": str(OUT / "per_image.json"),
            },
        )

        dump("per_image.json", rows)
        dump("summary.json", summary)

        print(json.dumps(summary, indent=2))

    except BaseException:
        summary.update(
            status="failed",
            traceback=traceback.format_exc(),
        )

        dump("summary.json", summary)
        raise


if __name__ == "__main__":
    main()
