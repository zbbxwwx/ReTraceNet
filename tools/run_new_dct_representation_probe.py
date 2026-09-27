"""Zero-training probe for qtable-free block-DCT representations."""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from run_qtr_localnorm_jpeg_pilot import Fixed

MANIFEST = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "balanced_residual_v3_full30_20260912/clean_manifest.csv"
)

PARENT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "mit_full_offline_20260918/CASIA2_mit_full100/best.pt"
)

OUT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "new_dct_representation_probe_20260922"
)

SEED = 3407
N_PER_GROUP = 64
EPS = 1e-8


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
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


def make_dct_matrix(device="cpu", dtype=torch.float32):
    n = 8
    C = torch.zeros((n, n), device=device, dtype=dtype)

    for k in range(n):
        alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)

        for x in range(n):
            C[k, x] = alpha * math.cos(
                math.pi * (2 * x + 1) * k / (2 * n)
            )

    return C


def zigzag_indices():
    coords = []

    for s in range(15):
        diag = []

        for i in range(8):
            j = s - i

            if 0 <= j < 8:
                diag.append((i, j))

        if s % 2 == 0:
            diag = diag[::-1]

        coords.extend(diag)

    assert len(coords) == 64
    assert coords[0] == (0, 0)

    return coords


ZZ = zigzag_indices()
AC63 = ZZ[1:]
AC20 = ZZ[1:21]


def block_dct_rep(sample):
    """
    decoded RGB
      -> Y
      -> non-overlapping 8x8 DCT
      -> AC coefficients
      -> signed-log
      -> optional per-frequency normalization
    """
    image = sample["image"].float()

    vh, vw = map(int, sample["valid_hw"].tolist())

    h = (vh // 8) * 8
    w = (vw // 8) * 8

    if h < 8 or w < 8:
        raise RuntimeError((vh, vw))

    rgb = image[:, :h, :w]

    y = (
        0.299 * rgb[0]
        + 0.587 * rgb[1]
        + 0.114 * rgb[2]
    )

    # [H/8, W/8, 8, 8]
    blocks = (
        y.unfold(0, 8, 8)
         .unfold(1, 8, 8)
         .contiguous()
    )

    C = make_dct_matrix(blocks.device, blocks.dtype)

    # D = C * block * C^T
    tmp = torch.einsum("ab,ijbc->ijac", C, blocks)
    dct = torch.einsum("ijab,cb->ijac", tmp, C)

    # [64, H/8, W/8]
    freq = dct.permute(2, 3, 0, 1).reshape(64, h // 8, w // 8)

    def select(coords):
        idx = [u * 8 + v for u, v in coords]
        return freq[idx]

    ac63 = select(AC63)
    ac20 = select(AC20)

    def slog(x):
        return torch.sign(x) * torch.log1p(x.abs())

    def pfnorm(x):
        mu = x.mean(dim=(-2, -1), keepdim=True)
        var = (
            (x - mu)
            .square()
            .mean(dim=(-2, -1), keepdim=True)
        )
        return (x - mu) * torch.rsqrt(var + 1e-6)

    ac63_log = slog(ac63)
    ac20_log = slog(ac20)

    return {
        "ac63_log": ac63_log,
        "ac63_log_norm": pfnorm(ac63_log),
        "ac20_log_norm": pfnorm(ac20_log),
    }


def met(a, b):
    x = a.double().reshape(-1)
    y = b.double().reshape(-1)

    nx = torch.linalg.vector_norm(x)
    ny = torch.linalg.vector_norm(y)

    return {
        "cosine": float(
            torch.dot(x, y) / (nx * ny + EPS)
        ),
        "relative_l2": float(
            torch.linalg.vector_norm(y - x) / (nx + EPS)
        ),
        "norm_ratio_q90_over_clean": float(
            ny / (nx + EPS)
        ),
    }


def aggregate(rows):
    out = {}

    variants = (
        "ac63_log",
        "ac63_log_norm",
        "ac20_log_norm",
    )

    for g in ("original_jpeg", "original_nonjpeg"):
        rr = [r for r in rows if r["group"] == g]

        out[g] = {
            "n": len(rr),
            "variants": {},
        }

        for name in variants:
            vals = [r["variants"][name] for r in rr]

            out[g]["variants"][name] = {}

            for k in (
                "cosine",
                "relative_l2",
                "norm_ratio_q90_over_clean",
            ):
                z = np.asarray([v[k] for v in vals])

                out[g]["variants"][name][k] = {
                    "mean": float(z.mean()),
                    "median": float(np.median(z)),
                    "std": float(z.std()),
                }

    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "preparing",
        "training": False,
        "experiment": "New_DCT_Representation_Probe",
        "representation": (
            "decoded RGB -> Y -> 8x8 orthonormal DCT -> "
            "AC -> signed-log -> optional per-frequency normalization"
        ),
        "uses_qtable": False,
        "n_per_group": N_PER_GROUP,
    }

    dump("summary.json", summary)

    ck = torch.load(PARENT, map_location="cpu")
    cfg = ck["config"]

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

        selected += [
            (g, i)
            for i in ids[:N_PER_GROUP]
        ]

    rows = []

    for n, (g, i) in enumerate(selected, 1):
        clean_rep = block_dct_rep(base[i])
        q90_rep = block_dct_rep(q90[i])

        row = {
            "index": i,
            "path": pathof(va[i]),
            "group": g,
            "variants": {
                name: met(clean_rep[name], q90_rep[name])
                for name in clean_rep
            },
        }

        rows.append(row)

        if n == 1 or n % 10 == 0 or n == len(selected):
            print(
                f"[{n:03d}/{len(selected)}] {g}",
                flush=True,
            )

            dump(
                "progress.json",
                {
                    "done": n,
                    "total": len(selected),
                    "group": g,
                },
            )

    agg = aggregate(rows)

    jpeg_cos = agg["original_jpeg"]["variants"][
        "ac63_log_norm"
    ]["cosine"]["mean"]

    nonjpeg_cos = agg["original_nonjpeg"]["variants"][
        "ac63_log_norm"
    ]["cosine"]["mean"]

    # Stability gate only.
    # Passing this does NOT prove localization quality.
    if nonjpeg_cos >= 0.75 and jpeg_cos >= 0.80:
        decision = "PROMISING_TEST_IN_MODEL"
    else:
        decision = "REPRESENTATION_STILL_TOO_UNSTABLE"

    summary.update(
        status="completed",
        aggregate=agg,
        key_result={
            "ac63_log_norm_original_jpeg_cosine": jpeg_cos,
            "ac63_log_norm_original_nonjpeg_cosine": nonjpeg_cos,
        },
        reference_old_cat_input={
            "original_jpeg_cosine": 0.7693180081903375,
            "original_nonjpeg_cosine": 0.5554163066538441,
        },
        decision=decision,
        note=(
            "This probe measures compression stability only; "
            "it does not establish manipulation-localization usefulness."
        ),
    )

    dump("per_image.json", rows)
    dump("summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
