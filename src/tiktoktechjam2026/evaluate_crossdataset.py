"""
Cross-dataset generalization eval.

V0/V1 are trained on SID_Set only (real vs full_synthetic). This harness
runs a trained checkpoint LIVE (no embedding cache) on a dataset it has
never seen, to measure transfer to a different generator family and image
domain. Default target: the CIFAKE test split
(data/cifake/test/{REAL,FAKE}).

Reported per condition:
  Acc@0.5   fixed 0.5 threshold                       -- zero-shot
  AUC       ranking quality, threshold-free           -- zero-shot
  Acc@opt   balanced-accuracy-optimal threshold fit    -- calibrated-on-target
            on a random 50% split of the target,           (NOT zero-shot)
            scored on the other 50%
  P/R/F1    per class at 0.5

Caveat for V1: CIFAKE is 32x32. Upscaled to 224 it has essentially no
genuine high-frequency content, so the frequency stream sees mostly
bicubic-upsampling artifacts -- read V1 cross-dataset numbers with that in
mind (this is why we do not train on CIFAKE; see the roadmap discussion).

CLI:
    python -m tiktoktechjam2026.evaluate_crossdataset --variant v0
    python -m tiktoktechjam2026.evaluate_crossdataset --variant v1 --per-class 2000
    python -m tiktoktechjam2026.evaluate_crossdataset --variant v0 --grid
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from tiktoktechjam2026 import config
from tiktoktechjam2026.cache_embeddings import render_condition
from tiktoktechjam2026.evaluate import calibrated_accuracy, optimal_threshold
from tiktoktechjam2026.models.detector import Detector

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

THRESHOLD_NOTE = (
    "Acc@0.5 and AUC are zero-shot: the SID-trained checkpoint is applied to "
    "CIFAKE with no target-domain tuning. Acc@opt is CALIBRATED ON TARGET -- "
    "the decision threshold is fit to maximise balanced accuracy on a random "
    "50% split of CIFAKE test (using CIFAKE labels), then scored on the "
    "held-out 50%. It answers 'how well could this model do on CIFAKE if "
    "allowed to tune one scalar', an upper bound on deployable transfer -- "
    "not zero-shot."
)


# --------------------------------------------------------------------------
# CIFAKE test loader
# --------------------------------------------------------------------------

def load_cifake_test(per_class: int | None = None, seed: int = None):
    """Return (paths list[str], labels np.int64[N]) for CIFAKE's test split.

    REAL -> 0, FAKE -> 1, matching SID_Set's real/fake convention. CIFAKE is
    never used for training, so its full test split is legitimately held out.
    """
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    root = os.path.join(config.DATA_DIR, "cifake", "test")
    rows = []
    for folder, label in (("REAL", 0), ("FAKE", 1)):
        d = os.path.join(root, folder)
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"{d!r} not found -- fetch CIFAKE first (downloads.py --cifake)"
            )
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS))
        rng.shuffle(files)
        if per_class:
            files = files[:per_class]
        rows += [(os.path.join(d, f), label) for f in files]
    rng.shuffle(rows)
    return [p for p, _ in rows], np.array([y for _, y in rows], dtype=np.int64)


# --------------------------------------------------------------------------
# Live inference
# --------------------------------------------------------------------------

@torch.no_grad()
def _live_probs(detector: Detector, paths, condition: str, batch_size: int):
    """P(fake) for every path, run live. `condition` != 'clean' renders the
    transform per image first (same per-image seed as the SID cache)."""
    detector.eval()
    n = len(paths)
    probs = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
            imgs.append(im if condition == "clean" else render_condition(im, p, condition))
        probs[start:start + len(imgs)] = detector.predict_proba(imgs)
        done = start + len(imgs)
        print(f"\r    {condition:16s} {done}/{n}  ({done / (time.time() - t0):.1f} img/s)",
              end="", flush=True)
    print()
    return probs


def _metrics(probs, labels, seed):
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    acc_opt, thr_opt = calibrated_accuracy(probs, labels, seed)
    # single-threshold optimum on the whole set, for reference
    thr_full, _ = optimal_threshold(labels, probs)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return {
        "accuracy_at_0.5": acc,
        "accuracy_at_opt": acc_opt,
        "opt_threshold": thr_opt,
        "opt_threshold_full_set": float(thr_full),
        "auc": auc,
        "per_class": {
            "real": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])},
            "fake": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1])},
        },
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def evaluate_crossdataset(variant: str, per_class: int = None, grid: bool = False,
                          batch_size: int = None, out_json: str = None):
    batch_size = batch_size or config.BATCH_SIZE
    out_json = out_json or os.path.join(config.RESULTS_DIR, f"{variant}_cifake.json")

    detector = Detector.from_checkpoint(config.CHECKPOINTS[variant])
    ckpt_meta = torch.load(config.CHECKPOINTS[variant], map_location="cpu").get("meta", {})

    paths, labels = load_cifake_test(per_class)
    print(f"[{variant}] CIFAKE test: {len(paths)} images "
          f"({int((labels == 0).sum())} real / {int((labels == 1).sum())} fake), live inference")

    conditions = [k for k, _, _ in config.EVAL_CONDITIONS] if grid else ["clean"]
    rows = {}
    raw = {"labels": labels}
    for i, key in enumerate(conditions):
        probs = _live_probs(detector, paths, key, batch_size)
        raw[key] = probs
        rows[key] = _metrics(probs, labels, seed=config.SEED + i)

    # Persist the raw per-image probabilities so metrics can be recomputed
    # (different threshold rule, calibration, bootstrap CI) without re-running
    # 20k live forward passes.
    probs_path = os.path.splitext(out_json)[0] + "_probs.npz"
    np.savez_compressed(probs_path, **raw)

    if grid:
        clean_acc = rows["clean"]["accuracy_at_0.5"]
        for key in rows:
            rows[key]["drop"] = clean_acc - rows[key]["accuracy_at_0.5"]

    _print_table(variant, rows, conditions)

    payload = {
        "variant": variant,
        "freq_mode": detector.freq_mode if variant == "v1" else None,
        "dataset": "CIFAKE test (data/cifake/test)",
        "trained_on": "SID_Set real vs full_synthetic -- CIFAKE never seen in training",
        "n": len(paths),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_meta": ckpt_meta,
        "threshold_note": THRESHOLD_NOTE,
        "ran_transform_grid": grid,
        "clean": rows["clean"],
        "conditions": rows if grid else None,
    }
    if variant == "v1":
        payload["v1_caveat"] = (
            "CIFAKE is 32px; the frequency stream sees mostly upscaling "
            "artifacts, not generator residuals."
        )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_json}  (+ raw probs: {probs_path})")
    return rows


def _print_table(variant, rows, conditions):
    print(f"\n{variant.upper()} CROSS-DATASET (CIFAKE test)")
    print("=" * 78)
    print(f"{'Condition':<20}{'Acc@0.5':>10}{'AUC':>10}{'Acc@opt':>10}{'Thr':>8}"
          f"{'F1 real':>10}{'F1 fake':>10}")
    print("-" * 78)
    for key in conditions:
        r = rows[key]
        print(f"{key:<20}{r['accuracy_at_0.5']*100:>9.2f}%{r['auc']:>10.4f}"
              f"{r['accuracy_at_opt']*100:>9.2f}%{r['opt_threshold']:>8.2f}"
              f"{r['per_class']['real']['f1']:>10.3f}{r['per_class']['fake']['f1']:>10.3f}")
    print("-" * 78)
    print(f"\n{THRESHOLD_NOTE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-dataset (CIFAKE) evaluation for V0/V1.")
    ap.add_argument("--variant", choices=["v0", "v1"], required=True)
    ap.add_argument("--per-class", type=int, default=None,
                    help="cap images per class (default: all ~10k)")
    ap.add_argument("--grid", action="store_true",
                    help="also run the full transform grid (slow; note the 32px caveat)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="override results JSON path")
    args = ap.parse_args()
    evaluate_crossdataset(args.variant, args.per_class, args.grid, args.batch_size, args.out)
