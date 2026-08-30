"""
Robustness evaluation harness.

Runs a trained detector variant on the clean test set and on every
transform x severity condition in the brief's grid
(config.EVAL_CONDITIONS), and reports, per condition:

    Accuracy   -- threshold 0.5
    Drop       -- clean accuracy minus this condition's accuracy
    AUC        -- ROC AUC of P(fake)

Prints the "<V?> ROBUSTNESS SUMMARY" table and persists every number to
config.RESULT_FILES[variant] (results/v0.json, results/v1.json) -- one
file per roadmap stage, never overwritten.

Spatial embeddings for each condition are read from the offline cache
(cache_embeddings.py). For V1 the frequency-stream input is rebuilt live
from the identical render (cache_embeddings.render_condition), so both
streams always see the same pixels.

CLI:
    python -m tiktoktechjam2026.evaluate --variant v0
    python -m tiktoktechjam2026.evaluate --variant v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from tiktoktechjam2026 import config
from tiktoktechjam2026.cache_embeddings import load_cache, render_condition
from tiktoktechjam2026.data.datasets import SidDataset
from tiktoktechjam2026.models.detector import Detector

# --------------------------------------------------------------------------
# Acc@opt -- reported alongside Acc@0.5 for every condition.
#
# The 0.5 threshold is fixed and dataset-agnostic; Acc@opt fits one scalar
# (the decision threshold) to maximise F1 on a random 50% "calibration" split
# of THIS condition's images, then scores the held-out 50%.
#
# On in-distribution SID this barely moves off 0.5 -- the model is
# well-calibrated in-domain (confidence-ECE ~0.006) -- so Acc@opt ~= Acc@0.5
# here and the column mostly serves as a calibration check. Acc@opt only
# becomes interesting for OUT-of-distribution evaluation
# (evaluate_crossdataset.py), where it is a *calibrated-on-target* number:
# an upper bound on deployable transfer if you were allowed to tune one
# threshold on the target domain -- NOT a zero-shot transfer metric. The
# zero-shot numbers are Acc@0.5 and AUC.
THRESHOLD_NOTE = (
    "Acc@opt = threshold that maximises balanced accuracy, fit on a random "
    "50% split of each condition and scored on the other 50%. In-domain "
    "Acc@opt ~= Acc@0.5 (model is calibrated); the fitted threshold itself "
    "wanders because the objective is flat across a wide range near the "
    "accuracy ceiling. Acc@opt matters off-domain as a calibrated-on-target "
    "upper bound -- it is NOT a zero-shot metric. Zero-shot = Acc@0.5 / AUC."
)


def optimal_threshold(y_true, y_score):
    """
    (threshold, balanced_accuracy) maximising balanced accuracy == Youden's J.

    Candidate thresholds come from the ROC curve (every distinct score),
    not a fixed 0..1 grid: under a severe distribution shift the whole score
    mass can sit below 0.01, and a coarse grid would never find the real
    operating point. Balanced accuracy (not F1) because the reported metric
    is accuracy and the test sets are class-balanced.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = int(np.argmax(tpr - fpr))
    t = float(thr[j])
    if not np.isfinite(t):                      # sklearn puts +inf at index 0
        t = 1.0
    return t, float((tpr[j] + (1.0 - fpr[j])) / 2.0)


def calibrated_accuracy(probs, labels, seed):
    """
    (acc_on_eval_half, threshold). Fit the threshold on a random 50%
    calibration split, score the held-out 50%. Falls back to 0.5 if the
    calibration half is degenerate (one class / too small).
    """
    n = len(labels)
    perm = np.random.default_rng(seed).permutation(n)
    calib, ev = perm[: n // 2], perm[n // 2:]
    if len(calib) >= 2 and len(np.unique(labels[calib])) == 2:
        thr, _ = optimal_threshold(labels[calib], probs[calib])
    else:
        thr = 0.5
    acc = float(((probs[ev] >= thr).astype(int) == labels[ev]).mean())
    return acc, float(thr)


@torch.no_grad()
def _probs_for_condition(detector, condition, batch_size, paths=None):
    """P(fake) for every test image under one condition."""
    from PIL import Image

    emb, labels, cache_paths = load_cache("test", condition)
    emb = torch.from_numpy(np.ascontiguousarray(emb)).float()
    n = len(labels)

    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        emb_b = emb[sl]
        if detector.variant == "v0":
            logits = detector.classify(emb_b)
        else:
            rendered = []
            for i in range(sl.start, sl.stop):
                with Image.open(paths[i]) as im:
                    rendered.append(render_condition(im.convert("RGB"), paths[i], condition))
            freq, energy = detector.frequency_batch(rendered)
            logits = detector.classify(emb_b, freq, energy)
        out[sl] = torch.sigmoid(logits).cpu().numpy()
    return out, labels, cache_paths


def evaluate(variant: str, batch_size: int = None, out_json: str = None):
    batch_size = batch_size or config.BATCH_SIZE
    out_json = out_json or config.RESULT_FILES[variant]

    detector = Detector.from_checkpoint(config.CHECKPOINTS[variant])
    detector.eval()

    # V1 needs the raw images to rebuild frequency inputs; V0 works from cache alone.
    paths = None
    if variant == "v1":
        _, _, cache_paths = load_cache("test", "clean")
        paths = SidDataset("test").paths
        if list(cache_paths) != list(paths):
            raise RuntimeError(
                "test cache is out of sync with the split file -- rerun cache_embeddings.py"
            )

    rows = {}
    for i, (key, _, _) in enumerate(config.EVAL_CONDITIONS):
        probs, labels, _ = _probs_for_condition(detector, key, batch_size, paths=paths)
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == labels).mean())
        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
        acc_opt, thr_opt = calibrated_accuracy(probs, labels, seed=config.SEED + i)
        rows[key] = {
            "accuracy": acc,               # Acc@0.5 (unchanged key, for backwards compat)
            "accuracy_at_opt": acc_opt,
            "opt_threshold": thr_opt,
            "auc": auc,
        }

    clean_acc = rows["clean"]["accuracy"]
    for key in rows:
        rows[key]["drop"] = clean_acc - rows[key]["accuracy"]

    _print_table(variant, rows)
    _write_json(variant, out_json, rows, detector, n_test=len(labels))
    return rows


def _print_table(variant: str, rows: dict):
    title = f"{variant.upper()} ROBUSTNESS SUMMARY"
    print("\n" + title)
    print("=" * 78)
    print(f"{'Condition':<20}{'Acc@0.5':>10}{'Drop':>10}{'Acc@opt':>10}{'Thr':>8}{'AUC':>10}")
    print("-" * 78)
    for key, _, _ in config.EVAL_CONDITIONS:
        r = rows[key]
        print(f"{key:<20}{r['accuracy']*100:>9.2f}%{r['drop']*100:>9.2f}%"
              f"{r['accuracy_at_opt']*100:>9.2f}%{r['opt_threshold']:>8.2f}{r['auc']:>10.4f}")
    print("-" * 78)
    drops = [rows[k]["drop"] for k, _, _ in config.EVAL_CONDITIONS if k != "clean"]
    print(f"{'avg transformed drop':<20}{'':>10}{np.mean(drops)*100:>9.2f}%")
    print(f"\n{THRESHOLD_NOTE}")


def _write_json(variant, out_json, rows, detector, n_test):
    meta = {}
    ckpt = config.CHECKPOINTS[variant]
    if os.path.exists(ckpt):
        meta = torch.load(ckpt, map_location="cpu").get("meta", {})

    drops = [rows[k]["drop"] for k, _, _ in config.EVAL_CONDITIONS if k != "clean"]
    payload = {
        "variant": variant,
        "freq_mode": detector.freq_mode if variant == "v1" else None,
        "dataset": "SID_Set (real vs full_synthetic)",
        "n_test": int(n_test),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_meta": meta,
        "threshold_note": THRESHOLD_NOTE,
        "clean_accuracy": rows["clean"]["accuracy"],
        "clean_accuracy_at_opt": rows["clean"]["accuracy_at_opt"],
        "clean_auc": rows["clean"]["auc"],
        "avg_transformed_drop": float(np.mean(drops)),
        "conditions": {
            key: {
                "accuracy": rows[key]["accuracy"],
                "accuracy_at_0.5": rows[key]["accuracy"],
                "accuracy_at_opt": rows[key]["accuracy_at_opt"],
                "opt_threshold": rows[key]["opt_threshold"],
                "drop": rows[key]["drop"],
                "auc": rows[key]["auc"],
            }
            for key, _, _ in config.EVAL_CONDITIONS
        },
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robustness evaluation for V0 / V1.")
    ap.add_argument("--variant", choices=["v0", "v1"], required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None, help="override results JSON path")
    args = ap.parse_args()
    evaluate(args.variant, args.batch_size, args.out)
