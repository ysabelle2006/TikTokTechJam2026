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
from sklearn.metrics import roc_auc_score

from tiktoktechjam2026 import config
from tiktoktechjam2026.cache_embeddings import load_cache, render_condition
from tiktoktechjam2026.data.datasets import SidDataset
from tiktoktechjam2026.models.detector import Detector


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
    for key, _, _ in config.EVAL_CONDITIONS:
        probs, labels, _ = _probs_for_condition(detector, key, batch_size, paths=paths)
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == labels).mean())
        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
        rows[key] = {"accuracy": acc, "auc": auc}

    clean_acc = rows["clean"]["accuracy"]
    for key in rows:
        rows[key]["drop"] = clean_acc - rows[key]["accuracy"]

    _print_table(variant, rows)
    _write_json(variant, out_json, rows, detector, n_test=len(labels))
    return rows


def _print_table(variant: str, rows: dict):
    title = f"{variant.upper()} ROBUSTNESS SUMMARY"
    print("\n" + title)
    print("=" * 72)
    print(f"{'Condition':<22}{'Accuracy':>12}{'Drop':>12}{'AUC':>12}")
    print("-" * 72)
    for key, _, _ in config.EVAL_CONDITIONS:
        r = rows[key]
        print(f"{key:<22}{r['accuracy']*100:>11.2f}%{r['drop']*100:>11.2f}%{r['auc']:>12.4f}")
    print("-" * 72)
    drops = [rows[k]["drop"] for k, _, _ in config.EVAL_CONDITIONS if k != "clean"]
    print(f"{'avg transformed drop':<22}{'':>12}{np.mean(drops)*100:>11.2f}%")


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
        "clean_accuracy": rows["clean"]["accuracy"],
        "clean_auc": rows["clean"]["auc"],
        "avg_transformed_drop": float(np.mean(drops)),
        "conditions": {
            key: {
                "accuracy": rows[key]["accuracy"],
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
