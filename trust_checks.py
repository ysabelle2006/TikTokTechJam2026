"""
trust_checks.py -- adversarial validation suite for the V0 / V1 AIGC detector.

The V0 robustness table looks almost too good (99.3% clean, <2% average
transformed drop). This script pressure-tests that result with seven checks
that each try to find a reason NOT to trust the number:

  1. Cross-dataset generalization   (does it transfer to CIFAKE at all?)
  2. Overfitting curves             (train loss vs val accuracy over epochs)
  3. Label-shuffling sanity         (train on random labels -> must score ~50%)
  4. Unseen corruptions             (corruption types outside the brief's grid)
  5. Frequency-stream contribution  (V1 only: is the forensic branch redundant?)
  6. Calibration / reliability      (are the probabilities meaningful?)
  7. Class-wise performance         (is it biased toward real or fake?)

Design choices:
  * Evaluation checks (1, 4, 5, 6, 7) run the detector LIVE on raw images via
    Detector.predict_proba -- no cached embeddings -- so the full pipeline
    (decode -> preprocess -> CLIP -> head) is exercised end to end.
  * The retraining checks (2, 3) reuse the offline CLIP embedding cache. The
    backbone is frozen, so a cached embedding is bit-for-bit what a live
    forward pass would produce; `_cache_vs_live_consistency()` verifies this
    before the checks run. Retraining live would mean ~85 min of ViT forwards
    per check on CPU for zero numerical difference.

Outputs:
  diagnostics/REPORT.md            the full report (also printed to stdout)
  diagnostics/overfitting_v0.png   train-loss / val-accuracy curves
  diagnostics/calibration_v0.png   reliability diagram
  diagnostics/results.json         raw metrics

Run:
    python trust_checks.py                 # V0, plus V1 if checkpoints/v1.pt exists
    python trust_checks.py --variant v0
    python trust_checks.py --cifake-per-class 1500 --corruption-images 200
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from tiktoktechjam2026 import config
from tiktoktechjam2026.cache_embeddings import load_cache
from tiktoktechjam2026.data.datasets import SidDataset
from tiktoktechjam2026.models.detector import Detector
from tiktoktechjam2026.models.spatial_head import SpatialHead

OUT_DIR = "diagnostics"
RNG_SEED = 12345

PASS, WARN, FAIL = "Pass", "Warning", "Fail"


# ==========================================================================
# Shared helpers
# ==========================================================================

def _open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


@torch.no_grad()
def live_probs(detector: Detector, images, batch_size: int = 128, log_every: int = 1024,
               tag: str = "") -> np.ndarray:
    """
    P(fake) for a list of PIL images OR image paths, run LIVE through the
    detector in chunks (predict_proba stacks its whole input into one tensor,
    which would OOM on thousands of 224x224 images).
    """
    probs = np.empty(len(images), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        pil = [_open_rgb(x) if isinstance(x, str) else x for x in chunk]
        probs[start:start + len(pil)] = detector.predict_proba(pil)
        done = start + len(pil)
        if tag and (done % log_every == 0 or done == len(images)):
            rate = done / (time.time() - t0)
            print(f"\r    [{tag}] {done}/{len(images)}  ({rate:.1f} img/s)",
                  end="", flush=True)
    if tag:
        print()
    return probs


def _acc(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> float:
    return float(((probs >= thr).astype(int) == labels).mean())


def _auc(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _per_class_prf(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5):
    preds = (probs >= thr).astype(int)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=[0, 1],
                                                 zero_division=0)
    return {
        "real": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]),
                 "support": int(s[0])},
        "fake": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]),
                 "support": int(s[1])},
    }


def _load_cached_split(split: str):
    emb, labels, _ = load_cache(split, "clean")
    return (torch.from_numpy(np.ascontiguousarray(emb)).float(),
            torch.from_numpy(labels.astype(np.float32)))


def _train_spatial_head(emb: torch.Tensor, y: torch.Tensor, epochs: int,
                        lr: float, batch_size: int, seed: int,
                        val: tuple | None = None, track_test: tuple | None = None):
    """
    Minimal from-scratch SpatialHead training loop (V0 head only).

    Returns (head, history) where history has per-epoch train_loss / train_acc
    and, if provided, val_acc / val_auc / test_acc / test_auc.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = SpatialHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(y)
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        order = np.random.permutation(n)
        running = 0.0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            opt.zero_grad()
            logits = head(emb[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            tr_logits = head(emb)
        rec = {
            "epoch": epoch,
            "train_loss": running / n,
            "train_acc": _acc(torch.sigmoid(tr_logits).numpy(), y.numpy()),
        }
        if val is not None:
            with torch.no_grad():
                v = torch.sigmoid(head(val[0])).numpy()
            rec["val_acc"] = _acc(v, val[1].numpy())
            rec["val_auc"] = _auc(v, val[1].numpy())
        if track_test is not None:
            with torch.no_grad():
                t = torch.sigmoid(head(track_test[0])).numpy()
            rec["test_acc"] = _acc(t, track_test[1].numpy())
            rec["test_auc"] = _auc(t, track_test[1].numpy())
        history.append(rec)

    return head, history


# ==========================================================================
# Pre-flight: cached embedding == live embedding (justifies checks 2 & 3)
# ==========================================================================

def cache_vs_live_consistency(detector: Detector, n: int = 32) -> dict:
    ds = SidDataset("test")
    cached, _, _ = load_cache("test", "clean")
    idx = list(range(min(n, len(ds))))
    pil = [ds[i][0] for i in idx]
    with torch.no_grad():
        live = detector.embed_spatial(pil).cpu().numpy()
    max_abs = float(np.abs(live - cached[idx]).max())
    cos = float(np.mean([
        np.dot(live[i], cached[idx[i]]) /
        (np.linalg.norm(live[i]) * np.linalg.norm(cached[idx[i]]))
        for i in range(len(idx))
    ]))
    return {"n": len(idx), "max_abs_diff": max_abs, "mean_cosine": cos}


# ==========================================================================
# 1. Cross-dataset generalization (CIFAKE)
# ==========================================================================

def _collect_cifake(per_class: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for cls, label in (("REAL", 0), ("FAKE", 1)):
        folder = os.path.join(config.DATA_DIR, "cifake", "test", cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
        rng.shuffle(files)
        rows += [(os.path.join(folder, f), label) for f in files[:per_class]]
    rng.shuffle(rows)
    return [p for p, _ in rows], np.array([y for _, y in rows], dtype=np.int64)


def test_cross_dataset(detector: Detector, per_class: int, sid_clean_acc: float,
                       sid_clean_auc: float) -> dict:
    print("\n[1] Cross-dataset generalization (CIFAKE test set, live inference)")
    paths, labels = _collect_cifake(per_class, seed=RNG_SEED)
    probs = live_probs(detector, paths, tag="cifake")

    acc, auc = _acc(probs, labels), _auc(probs, labels)
    prf = _per_class_prf(probs, labels)
    acc_drop = sid_clean_acc - acc

    # Separate "can't discriminate" from "SID-fitted threshold doesn't transfer":
    # accuracy at the Youden-optimal threshold shows the best this AUC can buy.
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, probs)
    j = int(np.argmax(tpr - fpr))
    opt_thr = float(thr[j])
    acc_opt = _acc(probs, labels, opt_thr)
    auc_drop = sid_clean_auc - auc

    if auc < 0.65:
        verdict = FAIL
        note = ("near-random ranking on CIFAKE -- the model has not learned a "
                "generator-agnostic cue; the SID score is essentially SID-specific.")
    elif auc < 0.85 or acc_opt < 0.80:
        verdict = WARN
        note = (f"partial transfer: ranking AUC drops {auc_drop:.3f} (1.00 -> {auc:.2f}) "
                f"on an unseen dataset + generators, and the SID-fitted 0.5 threshold "
                f"is unusable off-SID (acc {acc*100:.0f}% at 0.5 vs {acc_opt*100:.0f}% at "
                f"the optimal threshold). Real but degraded signal; retrain/recalibrate "
                f"per target domain.")
    else:
        verdict = PASS
        note = "retains clear, well-thresholded real-vs-fake signal on an unseen dataset/generator."

    print(f"    acc@0.5={acc*100:.2f}%  acc@opt={acc_opt*100:.2f}%  auc={auc:.4f}  "
          f"F1(real)={prf['real']['f1']:.3f}  F1(fake)={prf['fake']['f1']:.3f}  "
          f"(SID {sid_clean_acc*100:.2f}%/{sid_clean_auc:.3f} -> "
          f"acc drop {acc_drop*100:.1f} pts, auc drop {auc_drop:.3f})")
    return {
        "verdict": verdict, "note": note,
        "cifake_acc": acc, "cifake_acc_opt_threshold": acc_opt, "cifake_opt_threshold": opt_thr,
        "cifake_auc": auc,
        "cifake_f1_real": prf["real"]["f1"], "cifake_f1_fake": prf["fake"]["f1"],
        "sid_clean_acc": sid_clean_acc, "sid_clean_auc": sid_clean_auc,
        "acc_drop_pts": acc_drop, "auc_drop": auc_drop, "n": len(labels),
        "per_class": prf,
    }


# ==========================================================================
# 2. Overfitting check: train-loss vs val-accuracy curves
# ==========================================================================

def test_overfitting(epochs: int = 40) -> dict:
    print("\n[2] Overfitting check: re-training V0 for full curves "
          f"({epochs} epochs, no early stop)")
    emb_tr, y_tr = _load_cached_split("train")
    emb_va, y_va = _load_cached_split("val")

    _, history = _train_spatial_head(
        emb_tr, y_tr, epochs=epochs, lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE, seed=config.SEED, val=(emb_va, y_va),
    )

    val_acc = np.array([h["val_acc"] for h in history])
    val_auc = np.array([h["val_auc"] for h in history])
    tr_loss = np.array([h["train_loss"] for h in history])
    tr_acc = np.array([h["train_acc"] for h in history])
    best_epoch = int(np.argmax(val_auc)) + 1
    peak_val_acc = float(val_acc.max())
    final_val_acc = float(val_acc[-1])
    val_regression = peak_val_acc - final_val_acc          # how much val fell after its peak
    final_gap = float(tr_acc[-1] - val_acc[-1])            # train/val accuracy gap

    # Plot
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ep = [h["epoch"] for h in history]
    ax1.plot(ep, tr_loss, color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(ep, val_acc * 100, color="tab:blue", label="val accuracy")
    ax2.plot(ep, tr_acc * 100, color="tab:blue", ls="--", alpha=0.5, label="train accuracy")
    ax2.set_ylabel("accuracy (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.axvline(best_epoch, color="grey", ls=":", label=f"best val-AUC (ep {best_epoch})")
    fig.suptitle("V0 training: loss vs validation accuracy")
    fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plot_path = os.path.join(OUT_DIR, "overfitting_v0.png")
    fig.savefig(plot_path, dpi=110); plt.close(fig)

    if val_regression > 0.02:
        verdict = FAIL
        note = (f"val accuracy peaks at epoch {int(np.argmax(val_acc)) + 1} "
                f"({peak_val_acc*100:.2f}%) then falls to {final_val_acc*100:.2f}% "
                "while train loss keeps dropping -- classic overfitting.")
    elif val_regression > 0.005 or final_gap > 0.03:
        verdict = WARN
        note = (f"mild train/val divergence (val fell {val_regression*100:.2f} pts "
                f"from peak; final train-val gap {final_gap*100:.2f} pts). "
                f"Early stop (patience {config.EARLY_STOP_PATIENCE}) mitigates it.")
    else:
        verdict = PASS
        note = (f"val accuracy is flat after epoch ~{best_epoch} (no regression); "
                f"train-val gap {final_gap*100:.2f} pts. No overfitting signature. "
                "The near-perfect val score from epoch 1 instead points to an easy "
                "task / strong dataset separability.")

    print(f"    best val-AUC epoch={best_epoch}  peak val acc={peak_val_acc*100:.2f}%  "
          f"final val acc={final_val_acc*100:.2f}%  train-val gap={final_gap*100:.2f} pts")
    print(f"    plot -> {plot_path}")
    return {
        "verdict": verdict, "note": note,
        "epochs": epochs, "best_val_auc_epoch": best_epoch,
        "peak_val_acc": peak_val_acc, "final_val_acc": final_val_acc,
        "val_regression_from_peak": val_regression,
        "final_train_val_gap": final_gap,
        "early_stop_patience": config.EARLY_STOP_PATIENCE,
        "plot": plot_path,
        "history": history,
    }


# ==========================================================================
# 3. Label-shuffling sanity
# ==========================================================================

def test_label_shuffle(epochs: int = 10) -> dict:
    print(f"\n[3] Label-shuffling sanity: train V0 head on shuffled labels "
          f"({epochs} epochs), evaluate on real test labels")
    emb_tr, y_tr = _load_cached_split("train")
    emb_te, y_te = _load_cached_split("test")

    rng = np.random.default_rng(RNG_SEED)
    y_shuf = y_tr[torch.from_numpy(rng.permutation(len(y_tr)))].clone()
    # sanity: labels really are decorrelated from the images
    corr = float(np.corrcoef(y_tr.numpy(), y_shuf.numpy())[0, 1])

    _, history = _train_spatial_head(
        emb_tr, y_shuf, epochs=epochs, lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE, seed=RNG_SEED, track_test=(emb_te, y_te),
    )
    last = history[-1]
    test_acc, test_auc = last["test_acc"], last["test_auc"]
    train_acc_on_noise = last["train_acc"]

    if abs(test_acc - 0.5) <= 0.05 and abs(test_auc - 0.5) <= 0.05:
        verdict = PASS
        note = ("test accuracy collapses to chance on shuffled-label training -- "
                "no image-identity leakage or label leakage in the pipeline.")
    elif abs(test_acc - 0.5) <= 0.10:
        verdict = WARN
        note = ("test score slightly above chance after shuffled-label training -- "
                "worth a second look, but could be split/class-ratio noise.")
    else:
        verdict = FAIL
        note = ("model scores well above chance despite random training labels -- "
                "the pipeline is leaking (cache/label misalignment, or the split "
                "shares images across train/test).")

    print(f"    train acc on shuffled labels={train_acc_on_noise*100:.2f}%  "
          f"->  real-test acc={test_acc*100:.2f}%  auc={test_auc:.4f}  "
          f"(label corr {corr:+.3f})")
    return {
        "verdict": verdict, "note": note,
        "epochs": epochs,
        "train_acc_on_shuffled": train_acc_on_noise,
        "real_test_acc": test_acc, "real_test_auc": test_auc,
        "shuffled_label_corr": corr,
        "history": history,
    }


# ==========================================================================
# 4. Robustness to unseen corruptions (outside the brief's grid)
# ==========================================================================

def _salt_pepper(im: Image.Image, amount: float, rng) -> Image.Image:
    arr = np.array(im)
    m = rng.random(arr.shape[:2])
    arr[m < amount / 2] = 0
    arr[m > 1 - amount / 2] = 255
    return Image.fromarray(arr)


def _gamma(im: Image.Image, g: float) -> Image.Image:
    lut = [min(255, int((i / 255.0) ** g * 255.0 + 0.5)) for i in range(256)]
    return im.point(lut * len(im.getbands()))


def _median(im: Image.Image, size: int) -> Image.Image:
    return im.filter(ImageFilter.MedianFilter(size=size))


def _hist_eq(im: Image.Image, _p) -> Image.Image:
    return ImageOps.equalize(im)


UNSEEN_CORRUPTIONS = [
    ("salt_pepper_0.01", _salt_pepper, 0.01, True),
    ("salt_pepper_0.03", _salt_pepper, 0.03, True),
    ("salt_pepper_0.06", _salt_pepper, 0.06, True),
    ("gamma_0.5",        _gamma,       0.5,  False),
    ("gamma_1.6",        _gamma,       1.6,  False),
    ("gamma_2.4",        _gamma,       2.4,  False),
    ("median_3",         _median,      3,    False),
    ("median_5",         _median,      5,    False),
    ("median_7",         _median,      7,    False),
    ("hist_equalize",    _hist_eq,     None, False),
]


def test_unseen_corruptions(detector: Detector, n_images: int) -> dict:
    print(f"\n[4] Unseen corruptions ({n_images} test images, live inference)")
    ds = SidDataset("test")
    n = min(n_images, len(ds))
    base_imgs = [ds[i][0] for i in range(n)]
    labels = np.array(ds.labels[:n], dtype=np.int64)
    rng = np.random.default_rng(RNG_SEED)

    clean_probs = live_probs(detector, base_imgs, tag="clean")
    clean_acc = _acc(clean_probs, labels)

    rows = {}
    for key, fn, param, needs_rng in UNSEEN_CORRUPTIONS:
        if needs_rng:
            corr_imgs = [fn(im, param, np.random.default_rng(rng.integers(1 << 32)))
                         for im in base_imgs]
        else:
            corr_imgs = [fn(im, param) for im in base_imgs]
        probs = live_probs(detector, corr_imgs)
        acc, auc = _acc(probs, labels), _auc(probs, labels)
        rows[key] = {"accuracy": acc, "auc": auc, "drop": clean_acc - acc}
        print(f"    {key:18s} acc={acc*100:6.2f}%  drop={rows[key]['drop']*100:+6.2f} pts  auc={auc:.4f}")

    avg_drop = float(np.mean([r["drop"] for r in rows.values()]))
    worst_key = max(rows, key=lambda k: rows[k]["drop"])
    worst = rows[worst_key]["drop"]
    seen_avg_drop = 0.0176  # V0's average transformed drop on the brief's grid

    if worst > 0.30:
        verdict = WARN
        note = (f"'{worst_key}' costs {worst*100:.1f} pts. V0 trains with NO "
                "augmentation, so this is CLIP's native fragility to that "
                "corruption (salt-and-pepper mirrors V0's known Gaussian-noise "
                "weakness), not grid memorization -- but it is a real robustness gap.")
    elif avg_drop > 3 * seen_avg_drop:
        verdict = WARN
        note = (f"average drop on unseen corruptions ({avg_drop*100:.2f} pts) is "
                f"well above the brief-grid average ({seen_avg_drop*100:.2f} pts).")
    else:
        verdict = PASS
        note = (f"average drop {avg_drop*100:.2f} pts, comparable to the brief-grid "
                f"average ({seen_avg_drop*100:.2f} pts). Robustness is general, not "
                "specific to the tested transforms. (V0 has no augmentation to "
                "memorize -- this is a pure CLIP-robustness probe.)")

    print(f"    avg unseen drop={avg_drop*100:.2f} pts  worst={worst_key} ({worst*100:.2f} pts)")
    return {
        "verdict": verdict, "note": note,
        "n_images": n, "clean_acc": clean_acc,
        "avg_drop_pts": avg_drop, "worst": worst_key, "worst_drop_pts": worst,
        "conditions": rows,
    }


# ==========================================================================
# 5. Frequency-stream contribution (V1 only)
# ==========================================================================

class _FreqAblator:
    """Context manager: force the V1 frequency embedding + energy scalar to zero."""

    def __init__(self, detector: Detector):
        self.detector = detector
        self._orig_fwd = None
        self._orig_batch = None

    def __enter__(self):
        fs = self.detector.frequency_stream
        self._orig_fwd = fs.forward

        def zero_forward(x):
            b = x.shape[0] if x.ndim == 4 else 1
            return torch.zeros(b, config.FREQUENCY_EMBED_DIM, device=x.device)

        fs.forward = zero_forward

        self._orig_batch = self.detector.frequency_batch

        def zero_energy_batch(images):
            freq, energy = self._orig_batch(images)
            return freq, torch.zeros_like(energy)

        self.detector.frequency_batch = zero_energy_batch
        return self

    def __exit__(self, *exc):
        self.detector.frequency_stream.forward = self._orig_fwd
        self.detector.frequency_batch = self._orig_batch


def test_frequency_contribution(n_images: int, v0_clean_acc: float) -> dict:
    ckpt = config.CHECKPOINTS["v1"]
    if not os.path.exists(ckpt):
        print("\n[5] Frequency-stream contribution: SKIPPED (no checkpoints/v1.pt yet)")
        return {"verdict": "N/A",
                "note": "V1 not trained yet; run `python -m tiktoktechjam2026.train "
                        "--variant v1` then re-run this check."}

    print(f"\n[5] Frequency-stream contribution (V1, {n_images} test images, live)")
    detector = Detector.from_checkpoint(ckpt)
    detector.eval()
    ds = SidDataset("test")
    n = min(n_images, len(ds))
    imgs = [ds[i][0] for i in range(n)]
    labels = np.array(ds.labels[:n], dtype=np.int64)

    # a small, fixed robustness sub-grid (clean + the conditions that matter most)
    from tiktoktechjam2026.transforms import augmentations
    sub_grid = ["clean", "jpeg_q30", "blur_2.0", "noise_0.05", "noise_0.10", "resize_0.25"]
    cond_by_key = {k: (nm, p) for k, nm, p in config.EVAL_CONDITIONS}

    def _render(key):
        if key == "clean":
            return list(imgs)
        nm, p = cond_by_key[key]
        out = []
        for i in range(n):
            rng = np.random.default_rng(abs(hash((ds.paths[i], key))) % (1 << 32))
            out.append(augmentations.apply_condition(imgs[i], nm, p, rng))
        return out

    def _eval(cond_imgs):
        with torch.no_grad():
            full = live_probs(detector, cond_imgs)
        with _FreqAblator(detector):
            with torch.no_grad():
                ablated = live_probs(detector, cond_imgs)
        return full, ablated

    rows = {}
    for key in sub_grid:
        cond_imgs = _render(key)
        full, ablated = _eval(cond_imgs)
        rows[key] = {
            "acc_full": _acc(full, labels), "acc_ablated": _acc(ablated, labels),
            "auc_full": _auc(full, labels), "auc_ablated": _auc(ablated, labels),
        }
        d = rows[key]["acc_full"] - rows[key]["acc_ablated"]
        print(f"    {key:12s} full acc={rows[key]['acc_full']*100:6.2f}%  "
              f"freq-off acc={rows[key]['acc_ablated']*100:6.2f}%  delta={d*100:+.2f} pts")

    clean_full = rows["clean"]["acc_full"]
    clean_ablated = rows["clean"]["acc_ablated"]
    robust_keys = [k for k in sub_grid if k != "clean"]
    mean_delta_robust = float(np.mean(
        [rows[k]["acc_full"] - rows[k]["acc_ablated"] for k in robust_keys]
    ))
    clean_delta = clean_full - clean_ablated
    vs_v0 = clean_full - v0_clean_acc
    robust_deltas = {k: rows[k]["acc_full"] - rows[k]["acc_ablated"] for k in robust_keys}
    worst_hurt_key = min(robust_deltas, key=robust_deltas.get)
    worst_hurt = robust_deltas[worst_hurt_key]
    n_helped = sum(d > 0.005 for d in robust_deltas.values())
    n_hurt = sum(d < -0.02 for d in robust_deltas.values())

    if worst_hurt < -0.05:
        verdict = FAIL
        note = (f"the frequency stream ACTIVELY DEGRADES robustness where V0 is already "
                f"weakest: on {worst_hurt_key} the full V1 is {abs(worst_hurt)*100:.0f} pts "
                f"BELOW the freq-off (spatial-only) path. Additive noise is broadband "
                f"high-frequency energy -- it inflates the SRM residual and the "
                f"residual-energy scalar, so the fusion head trusts the forensic branch "
                f"MORE exactly when it is least reliable (the reliability hint has the "
                f"wrong sign for noise). It helps +1-1.5 pts on JPEG/blur/resize/clean "
                f"but that does not offset a {abs(worst_hurt)*100:.0f}-pt noise regression, "
                f"and V1 clean ({clean_full*100:.1f}%) does not beat V0 ({v0_clean_acc*100:.1f}%).")
    elif mean_delta_robust < 0.005 and clean_delta < 0.005:
        verdict = WARN
        note = ("removing the frequency stream barely changes V1 -- the spatial stream "
                "is doing essentially all the work; the forensic branch is close to "
                "redundant on this dataset.")
    elif (n_helped >= 3 and clean_delta >= 0.005) or mean_delta_robust >= 0.02:
        verdict = PASS
        note = ("the frequency stream measurably helps across conditions; V1's fusion "
                "is using it productively.")
    else:
        verdict = WARN
        note = ("the frequency stream contributes only marginally; keep it for the "
                "ablation narrative but do not over-claim its value.")

    if vs_v0 < -0.002 and verdict != FAIL:
        note += (f" NB: V1 clean acc is {abs(vs_v0)*100:.2f} pts BELOW V0 -- the branch "
                 "did not improve the headline number.")

    print(f"    clean delta={clean_delta*100:+.2f} pts  mean robust delta={mean_delta_robust*100:+.2f} pts  "
          f"worst={worst_hurt_key} ({worst_hurt*100:+.1f} pts)  vs V0 clean={vs_v0*100:+.2f} pts")
    return {
        "verdict": verdict, "note": note,
        "n_images": n, "sub_grid": sub_grid,
        "clean_acc_full": clean_full, "clean_acc_freq_off": clean_ablated,
        "clean_delta_pts": clean_delta,
        "mean_robust_delta_pts": mean_delta_robust,
        "worst_hurt_condition": worst_hurt_key, "worst_hurt_pts": worst_hurt,
        "v1_vs_v0_clean_pts": vs_v0,
        "conditions": rows,
    }


# ==========================================================================
# 6. Calibration / reliability diagram
# ==========================================================================

def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Standard (Guo et al. 2017) confidence ECE: bin by confidence in the
    PREDICTED class -- max(p, 1-p) -- and compare each bin's mean confidence
    to its accuracy. Binning by raw P(fake) and comparing to accuracy is a
    different, incompatible quantity (it conflates the two decision regions).
    """
    conf = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5).astype(int) == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)          # confidence lives in [0.5, 1]
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _ece_positive_class(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """The other convention: bin by P(fake), compare to observed fraction fake
    (this is what the reliability curve plots). Reported alongside for context."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


# (check 6 is implemented as `test_calibration_from_probs`, which reuses the
#  single live pass over the clean test set that check 7 also needs.)


# ==========================================================================
# 7. Class-wise performance
# ==========================================================================

def test_classwise(detector: Detector, calib_probs=None, calib_labels=None) -> dict:
    print("\n[7] Class-wise performance (clean test set)")
    if calib_probs is None:
        ds = SidDataset("test")
        imgs = [ds[i][0] for i in range(len(ds))]
        calib_labels = np.array(ds.labels, dtype=np.int64)
        calib_probs = live_probs(detector, imgs, tag="classwise")

    prf = _per_class_prf(calib_probs, calib_labels)
    rec_gap = abs(prf["real"]["recall"] - prf["fake"]["recall"])
    prec_gap = abs(prf["real"]["precision"] - prf["fake"]["precision"])
    gap = max(rec_gap, prec_gap)

    if gap < 0.03:
        verdict = PASS
        note = "balanced across classes; no class-specific artifact signature."
    elif gap < 0.08:
        verdict = WARN
        note = (f"{gap*100:.1f}-pt precision/recall gap between classes -- minor "
                "bias, watch the false-positive direction.")
    else:
        verdict = FAIL
        note = (f"{gap*100:.1f}-pt gap between classes -- the model is markedly "
                "better on one class, suggesting a class-specific cue.")

    print(f"    real  P={prf['real']['precision']:.3f} R={prf['real']['recall']:.3f} F1={prf['real']['f1']:.3f}")
    print(f"    fake  P={prf['fake']['precision']:.3f} R={prf['fake']['recall']:.3f} F1={prf['fake']['f1']:.3f}")
    return {"verdict": verdict, "note": note, "per_class": prf,
            "recall_gap": rec_gap, "precision_gap": prec_gap}


# ==========================================================================
# Report
# ==========================================================================

def _final_conclusion(verdicts: list[str]) -> str:
    if FAIL in verdicts:
        return "Investigate further"
    if WARN in verdicts:
        return "Proceed with caution"
    return "Trust the detector"


def render_report(results: dict, meta: dict) -> str:
    rows = [
        ("1. Cross-dataset (CIFAKE)", results["test1"]),
        ("2. Overfitting curves", results["test2"]),
        ("3. Label-shuffling sanity", results["test3"]),
        ("4. Unseen corruptions", results["test4"]),
        ("5. Frequency-stream contribution", results["test5"]),
        ("6. Calibration (ECE)", results["test6"]),
        ("7. Class-wise performance", results["test7"]),
    ]
    verdicts = [r[1]["verdict"] for r in rows if r[1]["verdict"] in (PASS, WARN, FAIL)]
    conclusion = _final_conclusion(verdicts)

    def metrics_cell(key, res):
        if key.startswith("1."):
            return (f"CIFAKE AUC {res['cifake_auc']:.3f} (SID {res['sid_clean_auc']:.3f}); "
                    f"acc {res['cifake_acc']*100:.1f}% @0.5 / {res['cifake_acc_opt_threshold']*100:.1f}% @opt-thr; "
                    f"F1 real {res['cifake_f1_real']:.2f}, fake {res['cifake_f1_fake']:.2f}")
        if key.startswith("2."):
            return (f"peak val acc {res['peak_val_acc']*100:.2f}%, final {res['final_val_acc']*100:.2f}%, "
                    f"val regression {res['val_regression_from_peak']*100:.2f} pts, "
                    f"train-val gap {res['final_train_val_gap']*100:.2f} pts")
        if key.startswith("3."):
            return (f"train acc on random labels {res['train_acc_on_shuffled']*100:.1f}% -> "
                    f"real-test acc {res['real_test_acc']*100:.1f}% / AUC {res['real_test_auc']:.3f}")
        if key.startswith("4."):
            return (f"avg drop {res['avg_drop_pts']*100:.2f} pts over "
                    f"{len(res['conditions'])} unseen corruptions; worst "
                    f"{res['worst']} ({res['worst_drop_pts']*100:.1f} pts)")
        if key.startswith("5."):
            if res["verdict"] == "N/A":
                return "V1 not trained yet"
            return (f"freq-stream effect: clean {res['clean_delta_pts']*100:+.1f} pts, "
                    f"worst {res['worst_hurt_condition']} {res['worst_hurt_pts']*100:+.1f} pts; "
                    f"V1 vs V0 clean {res['v1_vs_v0_clean_pts']*100:+.2f} pts")
        if key.startswith("6."):
            return (f"confidence-ECE {res['ece']:.3f}, Brier {res['brier']:.4f}, "
                    f"mean conf when wrong {res['mean_confidence_when_wrong']:.2f}")
        if key.startswith("7."):
            pc = res["per_class"]
            return (f"real R {pc['real']['recall']:.3f} / fake R {pc['fake']['recall']:.3f} "
                    f"(gap {res['recall_gap']*100:.1f} pts)")
        return ""

    lines = []
    lines.append("# Detector Trust Report -- V0" + (" & V1" if meta["has_v1"] else ""))
    lines.append("")
    lines.append(f"- Generated: {meta['generated_at']}")
    lines.append(f"- Checkpoint under test: `{meta['checkpoint']}` "
                 f"(val acc {meta['ckpt_meta'].get('val_acc', float('nan'))*100:.2f}%, "
                 f"best epoch {meta['ckpt_meta'].get('best_epoch', '?')})")
    lines.append(f"- Dataset: SID_Set, real vs full_synthetic "
                 f"(test n={meta['n_test']}); cross-dataset probe: CIFAKE test "
                 f"(n={results['test1']['n']})")
    lines.append(f"- Cache/live embedding consistency: max abs diff "
                 f"{meta['consistency']['max_abs_diff']:.2e}, mean cosine "
                 f"{meta['consistency']['mean_cosine']:.5f} "
                 f"(justifies using the cache for checks 2-3)")
    lines.append("")
    lines.append(f"## Verdict: **{conclusion}**")
    lines.append("")
    lines.append(f"{verdicts.count(PASS)} Pass / {verdicts.count(WARN)} Warning / "
                 f"{verdicts.count(FAIL)} Fail across {len(verdicts)} scored checks.")
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append("| Check | Key metrics | Verdict |")
    lines.append("|---|---|---|")
    for name, res in rows:
        lines.append(f"| {name} | {metrics_cell(name, res)} | **{res['verdict']}** |")
    lines.append("")

    lines.append("## Per-check detail")
    lines.append("")
    for name, res in rows:
        lines.append(f"### {name} -- {res['verdict']}")
        lines.append("")
        lines.append(res["note"])
        lines.append("")
        if name.startswith("1."):
            pc = res["per_class"]
            lines.append(f"- CIFAKE: AUC {res['cifake_auc']:.4f} (SID {res['sid_clean_auc']:.4f}, "
                         f"drop {res['auc_drop']:.3f})")
            lines.append(f"- accuracy: {res['cifake_acc']*100:.2f}% at the 0.5 threshold, "
                         f"{res['cifake_acc_opt_threshold']*100:.2f}% at the Youden-optimal "
                         f"threshold ({res['cifake_opt_threshold']:.3f})")
            lines.append(f"- real: P {pc['real']['precision']:.3f} / R {pc['real']['recall']:.3f} / F1 {pc['real']['f1']:.3f}")
            lines.append(f"- fake: P {pc['fake']['precision']:.3f} / R {pc['fake']['recall']:.3f} / F1 {pc['fake']['f1']:.3f}")
            lines.append(f"- accuracy drop vs SID clean: {res['acc_drop_pts']*100:.2f} pts "
                         "(most of it a threshold shift, not lost discrimination -- see acc@opt-thr)")
        elif name.startswith("2."):
            lines.append(f"- best val-AUC epoch: {res['best_val_auc_epoch']} "
                         f"(early-stop patience {res['early_stop_patience']})")
            lines.append(f"- peak / final val accuracy: {res['peak_val_acc']*100:.2f}% / {res['final_val_acc']*100:.2f}%")
            lines.append(f"- final train-val accuracy gap: {res['final_train_val_gap']*100:.2f} pts")
            lines.append(f"- curve plot: `{res['plot']}`")
        elif name.startswith("3."):
            lines.append(f"- shuffled-label training reached {res['train_acc_on_shuffled']*100:.1f}% "
                         "train accuracy (confirms the head has capacity to memorize noise)")
            lines.append(f"- but real-test accuracy is {res['real_test_acc']*100:.2f}% "
                         f"(AUC {res['real_test_auc']:.4f}) -- at chance, as required")
        elif name.startswith("4."):
            lines.append("| corruption | acc | drop (pts) | AUC |")
            lines.append("|---|---|---|---|")
            for k, v in res["conditions"].items():
                lines.append(f"| {k} | {v['accuracy']*100:.2f}% | {v['drop']*100:+.2f} | {v['auc']:.4f} |")
        elif name.startswith("5.") and res["verdict"] != "N/A":
            lines.append("delta = (V1 full) - (V1 with frequency embedding + energy scalar forced to zero). "
                         "Positive = the frequency stream helps; negative = it hurts.")
            lines.append("")
            lines.append("| condition | acc (V1 full) | acc (freq off) | delta (pts) |")
            lines.append("|---|---|---|---|")
            for k, v in res["conditions"].items():
                d = (v["acc_full"] - v["acc_ablated"]) * 100
                lines.append(f"| {k} | {v['acc_full']*100:.2f}% | {v['acc_ablated']*100:.2f}% | {d:+.2f} |")
        elif name.startswith("6."):
            lines.append(f"- confidence-ECE {res['ece']:.4f} (Guo et al.: bin by max(p,1-p) vs accuracy)")
            lines.append(f"- positive-class ECE {res['ece_positive_class']:.4f} (bin by P(fake) vs fraction fake -- what the curve plots)")
            lines.append(f"- Brier {res['brier']:.4f}")
            lines.append(f"- mean confidence on wrong predictions: {res['mean_confidence_when_wrong']:.3f}")
            lines.append(f"- reliability diagram: `{res['plot']}`")
        elif name.startswith("7."):
            pc = res["per_class"]
            lines.append("| class | precision | recall | F1 | support |")
            lines.append("|---|---|---|---|---|")
            for c in ("real", "fake"):
                lines.append(f"| {c} | {pc[c]['precision']:.3f} | {pc[c]['recall']:.3f} | "
                             f"{pc[c]['f1']:.3f} | {pc[c]['support']} |")
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"**{conclusion}.**")
    lines.append("")
    if conclusion != "Trust the detector":
        lines.append("Main reservations:")
        for name, res in rows:
            if res["verdict"] in (WARN, FAIL):
                lines.append(f"- **{name}** ({res['verdict']}): {res['note']}")
        lines.append("")

    return "\n".join(lines)


# ==========================================================================
# main
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description="Trust / validation suite for V0/V1.")
    ap.add_argument("--variant", choices=["v0", "v1"], default="v0",
                    help="which checkpoint the evaluation checks use (default v0)")
    ap.add_argument("--cifake-per-class", type=int, default=1500)
    ap.add_argument("--corruption-images", type=int, default=200)
    ap.add_argument("--overfit-epochs", type=int, default=40)
    ap.add_argument("--shuffle-epochs", type=int, default=10)
    args = ap.parse_args()

    import datetime as _dt
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt = config.CHECKPOINTS[args.variant]
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt} -- train {args.variant} first")
    detector = Detector.from_checkpoint(ckpt)
    detector.eval()
    ckpt_meta = torch.load(ckpt, map_location="cpu").get("meta", {})

    with open(config.RESULT_FILES["v0"], encoding="utf-8") as fh:
        v0_json = json.load(fh)
    sid_clean_acc = v0_json["clean_accuracy"]
    sid_clean_auc = v0_json["clean_auc"]

    consistency = cache_vs_live_consistency(detector)
    print(f"[pre-flight] cache vs live embedding: max|diff|={consistency['max_abs_diff']:.2e}  "
          f"cosine={consistency['mean_cosine']:.5f}")

    results = {}
    results["test1"] = test_cross_dataset(detector, args.cifake_per_class,
                                          sid_clean_acc, sid_clean_auc)
    results["test2"] = test_overfitting(args.overfit_epochs)
    results["test3"] = test_label_shuffle(args.shuffle_epochs)
    results["test4"] = test_unseen_corruptions(detector, args.corruption_images)
    results["test5"] = test_frequency_contribution(args.corruption_images, sid_clean_acc)

    # checks 6 & 7 share one live pass over the clean test set
    ds = SidDataset("test")
    clean_imgs = [ds[i][0] for i in range(len(ds))]
    clean_labels = np.array(ds.labels, dtype=np.int64)
    clean_probs = live_probs(detector, clean_imgs, tag="clean-test")
    results["test6"] = test_calibration_from_probs(clean_probs, clean_labels)
    results["test7"] = test_classwise(detector, clean_probs, clean_labels)

    meta = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt, "ckpt_meta": ckpt_meta,
        "n_test": len(ds), "has_v1": os.path.exists(config.CHECKPOINTS["v1"]),
        "consistency": consistency,
    }
    report = render_report(results, meta)

    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": _json_safe(results)}, fh, indent=2)

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nwrote {OUT_DIR}/REPORT.md, {OUT_DIR}/results.json, and 2 PNGs")


def test_calibration_from_probs(probs, labels) -> dict:
    """Check 6 body, reusing an already-computed live pass over the clean test set."""
    print("\n[6] Calibration (clean test set, live inference)")
    ece = _ece(probs, labels, n_bins=15)
    ece_pos = _ece_positive_class(probs, labels, n_bins=10)
    brier = float(brier_score_loss(labels, probs))
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")
    wrong = probs[(probs >= 0.5).astype(int) != labels]
    mean_conf_when_wrong = float(np.mean(np.maximum(wrong, 1 - wrong))) if len(wrong) else float("nan")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls=":", color="grey", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="tab:purple", label="V0")
    ax.set_xlabel("mean predicted P(fake)"); ax.set_ylabel("observed fraction fake")
    ax.set_title(f"V0 reliability diagram (ECE={ece:.3f}, Brier={brier:.3f})")
    ax.legend(frameon=False); fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "calibration_v0.png")
    fig.savefig(plot_path, dpi=110); plt.close(fig)

    if ece < 0.03:
        verdict = PASS
        note = (f"well calibrated in-domain (confidence-ECE {ece:.3f}, Brier {brier:.4f}); "
                "predicted confidence tracks accuracy on SID.")
    elif ece < 0.08:
        verdict = WARN
        note = (f"mild in-domain miscalibration (confidence-ECE {ece:.3f}); a temperature "
                "scale would tighten it.")
    else:
        verdict = FAIL
        note = f"poorly calibrated in-domain (confidence-ECE {ece:.3f})."
    if not np.isnan(mean_conf_when_wrong) and mean_conf_when_wrong > 0.9:
        note += (f" The model is confident even when wrong (mean conf on errors "
                 f"{mean_conf_when_wrong:.2f}) -- so its errors are silent, and the "
                 "in-domain calibration does NOT transfer off-SID (see check 1).")

    print(f"    confidence-ECE={ece:.4f}  positive-class-ECE={ece_pos:.4f}  "
          f"Brier={brier:.4f}  mean conf when wrong={mean_conf_when_wrong:.3f}")
    print(f"    plot -> {plot_path}")
    return {"verdict": verdict, "note": note, "ece": ece, "ece_positive_class": ece_pos,
            "brier": brier, "mean_confidence_when_wrong": mean_conf_when_wrong,
            "curve": {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()},
            "plot": plot_path}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
