"""
Training entry point for V0 and V1.

    V0  frozen CLIP -> SpatialHead
    V1  frozen CLIP + SRM/FFT CNN + residual-energy scalar -> FusionHead

Both train only a small head on top of a frozen backbone, on CLEAN images
only (transform-aware augmentation is V2). V0/V1 objective is a plain
binary cross-entropy:

    L = BCEWithLogits(Detector.classify(x), y)

The V3 objective (+ transformed-view classification + consistency term) is
layered on later; see the module docstring history / README roadmap.

Reads precomputed features so epochs never re-run the ViT:
  * spatial: cache/spatial_embeddings/<split>/clean.npy   (cache_embeddings.py)
  * frequency (V1 only): cache/spatial_embeddings/<split>/freq_<mode>_clean.npy

Each stage writes its own checkpoint (config.CHECKPOINTS[variant]) -- never
overwriting another stage.

CLI:
    python -m tiktoktechjam2026.train --variant v0
    python -m tiktoktechjam2026.train --variant v1 --epochs 40
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from tiktoktechjam2026 import config
from tiktoktechjam2026.cache_embeddings import (
    load_aug_pool,
    load_aug_pool_freq,
    load_cache,
    load_freq_cache,
    render_aug_variant,
)
from tiktoktechjam2026.models.detector import Detector
from tiktoktechjam2026.transforms import preprocessing


def set_seed(seed: int = None):
    seed = config.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class FeatureBank:
    """Holds cached features for one split and yields shuffled minibatches."""

    def __init__(self, variant: str, split: str, freq_mode: str):
        emb, labels, _ = load_cache(split, "clean")
        self.emb = torch.from_numpy(np.ascontiguousarray(emb)).float()
        self.y = torch.from_numpy(labels).float()
        self.n = len(labels)

        self.freq = None
        self.energy = None
        if variant == "v1":
            freq, energy = load_freq_cache(split, freq_mode)   # [N,C,224,224] f16, [N] f32
            if len(freq) != self.n or len(energy) != self.n:
                raise ValueError(
                    f"{split}: freq cache has {len(freq)} rows, spatial has {self.n} "
                    "-- rebuild caches (cache_embeddings.py)"
                )
            self.freq = freq                                  # memmap; slice + cast per batch
            self.energy = torch.from_numpy(energy).float()

    def batches(self, batch_size: int, shuffle: bool):
        order = np.random.permutation(self.n) if shuffle else np.arange(self.n)
        for start in range(0, self.n, batch_size):
            idx = order[start:start + batch_size]
            emb = self.emb[idx]
            y = self.y[idx]
            if self.freq is None:
                yield emb, None, None, y
            else:
                # Ordered memmap read, then re-align rows to `idx`.
                order_idx = np.sort(idx)
                reorder = np.argsort(np.argsort(idx))
                fb = torch.from_numpy(
                    np.asarray(self.freq[order_idx], dtype=np.float32)
                )[reorder]
                yield emb, fb, self.energy[idx], y


# --------------------------------------------------------------------------
# Eval
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bank(detector: Detector, bank: FeatureBank, batch_size: int):
    detector.eval()
    probs, ys = [], []
    for emb, freq, energy, y in bank.batches(batch_size, shuffle=False):
        logits = detector.classify(emb, freq, energy)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    acc = float(((probs >= 0.5).astype(int) == ys).mean())
    auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else float("nan")
    return acc, auc


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

def train(variant: str, epochs: int = None, batch_size: int = None,
          lr: float = None, freq_mode: str = None):
    if variant == "v2":
        return train_v2(epochs, batch_size, lr, freq_mode)

    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    freq_mode = freq_mode or config.FREQUENCY_MODE

    set_seed()
    detector = Detector(variant=variant, freq_mode=freq_mode)
    print(f"[{variant}] trainable params: "
          f"{sum(p.numel() for p in detector.trainable_parameters()):,}  "
          f"(freq_mode={freq_mode})")

    train_bank = FeatureBank(variant, "train", freq_mode)
    val_bank = FeatureBank(variant, "val", freq_mode)

    opt = torch.optim.Adam(
        detector.trainable_parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch = -1.0, None, -1
    patience = 0

    for epoch in range(1, epochs + 1):
        detector.train()
        t0 = time.time()
        running = 0.0
        for emb, freq, energy, y in train_bank.batches(batch_size, shuffle=True):
            opt.zero_grad()
            logits = detector.classify(emb, freq, energy)
            loss = loss_fn(logits, y.to(detector.device))
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        train_loss = running / train_bank.n

        val_acc, val_auc = evaluate_bank(detector, val_bank, batch_size)
        dt = time.time() - t0
        marker = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            # Snapshot only the trainable heads (the frozen ViT never changes).
            best_state = {
                name: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
                for name, mod in detector._head_modules().items()
            }
            patience = 0
            marker = "  *"
        else:
            patience += 1
        print(f"  epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val_acc {val_acc*100:.2f}%  val_auc {val_auc:.4f}  ({dt:.1f}s){marker}")
        if patience >= config.EARLY_STOP_PATIENCE:
            print(f"  early stop (no val-AUC gain for {patience} epochs)")
            break

    for name, mod in detector._head_modules().items():
        mod.load_state_dict(best_state[name])
    val_acc, val_auc = evaluate_bank(detector, val_bank, batch_size)
    meta = {
        "variant": variant,
        "freq_mode": freq_mode,
        "best_epoch": best_epoch,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "epochs_run": epoch,
        "batch_size": batch_size,
        "lr": lr,
        "n_train": train_bank.n,
    }
    path = config.CHECKPOINTS[variant]
    detector.save(path, meta)
    print(f"[{variant}] best epoch {best_epoch}: val_acc {val_acc*100:.2f}%  "
          f"val_auc {val_auc:.4f}  ->  {path}")
    return meta


# --------------------------------------------------------------------------
# V2: two-stream (== V1 architecture) + transform-aware augmentation
#
# Spatial embeddings for the clean image and for a fixed pool of 8 seeded
# augmented variants are read from the cache (cache_embeddings --aug-pool 8).
# Each training example is the clean view with probability
# config.AUG_CLEAN_FRACTION, otherwise a uniformly-random pool variant --
# resampled every epoch, so the frequency CNN sees a wide spread of
# transforms over training. The frequency map is recomputed live (in
# DataLoader workers) from the identical seeded render, so it stays aligned
# with the cached spatial vector.
# --------------------------------------------------------------------------

class _V2RenderDataset(Dataset):
    """One item per unique train image -> (idx, variant, freq, energy, label).

    `variant` is -1 for the clean view or a pool index in [0, POOL). The
    spatial embedding is *not* returned here -- it is gathered from the
    in-memory cache in the training loop by (idx, variant), which keeps this
    dataset light to ship to worker processes.
    """

    def __init__(self, paths, labels, pool_size, clean_fraction, freq_mode):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.pool_size = int(pool_size)
        self.clean_fraction = float(clean_fraction)
        self.freq_mode = freq_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if random.random() < self.clean_fraction:
            variant, view = -1, img
        else:
            variant = random.randrange(self.pool_size)
            view = render_aug_variant(img, path, variant)
        freq = preprocessing.prepare_frequency_input(view, self.freq_mode)
        energy = preprocessing.residual_energy(view)
        return (i, variant, freq,
                torch.tensor(energy, dtype=torch.float32),
                torch.tensor(self.labels[i], dtype=torch.float32))


def _gather_emb(clean_emb, pool_emb, idx, variant):
    """idx, variant: LongTensor [B]. variant<0 -> clean, else pool[idx, variant]."""
    out = clean_emb[idx].clone()
    aug = variant >= 0
    if aug.any():
        out[aug] = pool_emb[idx[aug], variant[aug].clamp(min=0)]
    return out


@torch.no_grad()
def _eval_arrays(detector, emb, freq, energy, labels, batch_size):
    detector.eval()
    emb = torch.as_tensor(np.ascontiguousarray(emb), dtype=torch.float32)
    energy = torch.as_tensor(np.ascontiguousarray(energy), dtype=torch.float32)
    labels = np.asarray(labels)
    probs = np.empty(len(labels), dtype=np.float32)
    for s in range(0, len(labels), batch_size):
        sl = slice(s, min(s + batch_size, len(labels)))
        fb = torch.as_tensor(np.asarray(freq[sl], dtype=np.float32))
        logits = detector.classify(emb[sl], fb, energy[sl])
        probs[sl] = torch.sigmoid(logits).cpu().numpy()
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    return acc, auc


def _v2_validate(detector, freq_mode, batch_size):
    """Returns (clean_acc, clean_auc, aug_acc, aug_auc) on the val split.

    clean: cached clean embedding + cached clean frequency map.
    aug:   every cached pool variant (cached embedding + cached frequency map).
    """
    c_emb, c_labels, _ = load_cache("val", "clean")
    c_freq, c_energy = load_freq_cache("val", freq_mode)
    clean_acc, clean_auc = _eval_arrays(
        detector, c_emb, c_freq, c_energy, c_labels, batch_size
    )

    p_emb, p_labels, _ = load_aug_pool("val")                # [N, POOL, 512]
    p_freq, p_energy = load_aug_pool_freq("val", freq_mode)  # [N, POOL, C, H, W], [N, POOL]
    n, pool = p_emb.shape[:2]
    aug_acc, aug_auc = _eval_arrays(
        detector,
        p_emb.reshape(n * pool, -1),
        p_freq.reshape(n * pool, *p_freq.shape[2:]),
        p_energy.reshape(n * pool),
        np.repeat(p_labels, pool),
        batch_size,
    )
    return clean_acc, clean_auc, aug_acc, aug_auc


def train_v2(epochs=None, batch_size=None, lr=None, freq_mode=None):
    epochs = epochs or config.EPOCHS_V2
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr or config.LEARNING_RATE
    freq_mode = freq_mode or config.FREQUENCY_MODE

    set_seed()
    # DataLoader workers render + compute frequency maps on their own cores;
    # leave some for them so the two do not thrash.
    if config.NUM_WORKERS > 0:
        torch.set_num_threads(max(2, torch.get_num_threads() - config.NUM_WORKERS))
    detector = Detector(variant="v2", freq_mode=freq_mode)
    print(f"[v2] trainable params: "
          f"{sum(p.numel() for p in detector.trainable_parameters()):,}  "
          f"(freq_mode={freq_mode}, clean_frac={config.AUG_CLEAN_FRACTION}, "
          f"pool={config.AUG_POOL_SIZE})")

    clean_emb_np, labels, paths = load_cache("train", "clean")
    pool_emb_np, _, _ = load_aug_pool("train")
    clean_emb = torch.from_numpy(np.ascontiguousarray(clean_emb_np)).float()
    pool_emb = torch.from_numpy(np.ascontiguousarray(pool_emb_np)).float()
    n_train = len(paths)
    print(f"[v2] train: {n_train} images, pool {pool_emb.shape[1]}/image")

    ds = _V2RenderDataset(paths, labels, pool_emb.shape[1],
                          config.AUG_CLEAN_FRACTION, freq_mode)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=config.NUM_WORKERS,
        persistent_workers=config.NUM_WORKERS > 0,
        prefetch_factor=4 if config.NUM_WORKERS > 0 else None,
    )

    opt = torch.optim.Adam(
        detector.trainable_parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_score, best_state, best_epoch = -1.0, None, -1
    patience = 0
    for epoch in range(1, epochs + 1):
        detector.train()
        t0 = time.time()
        running = seen = clean_seen = 0
        for bi, (idx, variant, freq, energy, y) in enumerate(loader):
            emb = _gather_emb(clean_emb, pool_emb, idx, variant)
            opt.zero_grad()
            logits = detector.classify(emb, freq, energy)
            loss = loss_fn(logits, y.to(detector.device))
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
            clean_seen += int((variant < 0).sum())
            if bi % 10 == 0:
                print(f"\r  epoch {epoch:3d}  {seen}/{n_train}  loss {running/seen:.4f}  "
                      f"({seen/(time.time()-t0):.0f} img/s)", end="", flush=True)
        print("\r" + " " * 78 + "\r", end="")
        train_loss = running / seen

        clean_acc, clean_auc, aug_acc, aug_auc = _v2_validate(detector, freq_mode, batch_size)
        score = 0.5 * (clean_auc + aug_auc)          # early-stop tracker
        gap = clean_acc - aug_acc
        dt = time.time() - t0
        marker = ""
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {
                name: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
                for name, mod in detector._head_modules().items()
            }
            patience = 0
            marker = "  *"
        else:
            patience += 1
        print(f"  epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"clean_acc {clean_acc*100:.2f}%  aug_acc {aug_acc*100:.2f}%  "
              f"gap {gap*100:+.2f}  (clean_auc {clean_auc:.4f} aug_auc {aug_auc:.4f})  "
              f"[clean {clean_seen/seen*100:.0f}%]  ({dt:.0f}s){marker}")
        if patience >= config.EARLY_STOP_PATIENCE:
            print(f"  early stop (no val-score gain for {patience} epochs)")
            break

    for name, mod in detector._head_modules().items():
        mod.load_state_dict(best_state[name])
    clean_acc, clean_auc, aug_acc, aug_auc = _v2_validate(detector, freq_mode, batch_size)
    meta = {
        "variant": "v2",
        "freq_mode": freq_mode,
        "best_epoch": best_epoch,
        "val_clean_acc": clean_acc, "val_clean_auc": clean_auc,
        "val_aug_acc": aug_acc, "val_aug_auc": aug_auc,
        "val_clean_aug_gap": clean_acc - aug_acc,
        "epochs_run": epoch,
        "batch_size": batch_size,
        "lr": lr,
        "aug_pool_size": int(pool_emb.shape[1]),
        "aug_clean_fraction": config.AUG_CLEAN_FRACTION,
        "n_train": n_train,
    }
    path = config.CHECKPOINTS["v2"]
    detector.save(path, meta)
    print(f"[v2] best epoch {best_epoch}: clean_acc {clean_acc*100:.2f}%  "
          f"aug_acc {aug_acc*100:.2f}%  gap {(clean_acc - aug_acc)*100:+.2f}  ->  {path}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train V0, V1 or V2.")
    ap.add_argument("--variant", choices=["v0", "v1", "v2"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--freq-mode", choices=["srm", "fft"], default=None)
    args = ap.parse_args()
    train(args.variant, args.epochs, args.batch_size, args.lr, args.freq_mode)
