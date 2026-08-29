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
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from tiktoktechjam2026 import config
from tiktoktechjam2026.cache_embeddings import load_cache, load_freq_cache
from tiktoktechjam2026.models.detector import Detector


def set_seed(seed: int = None):
    seed = config.SEED if seed is None else seed
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train V0 or V1.")
    ap.add_argument("--variant", choices=["v0", "v1"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--freq-mode", choices=["srm", "fft"], default=None)
    args = ap.parse_args()
    train(args.variant, args.epochs, args.batch_size, args.lr, args.freq_mode)
