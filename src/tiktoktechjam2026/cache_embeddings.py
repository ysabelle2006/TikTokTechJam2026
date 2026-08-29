"""
Offline CLIP feature extraction + caching.

Running a ViT-B/32 forward pass on CPU, once per image per epoch, is the
real compute bottleneck here -- not the small frequency CNN or the fusion
head. So we run the frozen backbone once per (image, condition) and cache
the 512-d embeddings; training and evaluation then read cached vectors.

This is only valid because the backbone is frozen (config.FREEZE_BACKBONE).
If V4 unfreezes it, embeddings change every step and that stage has to run
CLIP live instead.

What gets cached (see config.EVAL_CONDITIONS for the condition keys):
  train / val : "clean" only        -- V0/V1 train on clean images
  test        : every condition     -- the robustness grid

Layout:
  cache/spatial_embeddings/<split>/<condition>.npy   float32 [N, 512]
  cache/spatial_embeddings/<split>/labels.npy        int64   [N]
  cache/spatial_embeddings/<split>/manifest.json     {paths: [...], ...}

The row order is identical across every condition file and matches
manifest["paths"], so evaluate.py can line up embeddings, labels and the
raw images it needs for the frequency stream.

CLI:
    python -m tiktoktechjam2026.cache_embeddings                 # everything needed for V0/V1
    python -m tiktoktechjam2026.cache_embeddings --splits test   # just the eval grid
    python -m tiktoktechjam2026.cache_embeddings --limit 64      # quick smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from tiktoktechjam2026 import config
from tiktoktechjam2026.data.datasets import SidDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.transforms import augmentations, preprocessing

# Which conditions to cache per split by default.
_DEFAULT_CONDITIONS = {
    "train": ["clean"],
    "val": ["clean"],
    "test": [key for key, _, _ in config.EVAL_CONDITIONS],
}

_CONDITION_BY_KEY = {key: (name, param) for key, name, param in config.EVAL_CONDITIONS}


def split_cache_dir(split: str) -> str:
    return os.path.join(config.EMBEDDING_CACHE_DIR, split)


def condition_cache_path(split: str, condition: str) -> str:
    return os.path.join(split_cache_dir(split), f"{condition}.npy")


def _seed_for(path: str, condition: str) -> int:
    """Deterministic per-(image, condition) seed for the stochastic transforms."""
    digest = hashlib.sha1(f"{path}|{condition}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _bounded_pmap(fn, n, workers, lookahead):
    """Yield (i, fn(i)) for i in range(n), in order, with at most `lookahead` in flight."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending, results = {}, {}
        submitted = delivered = 0
        while delivered < n:
            while submitted < n and len(pending) < lookahead:
                pending[pool.submit(fn, submitted)] = submitted
                submitted += 1
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                results[pending.pop(fut)] = fut.result()
            while delivered in results:
                yield delivered, results.pop(delivered)
                delivered += 1


def render_condition(image, path: str, condition: str):
    """
    Apply robustness condition `condition` to `image` exactly as the cache did.

    evaluate.py uses this to rebuild the frequency-stream input for the same
    render whose CLIP embedding is already cached -- so the two streams see
    identical pixels.
    """
    name, param = _CONDITION_BY_KEY[condition]
    rng = np.random.default_rng(_seed_for(path, condition))
    return augmentations.apply_condition(image, name, param, rng)


def load_cache(split: str, condition: str = "clean"):
    """Return (embeddings [N, 512] float32, labels [N] int64, paths list[str])."""
    emb = np.load(condition_cache_path(split, condition))
    labels = np.load(os.path.join(split_cache_dir(split), "labels.npy"))
    with open(os.path.join(split_cache_dir(split), "manifest.json"), encoding="utf-8") as fh:
        paths = json.load(fh)["paths"]
    return emb, labels, paths


# --------------------------------------------------------------------------
# Frequency-stream inputs (V1 training only)
#
# The SRM / FFT map for a clean image is deterministic, so for V1 training we
# render it once instead of every epoch. Stored as float16 to keep the file
# small; row order matches the spatial cache / manifest. The eval grid is
# small enough (test x 15 conditions) that evaluate.py builds those live.
# --------------------------------------------------------------------------

def freq_cache_path(split: str, mode: str = None) -> str:
    mode = mode or config.FREQUENCY_MODE
    return os.path.join(split_cache_dir(split), f"freq_{mode}_clean.npy")


def freq_energy_cache_path(split: str, mode: str = None) -> str:
    mode = mode or config.FREQUENCY_MODE
    return os.path.join(split_cache_dir(split), f"freq_{mode}_clean_energy.npy")


def load_freq_cache(split: str, mode: str = None):
    """
    Return (maps [N, C, 224, 224] float16, energy [N] float32) for the clean
    render of `split`. `energy` is preprocessing.residual_energy -- the same
    scalar evaluate.py computes live -- so training and eval never diverge.
    """
    maps = np.load(freq_cache_path(split, mode), mmap_mode="r")
    energy = np.load(freq_energy_cache_path(split, mode))
    return maps, energy


def cache_frequency_inputs(splits, mode=None, limit=None):
    mode = mode or config.FREQUENCY_MODE
    for split in splits:
        dataset = SidDataset(split)
        n = len(dataset) if limit is None else min(limit, len(dataset))
        path = freq_cache_path(split, mode)
        if os.path.exists(path) and limit is None:
            print(f"[{split}] freq ({mode}) cached, skipping")
            continue
        os.makedirs(split_cache_dir(split), exist_ok=True)

        sample = preprocessing.prepare_frequency_input(dataset[0][0], mode)
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(n, *sample.shape),
        )
        energy = np.empty(n, dtype=np.float32)
        t0 = time.time()

        def _one(i):
            image = dataset[i][0]
            return (
                preprocessing.prepare_frequency_input(image, mode).numpy(),
                preprocessing.residual_energy(image),
            )

        for i, (fmap, e) in _bounded_pmap(_one, n, workers=4, lookahead=256):
            out[i] = fmap
            energy[i] = e
            if (i + 1) % 512 == 0 or i == n - 1:
                rate = (i + 1) / (time.time() - t0)
                print(f"\r[{split}] freq {mode}  {i + 1}/{n}  ({rate:.1f} img/s)",
                      end="", flush=True)
        out.flush()
        np.save(freq_energy_cache_path(split, mode), energy)
        print()


@torch.no_grad()
def _embed_condition(stream, dataset, condition, batch_size, limit, workers=4):
    """
    Render `condition` for every image and CLIP-embed it.

    Image decode + resize + transform runs on a small thread pool (PIL /
    numpy release the GIL) with a bounded look-ahead, so rendering overlaps
    the ViT forward pass -- roughly doubles CPU throughput.
    """
    name, param = _CONDITION_BY_KEY[condition]
    n = len(dataset) if limit is None else min(limit, len(dataset))
    out = np.empty((n, config.SPATIAL_EMBED_DIM), dtype=np.float32)
    t0 = time.time()

    def _render(i):
        image, _, path = dataset[i]
        rng = np.random.default_rng(_seed_for(path, condition))
        rendered = augmentations.apply_condition(image, name, param, rng)
        return preprocessing.prepare_spatial_input(rendered)

    batch, rows = [], []
    for i, tensor in _bounded_pmap(_render, n, workers, lookahead=batch_size * 3):
        batch.append(tensor)
        rows.append(i)
        if len(batch) == batch_size or i == n - 1:
            out[rows] = stream.encode(torch.stack(batch)).cpu().numpy()
            batch, rows = [], []
            rate = (i + 1) / (time.time() - t0)
            print(f"\r    {condition:16s} {i + 1}/{n}  ({rate:.1f} img/s)",
                  end="", flush=True)

    print()
    return out


def main(splits=None, conditions=None, batch_size=64, limit=None, freq=True):
    splits = splits or ["train", "val", "test"]
    stream = SpatialStream()
    stream.eval()

    for split in splits:
        dataset = SidDataset(split)
        keys = conditions or _DEFAULT_CONDITIONS[split]
        cache_dir = split_cache_dir(split)
        os.makedirs(cache_dir, exist_ok=True)

        n = len(dataset) if limit is None else min(limit, len(dataset))
        labels = np.asarray(dataset.labels[:n], dtype=np.int64)
        np.save(os.path.join(cache_dir, "labels.npy"), labels)
        with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"split": split, "n": int(n), "paths": dataset.paths[:n],
                 "conditions": keys},
                fh, indent=0,
            )

        print(f"[{split}] {n} images  ->  {cache_dir}")
        for condition in keys:
            path = condition_cache_path(split, condition)
            if os.path.exists(path) and limit is None:
                print(f"    {condition:16s} cached, skipping")
                continue
            emb = _embed_condition(stream, dataset, condition, batch_size, limit)
            np.save(path, emb)

    # Frequency-stream inputs for V1 training (clean render of train/val).
    if freq:
        freq_splits = [s for s in splits if s in ("train", "val")]
        if freq_splits:
            cache_frequency_inputs(freq_splits, limit=limit)

    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cache frozen-CLIP embeddings for V0/V1.")
    ap.add_argument("--splits", nargs="+", choices=["train", "val", "test"])
    ap.add_argument("--conditions", nargs="+",
                    help="override which condition keys to cache (default: per-split)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per split (smoke test; forces recompute)")
    ap.add_argument("--no-freq", action="store_true",
                    help="skip caching frequency-stream inputs for V1")
    args = ap.parse_args()
    main(args.splits, args.conditions, args.batch_size, args.limit, freq=not args.no_freq)
