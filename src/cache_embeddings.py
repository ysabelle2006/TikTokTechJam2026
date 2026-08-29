"""
Offline feature-extraction step: run the frozen CLIP backbone once over
every training image and save the resulting 512-d embeddings to disk,
so later training epochs read cached vectors instead of re-running
CLIP's forward pass every time -- see the architecture doc for why
that repeated forward pass, not the frequency CNN or fusion head, is
the actual CPU bottleneck in this project.

Two variants, both writing under config.EMBEDDING_CACHE_DIR:

  --variant clean (default, used by V0 and V1): one embedding per
  image, from the untouched image.
      <split>_clean.npy         (N, 512) float32, row i = embedding of index_rows[i]
      <split>_clean_index.csv   N rows, SAME ORDER as the .npy array:
                                 path,label,source,generator

  --variant augmented (V2): K distinct conditions PER IMAGE (from
  transforms.augmentations.sample_condition_names, weighted per the
  architecture doc's §01 crop-over-resize note), applied and encoded
  once, so V2's training loop reads a fixed, reproducible set of
  transformed embeddings per epoch instead of re-running CLIP on fresh
  random augmentation every epoch -- the same "cache once, don't
  recompute" reasoning that makes V0/V1 CPU-feasible at all. K defaults
  to 3 of the 19 available conditions (see augmentations.ALL_CONDITIONS)
  rather than all 19: caching every condition for all ~123K train
  images would be a ~19x cost multiplier over the clean cache, which is
  disproportionate for a hackathon budget -- the same "sample, don't
  do everything" call scripts/build_eval_grid.py already makes for the
  eval grid (SAMPLE_PER_SOURCE), just on the conditions axis instead of
  the images axis. Every condition still gets applied to a large,
  representative subset of images in aggregate (~123,000*K/19 images
  per condition at the default), which is what training actually needs
  -- it never needs to see literally every condition on literally every
  image.
      <split>_augmented.npy         (N*K, 512) float32
      <split>_augmented_index.csv   N*K rows, SAME ORDER as the .npy array:
                                     path,label,source,generator,condition

Same NaN-row contract in both variants: a row whose image failed to
load (or failed mid-transform) is still WRITTEN to both the array (as
NaN) and the index, rather than silently skipped -- skipping would
shift every later row's index out of alignment with the .npy array
without anything erroring. train.py drops NaN rows explicitly instead.

Run with:  uv run python src/cache_embeddings.py --split train
           uv run python src/cache_embeddings.py --split train --limit 200        (fast smoke test)
           uv run python src/cache_embeddings.py --split train --variant augmented
           uv run python src/cache_embeddings.py --split train --variant augmented --k-per-image 2 --limit 2000  (smoke test)
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import EMBEDDING_CACHE_DIR, SPATIAL_EMBED_DIM
from data.datasets import load_manifest
from models.spatial_stream import SpatialStream
from transforms.augmentations import apply_condition, sample_condition_names


def _shuffled_subset(rows, limit, seed):
    """Shared by cache_split and cache_augmented_split: shuffle before
    truncating, not just rows[:limit] -- the manifest is built source-
    by-source and label-by-label (see data/datasets.py's scan_cifake,
    which appends ALL of train/REAL before any train/FAKE), so an
    unshuffled head slice can easily be single-class. A --limit run is
    meant as a fast smoke test of the pipeline plumbing; a degenerate
    one-class sample defeats that purpose silently instead of erroring."""
    if not limit:
        return rows
    rng = random.Random(seed)
    rows = rows.copy()
    rng.shuffle(rows)
    return rows[:limit]


def cache_split(split: str, cache_dir: str = EMBEDDING_CACHE_DIR, batch_size: int = 64, limit: int = None, seed: int = 0):
    rows = load_manifest(split=split)
    rows = _shuffled_subset(rows, limit, seed)
    if not rows:
        print(f"No rows for split={split!r} -- run `python src/data/datasets.py` first.")
        return

    print(f"Loading CLIP backbone (this is the one-time cost caching amortizes)...")
    stream = SpatialStream()  # frozen by default, per config.FREEZE_BACKBONE

    embeddings = np.empty((len(rows), SPATIAL_EMBED_DIM), dtype=np.float32)
    batch_tensors, batch_indices = [], []
    skipped = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors)
        out = stream.encode(batch).cpu().numpy()
        for local_i, global_i in enumerate(batch_indices):
            embeddings[global_i] = out[local_i]
        batch_tensors.clear()
        batch_indices.clear()

    for i, r in enumerate(tqdm(rows, desc=f"caching {split}")):
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception as e:
            skipped.append((r["path"], str(e)))
            embeddings[i] = np.nan
            continue
        batch_tensors.append(stream.prepare(img))
        batch_indices.append(i)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    if skipped:
        print(f"\nWARNING: {len(skipped)} image(s) failed to load -- written as NaN rows "
              f"(train.py filters these out rather than training on NaN embeddings):")
        for path, err in skipped[:5]:
            print(f"  {path}: {err}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{split}_clean.npy"
    index_path = out_dir / f"{split}_clean_index.csv"

    np.save(npy_path, embeddings)
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "generator"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"path": r["path"], "label": r["label"], "source": r["source"], "generator": r["generator"]})

    print(f"\nCached {len(rows)} embeddings ({SPATIAL_EMBED_DIM}-d) -> {npy_path}")
    print(f"Index -> {index_path}")


def cache_augmented_split(
    split: str,
    k_per_image: int = 3,
    cache_dir: str = EMBEDDING_CACHE_DIR,
    batch_size: int = 64,
    limit: int = None,
    seed: int = 0,
):
    rows = load_manifest(split=split)
    rows = _shuffled_subset(rows, limit, seed)
    if not rows:
        print(f"No rows for split={split!r} -- run `python src/data/datasets.py` first.")
        return

    print("Loading CLIP backbone (this is the one-time cost caching amortizes)...")
    stream = SpatialStream()

    # One shared, seeded sampler advanced across every row in manifest
    # order -- NOT reseeded per row -- so a rerun with the same seed
    # reproduces the exact same per-image condition assignment. That's
    # the "fixed set of augmented variants" the architecture doc's
    # caching design calls for, as opposed to fresh random augmentation
    # every run (which would make V2's training set non-reproducible
    # across reruns, and non-comparable to itself epoch over epoch).
    sampler_rng = random.Random(seed)
    variant_rows = []
    for r in rows:
        for condition in sample_condition_names(k_per_image, sampler_rng):
            variant_rows.append((r, condition))
    print(f"{len(rows)} images x up to {k_per_image} sampled conditions each "
          f"-> {len(variant_rows)} augmented embeddings to compute")

    embeddings = np.empty((len(variant_rows), SPATIAL_EMBED_DIM), dtype=np.float32)
    batch_tensors, batch_indices = [], []
    skipped = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors)
        out = stream.encode(batch).cpu().numpy()
        for local_i, global_i in enumerate(batch_indices):
            embeddings[global_i] = out[local_i]
        batch_tensors.clear()
        batch_indices.clear()

    for i, (r, condition) in enumerate(tqdm(variant_rows, desc=f"caching {split} augmented (k={k_per_image})")):
        try:
            img = Image.open(r["path"]).convert("RGB")
            img = apply_condition(img, condition)
        except Exception as e:
            skipped.append((r["path"], condition, str(e)))
            embeddings[i] = np.nan
            continue
        batch_tensors.append(stream.prepare(img))
        batch_indices.append(i)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    if skipped:
        print(f"\nWARNING: {len(skipped)} image/condition pair(s) failed -- written as NaN rows "
              f"(train.py's V2 dataset filters these out rather than training on NaN embeddings):")
        for path, condition, err in skipped[:5]:
            print(f"  {path} [{condition}]: {err}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{split}_augmented.npy"
    index_path = out_dir / f"{split}_augmented_index.csv"

    np.save(npy_path, embeddings)
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "generator", "condition"])
        writer.writeheader()
        for r, condition in variant_rows:
            writer.writerow(
                {
                    "path": r["path"],
                    "label": r["label"],
                    "source": r["source"],
                    "generator": r["generator"],
                    "condition": condition,
                }
            )

    print(f"\nCached {len(variant_rows)} augmented embeddings from {len(rows)} images "
          f"(k={k_per_image} conditions/image) -> {npy_path}")
    print(f"Index -> {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Cache spatial-stream embeddings for a manifest split.")
    parser.add_argument("--split", default="train", choices=["train", "validation_demo"])
    parser.add_argument("--variant", default="clean", choices=["clean", "augmented"],
                         help="'clean' (default): one embedding per image, for V0/V1. "
                              "'augmented': K transformed variants per image, for V2's transform-aware training.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cache only a random N-row subset (fast iteration/debugging)")
    parser.add_argument("--k-per-image", type=int, default=3,
                         help="--variant augmented only: how many distinct conditions to sample per image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.variant == "clean":
        cache_split(args.split, batch_size=args.batch_size, limit=args.limit, seed=args.seed)
    else:
        cache_augmented_split(
            args.split,
            k_per_image=args.k_per_image,
            batch_size=args.batch_size,
            limit=args.limit,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
