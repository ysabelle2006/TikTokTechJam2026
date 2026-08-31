"""
Offline CLIP embedding cache.

Two modes:

1. clean
   One embedding per original training image.

   Outputs:
       train_clean.npy
       train_clean_index.csv

2. augmented
   K sampled robustness transforms per original image.

   Outputs:
       train_augmented.npy
       train_augmented_index.csv

The frozen CLIP backbone is run only once per cached variant so later
classifier training can use the embeddings directly.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from tiktoktechjam2026.data.datasets import load_manifest
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.transforms.augmentations import (
    apply_condition,
    sample_condition_names,
)


CACHE_DIR = Path("cache/spatial_embeddings")
EMBED_DIM = 512


# ============================================================
# Helpers
# ============================================================

def shuffled_subset(rows, limit=None, seed=0):
    if limit is None:
        return rows

    rng = random.Random(seed)

    rows = rows.copy()
    rng.shuffle(rows)

    return rows[:limit]


def load_spatial_stream():
    print("\nLoading CLIP spatial stream...")

    spatial = SpatialStream()

    spatial.model.eval()

    for parameter in spatial.model.parameters():
        parameter.requires_grad = False

    return spatial


# ============================================================
# Clean cache
# ============================================================

def cache_clean(
    rows,
    split,
    batch_size,
):
    spatial = load_spatial_stream()

    embeddings = np.empty(
        (len(rows), EMBED_DIM),
        dtype=np.float32,
    )

    skipped = []

    batch_tensors = []
    batch_indices = []

    def flush_batch():
        if not batch_tensors:
            return

        batch = torch.stack(
            batch_tensors
        )

        with torch.no_grad():
            output = spatial.encode(
                batch
            )

        output = (
            output
            .float()
            .cpu()
            .numpy()
        )

        for local_i, global_i in enumerate(
            batch_indices
        ):
            embeddings[
                global_i
            ] = output[
                local_i
            ]

        batch_tensors.clear()
        batch_indices.clear()

    for i, row in enumerate(
        tqdm(
            rows,
            desc=f"Caching {split} clean CLIP embeddings",
        )
    ):

        try:
            image = Image.open(
                row["path"]
            ).convert("RGB")

            image_tensor = spatial.preprocess(
                image
            )

        except Exception as e:
            skipped.append(
                (
                    row["path"],
                    str(e),
                )
            )

            embeddings[i] = np.nan
            continue

        batch_tensors.append(
            image_tensor
        )

        batch_indices.append(i)

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        CACHE_DIR
        / f"{split}_clean.npy"
    )

    index_path = (
        CACHE_DIR
        / f"{split}_clean_index.csv"
    )

    np.save(
        embedding_path,
        embeddings,
    )

    with open(
        index_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "source",
                "generator",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "source": row["source"],
                    "generator": row["generator"],
                }
            )

    print("\nCLEAN CACHE DONE")
    print(
        "Embeddings saved to:",
        embedding_path,
    )
    print(
        "Index saved to:",
        index_path,
    )
    print(
        "Embedding shape:",
        embeddings.shape,
    )
    print(
        "Skipped:",
        len(skipped),
    )


# ============================================================
# Augmented cache
# ============================================================

def cache_augmented(
    rows,
    split,
    batch_size,
    k_per_image,
    seed,
):
    spatial = load_spatial_stream()

    # --------------------------------------------------------
    # Choose a fixed set of transforms for every image.
    #
    # One seeded RNG is shared across the whole dataset so
    # rerunning with the same seed reproduces the same variants.
    # --------------------------------------------------------

    sampler_rng = random.Random(
        seed
    )

    variant_rows = []

    for row in rows:

        conditions = sample_condition_names(
            k_per_image,
            sampler_rng,
        )

        for condition in conditions:
            variant_rows.append(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "source": row["source"],
                    "generator": row["generator"],
                    "condition": condition,
                }
            )

    print()
    print(
        f"{len(rows)} images × "
        f"{k_per_image} transforms"
    )
    print(
        f"= {len(variant_rows)} "
        f"augmented embeddings"
    )

    embeddings = np.empty(
        (
            len(variant_rows),
            EMBED_DIM,
        ),
        dtype=np.float32,
    )

    skipped = []

    batch_tensors = []
    batch_indices = []

    def flush_batch():
        if not batch_tensors:
            return

        batch = torch.stack(
            batch_tensors
        )

        with torch.no_grad():
            output = spatial.encode(
                batch
            )

        output = (
            output
            .float()
            .cpu()
            .numpy()
        )

        for local_i, global_i in enumerate(
            batch_indices
        ):
            embeddings[
                global_i
            ] = output[
                local_i
            ]

        batch_tensors.clear()
        batch_indices.clear()

    # --------------------------------------------------------
    # Transform + encode
    # --------------------------------------------------------

    for i, row in enumerate(
        tqdm(
            variant_rows,
            desc=(
                f"Caching {split} "
                f"augmented CLIP embeddings"
            ),
        )
    ):

        try:
            image = Image.open(
                row["path"]
            ).convert("RGB")

            image = apply_condition(
                image,
                row["condition"],
            )

            image_tensor = spatial.preprocess(
                image
            )

        except Exception as e:
            skipped.append(
                (
                    row["path"],
                    row["condition"],
                    str(e),
                )
            )

            embeddings[i] = np.nan
            continue

        batch_tensors.append(
            image_tensor
        )

        batch_indices.append(i)

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    # --------------------------------------------------------
    # Save separately from clean cache
    # --------------------------------------------------------

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        CACHE_DIR
        / f"{split}_augmented.npy"
    )

    index_path = (
        CACHE_DIR
        / f"{split}_augmented_index.csv"
    )

    np.save(
        embedding_path,
        embeddings,
    )

    with open(
        index_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "source",
                "generator",
                "condition",
            ],
        )

        writer.writeheader()

        for row in variant_rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "label": row["label"],
                    "source": row["source"],
                    "generator": row["generator"],
                    "condition": row["condition"],
                }
            )

    print("\nAUGMENTED CACHE DONE")
    print(
        "Embeddings saved to:",
        embedding_path,
    )
    print(
        "Index saved to:",
        index_path,
    )
    print(
        "Embedding shape:",
        embeddings.shape,
    )
    print(
        "Skipped:",
        len(skipped),
    )

    if skipped:
        print(
            "\nFirst few skipped variants:"
        )

        for (
            path,
            condition,
            error,
        ) in skipped[:5]:

            print(
                f"  {path} "
                f"[{condition}]: "
                f"{error}"
            )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="train",
        choices=[
            "train",
            "validation_demo",
        ],
    )

    parser.add_argument(
        "--variant",
        default="clean",
        choices=[
            "clean",
            "augmented",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Use only a random N-image subset "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--k-per-image",
        type=int,
        default=3,
        help=(
            "For augmented mode only: "
            "number of distinct transforms "
            "sampled per image."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load shared manifest
    # --------------------------------------------------------

    rows = load_manifest(
        split=args.split
    )

    rows = shuffled_subset(
        rows,
        limit=args.limit,
        seed=args.seed,
    )

    if not rows:
        raise RuntimeError(
            f"No rows found for "
            f"split={args.split}"
        )

    print(
        f"\nVariant: {args.variant}"
    )

    print(
        f"Images selected: {len(rows)}"
    )

    # --------------------------------------------------------
    # Cache selected variant
    # --------------------------------------------------------

    if args.variant == "clean":

        cache_clean(
            rows=rows,
            split=args.split,
            batch_size=args.batch_size,
        )

    else:

        cache_augmented(
            rows=rows,
            split=args.split,
            batch_size=args.batch_size,
            k_per_image=args.k_per_image,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()