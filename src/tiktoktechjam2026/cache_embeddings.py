"""
Offline feature-extraction step: run the frozen CLIP backbone once over
every image (the clean version, plus a fixed set of augmented variants)
and save the resulting 512-d embeddings to disk.

Why this exists: repeatedly running a ViT-B/32 forward pass on CPU,
once per image per epoch, is the actual compute bottleneck in this
project -- not the small frequency CNN or fusion head. Precomputing
embeddings once means later training epochs read cached vectors
instead of recomputing them, which is what actually makes the
frozen-backbone version CPU-feasible.

Trade-off worth knowing: this only works because the backbone is
frozen. If V4 unfreezes even part of it, embeddings change every
training step and this caching step no longer applies for that stage
-- fall back to running CLIP live there.

Also implies a design choice: rather than sampling a fresh random
augmentation every epoch, we fix a finite set of variants per image
(one rendering per parameter value in the brief's transform grid) and
cache all of them. That's a reasonable trade for CPU feasibility, and
it conveniently matches how the robustness evaluation is already
structured around discrete severities.

TODO: implement once transforms/preprocessing.py and
models/spatial_stream.py exist.
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


CACHE_DIR = Path("cache/spatial_embeddings")
EMBED_DIM = 512


def shuffled_subset(rows, limit=None, seed=0):
    if limit is None:
        return rows

    rng = random.Random(seed)

    rows = rows.copy()
    rng.shuffle(rows)

    return rows[:limit]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation_demo"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only cache a random N-image subset for testing.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # ============================================================
    # Load manifest
    # ============================================================

    rows = load_manifest(split=args.split)

    rows = shuffled_subset(
        rows,
        limit=args.limit,
        seed=args.seed,
    )

    if not rows:
        raise RuntimeError(
            f"No rows found for split={args.split}"
        )

    print(
        f"\nCaching {len(rows)} images "
        f"from split={args.split}"
    )

    # ============================================================
    # Load frozen CLIP spatial stream
    # ============================================================

    print("\nLoading CLIP spatial stream...")

    spatial = SpatialStream()

    spatial.model.eval()

    for parameter in spatial.model.parameters():
        parameter.requires_grad = False

    # ============================================================
    # Allocate output
    # ============================================================

    embeddings = np.empty(
        (len(rows), EMBED_DIM),
        dtype=np.float32,
    )

    skipped = []

    batch_tensors = []
    batch_indices = []

    # ============================================================
    # Batch CLIP encoding
    # ============================================================

    def flush_batch():
        if not batch_tensors:
            return

        batch = torch.stack(
            batch_tensors
        )

        with torch.no_grad():
            output = spatial.encode(batch)

        output = output.float().cpu().numpy()

        for local_i, global_i in enumerate(
            batch_indices
        ):
            embeddings[global_i] = output[local_i]

        batch_tensors.clear()
        batch_indices.clear()

    # ============================================================
    # Cache loop
    # ============================================================

    for i, row in enumerate(
        tqdm(
            rows,
            desc=f"Caching {args.split} CLIP embeddings",
        )
    ):

        try:
            image = Image.open(
                row["path"]
            ).convert("RGB")

        except Exception as e:
            skipped.append(
                (
                    row["path"],
                    str(e),
                )
            )

            embeddings[i] = np.nan
            continue

        image_tensor = spatial.preprocess(
            image
        )

        batch_tensors.append(
            image_tensor
        )

        batch_indices.append(i)

        if len(batch_tensors) >= args.batch_size:
            flush_batch()

    flush_batch()

    # ============================================================
    # Save cache
    # ============================================================

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        CACHE_DIR
        / f"{args.split}_clean.npy"
    )

    index_path = (
        CACHE_DIR
        / f"{args.split}_clean_index.csv"
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

    # ============================================================
    # Summary
    # ============================================================

    labels = np.array(
        [
            int(row["label"])
            for row in rows
        ]
    )

    print("\nDONE")

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
        "REAL:",
        int((labels == 0).sum()),
    )

    print(
        "FAKE:",
        int((labels == 1).sum()),
    )

    print(
        "Skipped:",
        len(skipped),
    )

    if skipped:
        print("\nFirst few skipped files:")

        for path, error in skipped[:5]:
            print(
                f"  {path}: {error}"
            )


if __name__ == "__main__":
    main()