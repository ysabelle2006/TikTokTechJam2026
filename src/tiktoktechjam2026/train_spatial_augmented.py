import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


# ============================================================
# Paths
# ============================================================

CACHE_DIR = Path("cache/spatial_embeddings")

CLEAN_EMBEDDING_PATH = CACHE_DIR / "train_clean.npy"
CLEAN_INDEX_PATH = CACHE_DIR / "train_clean_index.csv"

AUG_EMBEDDING_PATH = CACHE_DIR / "train_augmented.npy"
AUG_INDEX_PATH = CACHE_DIR / "train_augmented_index.csv"

CHECKPOINT_PATH = Path(
    "checkpoints/v2_spatial_augmented_best.pt"
)


# ============================================================
# Model
# ============================================================

class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# Cache loader
# ============================================================

def load_cache(
    embedding_path,
    index_path,
):
    embeddings = np.load(
        embedding_path
    )

    with open(
        index_path,
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if len(embeddings) != len(rows):
        raise RuntimeError(
            f"Cache mismatch:\n"
            f"  embeddings: {len(embeddings)}\n"
            f"  index rows: {len(rows)}\n"
            f"  file: {embedding_path}"
        )

    return embeddings, rows


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--limit-clean",
        type=int,
        default=None,
        help=(
            "Optional limit on number of unique clean "
            "source images used."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    rng = np.random.default_rng(
        args.seed
    )

    # ========================================================
    # Load CLEAN cache
    # ========================================================

    clean_embeddings, clean_rows = load_cache(
        CLEAN_EMBEDDING_PATH,
        CLEAN_INDEX_PATH,
    )

    # ========================================================
    # Load AUGMENTED cache
    # ========================================================

    aug_embeddings, aug_rows = load_cache(
        AUG_EMBEDDING_PATH,
        AUG_INDEX_PATH,
    )

    # ========================================================
    # Match clean rows to images that actually have
    # augmented versions.
    #
    # This is important because the augmented cache may only
    # contain a subset of the full clean training set.
    # ========================================================

    augmented_paths = {
        row["path"]
        for row in aug_rows
    }

    clean_keep_indices = [
        i
        for i, row in enumerate(clean_rows)
        if row["path"] in augmented_paths
    ]

    clean_embeddings = clean_embeddings[
        clean_keep_indices
    ]

    clean_rows = [
        clean_rows[i]
        for i in clean_keep_indices
    ]

    print(
        f"\nMatched clean source images: "
        f"{len(clean_rows)}"
    )

    # ========================================================
    # Optional source-image limit
    # ========================================================

    if args.limit_clean is not None:

        if args.limit_clean > len(clean_rows):
            raise ValueError(
                f"--limit-clean={args.limit_clean} "
                f"but only {len(clean_rows)} matched "
                f"clean source images exist."
            )

        source_indices = rng.choice(
            len(clean_rows),
            size=args.limit_clean,
            replace=False,
        )

        selected_paths = {
            clean_rows[i]["path"]
            for i in source_indices
        }

        clean_keep_indices = [
            i
            for i, row in enumerate(clean_rows)
            if row["path"] in selected_paths
        ]

        aug_keep_indices = [
            i
            for i, row in enumerate(aug_rows)
            if row["path"] in selected_paths
        ]

        clean_embeddings = clean_embeddings[
            clean_keep_indices
        ]

        clean_rows = [
            clean_rows[i]
            for i in clean_keep_indices
        ]

        aug_embeddings = aug_embeddings[
            aug_keep_indices
        ]

        aug_rows = [
            aug_rows[i]
            for i in aug_keep_indices
        ]

    # ========================================================
    # Add explicit condition to clean rows
    # ========================================================

    for row in clean_rows:
        row["condition"] = "clean"

    # ========================================================
    # Combine clean + augmented
    # ========================================================

    embeddings = np.concatenate(
        [
            clean_embeddings,
            aug_embeddings,
        ],
        axis=0,
    )

    rows = (
        clean_rows
        + aug_rows
    )

    labels = np.array(
        [
            int(row["label"])
            for row in rows
        ],
        dtype=np.float32,
    )

    conditions = np.array(
        [
            row.get(
                "condition",
                "clean",
            )
            for row in rows
        ]
    )

    paths = np.array(
        [
            row["path"]
            for row in rows
        ]
    )

    # ========================================================
    # Dataset summary
    # ========================================================

    print()

    print(
        "Clean rows:     ",
        len(clean_rows),
    )

    print(
        "Augmented rows: ",
        len(aug_rows),
    )

    print(
        "Total rows:     ",
        len(rows),
    )

    print(
        "REAL:",
        int(
            (labels == 0).sum()
        ),
    )

    print(
        "FAKE:",
        int(
            (labels == 1).sum()
        ),
    )

    # ========================================================
    # IMPORTANT:
    # Split by SOURCE IMAGE, not individual rows.
    #
    # Otherwise an augmented version of an image could enter
    # validation while the clean version is in training.
    # That would leak information.
    # ========================================================

    unique_paths = np.unique(
        paths
    )

    rng.shuffle(
        unique_paths
    )

    val_source_count = int(
        len(unique_paths) * 0.2
    )

    val_sources = set(
        unique_paths[
            :val_source_count
        ]
    )

    train_sources = set(
        unique_paths[
            val_source_count:
        ]
    )

    train_mask = np.array(
        [
            path in train_sources
            for path in paths
        ]
    )

    val_mask = np.array(
        [
            path in val_sources
            for path in paths
        ]
    )

    train_idx = np.where(
        train_mask
    )[0]

    val_idx = np.where(
        val_mask
    )[0]

    print()

    print(
        "Training rows:",
        len(train_idx),
    )

    print(
        "Validation rows:",
        len(val_idx),
    )

    print(
        "Unique source images:",
        len(unique_paths),
    )

    # ========================================================
    # Convert to tensors
    # ========================================================

    X = torch.from_numpy(
        embeddings
    ).float()

    y = torch.from_numpy(
        labels
    ).float()

    # ========================================================
    # Model
    # ========================================================

    model = SpatialHead()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float(
        "inf"
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        shuffled_train_idx = (
            train_idx.copy()
        )

        rng.shuffle(
            shuffled_train_idx
        )

        train_loss = 0.0
        train_correct = 0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for start in range(
            0,
            len(shuffled_train_idx),
            args.batch_size,
        ):

            idx_np = shuffled_train_idx[
                start:
                start + args.batch_size
            ]

            idx = torch.from_numpy(
                idx_np
            ).long()

            optimizer.zero_grad()

            logits = model(
                X[idx]
            )

            loss = loss_fn(
                logits,
                y[idx],
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * len(idx)
            )

            train_correct += (
                (
                    (logits > 0).float()
                    == y[idx]
                )
                .sum()
                .item()
            )

        train_loss /= len(
            train_idx
        )

        train_acc = (
            train_correct
            / len(train_idx)
        )

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_idx_tensor = torch.from_numpy(
            val_idx
        ).long()

        with torch.no_grad():

            val_logits = model(
                X[val_idx_tensor]
            )

            val_loss = loss_fn(
                val_logits,
                y[val_idx_tensor],
            ).item()

            val_predictions = (
                val_logits > 0
            ).float()

            val_acc = (
                (
                    val_predictions
                    == y[val_idx_tensor]
                )
                .float()
                .mean()
                .item()
            )

        # ====================================================
        # Clean / augmented validation accuracy
        # ====================================================

        val_conditions = conditions[
            val_idx
        ]

        clean_mask = (
            val_conditions == "clean"
        )

        aug_mask = (
            val_conditions != "clean"
        )

        clean_acc = float("nan")
        aug_acc = float("nan")

        if clean_mask.any():

            clean_predictions = (
                val_predictions[
                    torch.from_numpy(
                        clean_mask
                    )
                ]
            )

            clean_labels = (
                y[val_idx_tensor][
                    torch.from_numpy(
                        clean_mask
                    )
                ]
            )

            clean_acc = (
                (
                    clean_predictions
                    == clean_labels
                )
                .float()
                .mean()
                .item()
            )

        if aug_mask.any():

            aug_predictions = (
                val_predictions[
                    torch.from_numpy(
                        aug_mask
                    )
                ]
            )

            aug_labels = (
                y[val_idx_tensor][
                    torch.from_numpy(
                        aug_mask
                    )
                ]
            )

            aug_acc = (
                (
                    aug_predictions
                    == aug_labels
                )
                .float()
                .mean()
                .item()
            )

        # ====================================================
        # Print epoch
        # ====================================================

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.3f} | "
            f"clean {clean_acc:.3f} | "
            f"aug {aug_acc:.3f}"
        )

        # ====================================================
        # Save best checkpoint
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH,
            )

            print(
                "  -> best checkpoint saved"
            )

    # ========================================================
    # Done
    # ========================================================

    print(
        "\nV2 SPATIAL AUGMENTED "
        "TRAINING COMPLETE"
    )

    print(
        "Saved:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()