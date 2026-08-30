import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead


CACHE_DIR = Path("cache/spatial_embeddings")

EMBEDDING_PATH = CACHE_DIR / "train_clean.npy"
INDEX_PATH = CACHE_DIR / "train_clean_index.csv"

CHECKPOINT_PATH = Path(
    "checkpoints/v1_fusion_sharedtrain_best.pt"
)


# ============================================================
# Dataset
# ============================================================

class CachedFusionDataset(Dataset):
    def __init__(self, embeddings, rows):
        assert len(embeddings) == len(rows)

        self.embeddings = embeddings
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        spatial_embedding = torch.from_numpy(
            self.embeddings[index]
        ).float()

        label = float(row["label"])

        return {
            "spatial_embedding": spatial_embedding,
            "path": row["path"],
            "label": label,
        }


# ============================================================
# Load cache
# ============================================================

def load_cache():
    if not EMBEDDING_PATH.is_file():
        raise FileNotFoundError(
            f"{EMBEDDING_PATH} not found.\n"
            "Run cache_embeddings.py first."
        )

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"{INDEX_PATH} not found."
        )

    embeddings = np.load(
        EMBEDDING_PATH
    )

    with open(
        INDEX_PATH,
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(embeddings) != len(rows):
        raise RuntimeError(
            "Embedding array and index CSV "
            "have different lengths."
        )

    # Remove any failed/NaN cache rows
    valid_mask = ~np.isnan(
        embeddings
    ).any(axis=1)

    skipped = int(
        (~valid_mask).sum()
    )

    if skipped:
        print(
            f"Dropping {skipped} "
            f"NaN cached rows."
        )

    embeddings = embeddings[
        valid_mask
    ]

    rows = [
        row
        for row, keep
        in zip(rows, valid_mask)
        if keep
    ]

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
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ========================================================
    # 1. Load cached CLIP embeddings
    # ========================================================

    embeddings, rows = load_cache()

    labels_preview = np.array(
        [
            int(row["label"])
            for row in rows
        ]
    )

    print(
        "\nLoaded cached training set:"
    )

    print(
        "Total images:",
        len(rows),
    )

    print(
        "REAL:",
        int(
            (labels_preview == 0).sum()
        ),
    )

    print(
        "FAKE:",
        int(
            (labels_preview == 1).sum()
        ),
    )

    # ========================================================
    # 2. Fixed 80 / 20 split
    # ========================================================

    rng = np.random.default_rng(
        42
    )

    permutation = rng.permutation(
        len(rows)
    )

    val_size = max(
        1,
        int(
            len(rows) * 0.2
        ),
    )

    val_indices = permutation[
        :val_size
    ]

    train_indices = permutation[
        val_size:
    ]

    train_embeddings = embeddings[
        train_indices
    ]

    val_embeddings = embeddings[
        val_indices
    ]

    train_rows = [
        rows[i]
        for i in train_indices
    ]

    val_rows = [
        rows[i]
        for i in val_indices
    ]

    train_dataset = CachedFusionDataset(
        train_embeddings,
        train_rows,
    )

    val_dataset = CachedFusionDataset(
        val_embeddings,
        val_rows,
    )

    # Keep batches as Python lists because original
    # images may have different dimensions.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    # ========================================================
    # 3. Build TRAINABLE SRM stream
    # ========================================================

    print(
        "\nBuilding SRM frequency stream "
        "(TRAINABLE, from scratch)..."
    )

    frequency = FrequencyStream(
        "srm"
    ).to(device)

    frequency.train()

    # IMPORTANT:
    # Unlike your old train_fusion.py,
    # we DO NOT load v1_srm_best.pt
    # and DO NOT freeze the SRM branch.

    # ========================================================
    # 4. Build TRAINABLE fusion head
    # ========================================================

    print(
        "Building fusion head "
        "(TRAINABLE)..."
    )

    fusion = FusionHead().to(
        device
    )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        list(
            frequency.parameters()
        )
        + list(
            fusion.parameters()
        ),
        lr=args.lr,
    )

    to_tensor = ToTensor()

    # ========================================================
    # Helper: construct one batch
    # ========================================================

    def get_features(batch):
        spatial_embeddings = []
        frequency_embeddings = []
        energies = []
        labels = []

        for item in batch:
            # ------------------------------------------
            # Cached CLIP feature
            # ------------------------------------------

            spatial_embedding = (
                item[
                    "spatial_embedding"
                ]
                .unsqueeze(0)
                .to(device)
            )

            # Match the normalization used by
            # your previous fusion training/eval.
            spatial_embedding = F.normalize(
                spatial_embedding,
                dim=1,
            )

            # ------------------------------------------
            # Original image -> trainable SRM
            # ------------------------------------------

            image = Image.open(
                item["path"]
            ).convert("RGB")

            image_tensor = (
                to_tensor(image)
                .unsqueeze(0)
                .to(device)
            )

            frequency_embedding, energy = (
                frequency(
                    image_tensor
                )
            )

            spatial_embeddings.append(
                spatial_embedding.squeeze(0)
            )

            frequency_embeddings.append(
                frequency_embedding.squeeze(0)
            )

            energies.append(
                energy.squeeze(0)
            )

            labels.append(
                item["label"]
            )

        spatial_embeddings = (
            torch.stack(
                spatial_embeddings
            )
        )

        frequency_embeddings = (
            torch.stack(
                frequency_embeddings
            )
        )

        energies = torch.stack(
            energies
        )

        labels = torch.tensor(
            labels,
            dtype=torch.float32,
            device=device,
        )

        return (
            spatial_embeddings,
            frequency_embeddings,
            energies,
            labels,
        )

    # ========================================================
    # 5. Training
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        frequency.train()
        fusion.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            (
                spatial_embeddings,
                frequency_embeddings,
                energies,
                labels,
            ) = get_features(
                batch
            )

            logits = fusion(
                spatial_embeddings,
                frequency_embeddings,
                energies,
            ).squeeze(1)

            loss = loss_fn(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = len(
                labels
            )

            train_loss += (
                loss.item()
                * batch_size
            )

            predictions = (
                logits > 0
            ).float()

            train_correct += (
                predictions
                == labels
            ).sum().item()

            train_total += (
                batch_size
            )

        train_loss /= (
            train_total
        )

        train_accuracy = (
            train_correct
            / train_total
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        frequency.eval()
        fusion.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                (
                    spatial_embeddings,
                    frequency_embeddings,
                    energies,
                    labels,
                ) = get_features(
                    batch
                )

                logits = fusion(
                    spatial_embeddings,
                    frequency_embeddings,
                    energies,
                ).squeeze(1)

                loss = loss_fn(
                    logits,
                    labels,
                )

                batch_size = len(
                    labels
                )

                val_loss += (
                    loss.item()
                    * batch_size
                )

                predictions = (
                    logits > 0
                ).float()

                val_correct += (
                    predictions
                    == labels
                ).sum().item()

                val_total += (
                    batch_size
                )

        val_loss /= (
            val_total
        )

        val_accuracy = (
            val_correct
            / val_total
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f}"
        )

        # ----------------------------------------------------
        # Best checkpoint by VALIDATION LOSS
        # ----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            torch.save(
                {
                    "frequency": (
                        frequency.state_dict()
                    ),
                    "fusion": (
                        fusion.state_dict()
                    ),
                    "freq_mode": "srm",
                    "best_epoch": (
                        best_epoch
                    ),
                    "best_val_loss": (
                        best_val_loss
                    ),
                    "trained_epochs": (
                        args.epochs
                    ),
                },
                CHECKPOINT_PATH,
            )

            print(
                "  -> best checkpoint saved"
            )

    # ========================================================
    # Done
    # ========================================================

    print(
        "\nSHARED-DATA FUSION "
        "TRAINING COMPLETE"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        "Saved model:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()