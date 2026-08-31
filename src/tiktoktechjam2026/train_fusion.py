import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import ResidualFusionHead


CACHE_DIR = Path("cache/spatial_embeddings")

EMBEDDING_PATH = CACHE_DIR / "train_clean.npy"
INDEX_PATH = CACHE_DIR / "train_clean_index.csv"

SPATIAL_CHECKPOINT = Path(
    "checkpoints/v0_spatial_sharedtrain_best.pt"
)

OUTPUT_CHECKPOINT = Path(
    "checkpoints/v2_residual_fusion_fft_best.pt"
)


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

        return {
            "spatial_embedding": spatial_embedding,
            "path": row["path"],
            "label": float(row["label"]),
        }


def load_cache():
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

    valid_mask = ~np.isnan(
        embeddings
    ).any(axis=1)

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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

    parser.add_argument(
    "--limit",
        type=int,
        default=None,
        help="Use only the first N cached rows for a quick smoke test.",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ========================================================
    # 1. Load cached CLIP embeddings
    # ========================================================

    embeddings, rows = load_cache()

    if args.limit is not None:
        embeddings = embeddings[:args.limit]
        rows = rows[:args.limit]

        print(
            f"LIMIT ENABLED: using only {len(rows)} images"
        )

    labels_preview = np.array(
        [
            int(row["label"])
            for row in rows
        ]
    )

    print("\nLoaded cached training set:")
    print("Total images:", len(rows))
    print(
        "REAL:",
        int((labels_preview == 0).sum()),
    )
    print(
        "FAKE:",
        int((labels_preview == 1).sum()),
    )

    # ========================================================
    # 2. Fixed 80/20 split
    # ========================================================

    rng = np.random.default_rng(42)

    permutation = rng.permutation(
        len(rows)
    )

    val_size = max(
        1,
        int(len(rows) * 0.2),
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
    # 3. Load GOOD spatial classifier and FREEZE it
    # ========================================================

    print(
        "\nLoading frozen spatial classifier..."
    )

    spatial_head = SpatialHead().to(device)

    spatial_head.load_state_dict(
        torch.load(
            SPATIAL_CHECKPOINT,
            map_location=device,
        )
    )

    spatial_head.eval()

    for parameter in spatial_head.parameters():
        parameter.requires_grad = False

    # ========================================================
    # 4. Trainable SRM
    # ========================================================

    print(
        "Building trainable FFT frequency stream..."
    )

    frequency = FrequencyStream(
    "fft"
    ).to(device)

    # ========================================================
    # 5. Residual fusion correction
    # ========================================================

    print(
        "Building residual fusion head..."
    )

    fusion = ResidualFusionHead().to(
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
    # Batch helper
    # ========================================================

    def get_features(batch):
        from collections import defaultdict

        spatial_logits = []
        labels = []

        # --------------------------------------------------
        # 1. Spatial logits are cheap because CLIP is cached
        # --------------------------------------------------

        spatial_embeddings = torch.stack(
            [
                item["spatial_embedding"]
                for item in batch
            ]
        ).to(device)

        with torch.no_grad():
            spatial_logits = spatial_head(
                spatial_embeddings
            ).unsqueeze(1)

        # --------------------------------------------------
        # 2. Load raw images and group by image dimensions
        #
        # Images in one torch batch must have the same H/W.
        # Grouping lets us batch SRM without resizing and
        # therefore WITHOUT changing the model behaviour.
        # --------------------------------------------------

        groups = defaultdict(list)

        for i, item in enumerate(batch):
            image = Image.open(
                item["path"]
            ).convert("RGB")

            image_tensor = to_tensor(
                image
            )

            shape = tuple(
                image_tensor.shape
            )

            groups[shape].append(
                (i, image_tensor)
            )

        frequency_outputs = [
            None
        ] * len(batch)

        energy_outputs = [
            None
        ] * len(batch)

        # --------------------------------------------------
        # 3. Run SRM in REAL batches
        # --------------------------------------------------

        for shape, items in groups.items():

            indices = [
                i
                for i, _ in items
            ]

            tensors = torch.stack(
                [
                    tensor
                    for _, tensor in items
                ]
            ).to(device)

            group_embeddings, group_energies = (
                frequency(tensors)
            )

            for local_i, original_i in enumerate(
                indices
            ):
                frequency_outputs[
                    original_i
                ] = group_embeddings[
                    local_i:local_i + 1
                ]

                energy_outputs[
                    original_i
                ] = group_energies[
                    local_i:local_i + 1
                ]

        # --------------------------------------------------
        # 4. Restore original batch order
        # --------------------------------------------------

        frequency_embeddings = torch.cat(
            frequency_outputs,
            dim=0,
        )

        energies = torch.cat(
            energy_outputs,
            dim=0,
        )

        labels = torch.tensor(
            [
                item["label"]
                for item in batch
            ],
            dtype=torch.float32,
            device=device,
        )

        return (
            spatial_logits,
            frequency_embeddings,
            energies,
            labels,
        )

    # ========================================================
    # 6. Train
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    OUTPUT_CHECKPOINT.parent.mkdir(
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
                spatial_logits,
                frequency_embeddings,
                energies,
                labels,
            ) = get_features(
                batch
            )

            logits = fusion(
                spatial_logits,
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
                    spatial_logits,
                    frequency_embeddings,
                    energies,
                    labels,
                ) = get_features(
                    batch
                )

                logits = fusion(
                    spatial_logits,
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
            f"val acc {val_accuracy:.3f} | "
            f"alpha {fusion.alpha.item():.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "frequency": (
                        frequency.state_dict()
                    ),
                    "fusion": (
                        fusion.state_dict()
                    ),
                    "spatial_checkpoint": str(
                        SPATIAL_CHECKPOINT
                    ),
                    "best_epoch": best_epoch,
                    "best_val_loss": (
                        best_val_loss
                    ),
                    "trained_epochs": (
                        args.epochs
                    ),
                },
                OUTPUT_CHECKPOINT,
            )

            print(
                "  -> best checkpoint saved"
            )

    print(
        "\nRESIDUAL FUSION TRAINING COMPLETE"
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}",
    )

    print(
        "Saved model:",
        OUTPUT_CHECKPOINT,
    )


if __name__ == "__main__":
    main()