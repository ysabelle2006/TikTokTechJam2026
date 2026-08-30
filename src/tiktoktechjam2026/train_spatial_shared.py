import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


CACHE_DIR = Path("cache/spatial_embeddings")

EMBEDDING_PATH = CACHE_DIR / "train_clean.npy"
INDEX_PATH = CACHE_DIR / "train_clean_index.csv"

CHECKPOINT_PATH = Path(
    "checkpoints/v0_spatial_sharedtrain_best.pt"
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


def main():
    torch.manual_seed(42)

    embeddings = np.load(
        EMBEDDING_PATH
    )

    with open(
        INDEX_PATH,
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    labels = np.array(
        [
            int(row["label"])
            for row in rows
        ],
        dtype=np.float32,
    )

    print("Total:", len(labels))
    print("REAL:", int((labels == 0).sum()))
    print("FAKE:", int((labels == 1).sum()))

    rng = np.random.default_rng(42)

    permutation = rng.permutation(
        len(labels)
    )

    val_size = int(
        len(labels) * 0.2
    )

    val_idx = permutation[:val_size]
    train_idx = permutation[val_size:]

    X_train = torch.from_numpy(
        embeddings[train_idx]
    ).float()

    y_train = torch.from_numpy(
        labels[train_idx]
    ).float()

    X_val = torch.from_numpy(
        embeddings[val_idx]
    ).float()

    y_val = torch.from_numpy(
        labels[val_idx]
    ).float()

    model = SpatialHead()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 10
    batch_size = 256

    best_val_loss = float("inf")

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        permutation = torch.randperm(
            len(X_train)
        )

        train_loss = 0.0
        train_correct = 0

        for start in range(
            0,
            len(X_train),
            batch_size,
        ):
            idx = permutation[
                start:start + batch_size
            ]

            optimizer.zero_grad()

            logits = model(
                X_train[idx]
            )

            loss = loss_fn(
                logits,
                y_train[idx],
            )

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item()
                * len(idx)
            )

            train_correct += (
                (logits > 0).float()
                == y_train[idx]
            ).sum().item()

        train_loss /= len(
            X_train
        )

        train_acc = (
            train_correct
            / len(X_train)
        )

        model.eval()

        with torch.no_grad():
            val_logits = model(
                X_val
            )

            val_loss = loss_fn(
                val_logits,
                y_val,
            ).item()

            val_acc = (
                (val_logits > 0).float()
                == y_val
            ).float().mean().item()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH,
            )

            print(
                "  -> best checkpoint saved"
            )

    print(
        "\nSaved:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()