"""
Training entry point.

Objective (V3, our main proposed method -- see README roadmap for
V0-V2):

    L = L_classification(y_hat, y) + L_classification(y_hat_t, y)
        + lambda * L_consistency(y_hat, y_hat_t)

where y_hat = Detector.predict(image), y_hat_t = Detector.predict(transform(image))
using the SAME weights, and L_consistency penalizes the two predictions
for disagreeing (e.g. MSE or symmetric KL on the probabilities). lambda
is config.CONSISTENCY_LOSS_WEIGHT -- log L_classification and
L_consistency separately during training, not just their sum: a
consistency term weighted too heavily can collapse both predictions
toward an uninformative middle value instead of actually improving
robustness.

Roadmap (build and evaluate in this order; keep every checkpoint and
its metrics under results/, never overwrite):
    V0  spatial stream only                    (baseline)
    V1  + frequency stream                     (does fusion help?)
    V2  + transform-aware augmentation         (does robustness improve?)
    V3  + consistency loss                     (our main method)
    V4  optional: learned gating / partial backbone fine-tuning

TODO: implement once data/datasets.py and the transform pipeline exist.
Depends on cache_embeddings.py if training against precomputed spatial
embeddings rather than running CLIP live (see that file for why).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def accuracy_from_logits(logits, labels):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    return (predictions == labels).float().mean().item()


def main():
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # 1. Load cached CLIP embeddings
    # ---------------------------------------------------------
    cache_path = Path("cache/spatial_embeddings/cifake_train.pt")

    data = torch.load(cache_path)

    embeddings = data["embeddings"].float()
    labels = data["labels"].float()

    # Normalize CLIP embeddings before classification
    embeddings = F.normalize(embeddings, dim=1)

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)
    print("REAL:", (labels == 0).sum().item())
    print("FAKE:", (labels == 1).sum().item())

    # ---------------------------------------------------------
    # 2. Create 80/20 train-validation split
    # ---------------------------------------------------------
    num_samples = len(labels)

    indices = torch.randperm(num_samples)

    split_point = int(0.8 * num_samples)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_dataset = TensorDataset(
        embeddings[train_indices],
        labels[train_indices],
    )

    val_dataset = TensorDataset(
        embeddings[val_indices],
        labels[val_indices],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # ---------------------------------------------------------
    # 3. V0 classifier
    # frozen CLIP embedding (512-d) -> one AI/REAL logit
    # ---------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    classifier = nn.Linear(512, 1).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=1e-3,
    )

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    epochs = 20

    best_val_accuracy = 0.0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):

        # ----- TRAIN -----
        classifier.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = classifier(x).squeeze(1)

            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

            predictions = (torch.sigmoid(logits) >= 0.5).float()

            train_correct += (predictions == y).sum().item()
            train_total += len(y)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # ----- VALIDATION -----
        classifier.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = classifier(x).squeeze(1)

                loss = loss_fn(logits, y)

                val_loss += loss.item() * len(y)

                predictions = (
                    torch.sigmoid(logits) >= 0.5
                ).float()

                val_correct += (predictions == y).sum().item()
                val_total += len(y)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                classifier.state_dict(),
                checkpoint_dir / "v0_spatial_best.pt",
            )

    print("\nV0 TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print("Saved model: checkpoints/v0_spatial_best.pt")


if __name__ == "__main__":
    main()