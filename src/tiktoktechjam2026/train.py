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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


from tiktoktechjam2026.models.v0_classifier import V0Classifier


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load full cached CIFAKE training embeddings
    cache = torch.load(
        "results/v0/train_embeddings_full.pt",
        map_location="cpu"
    )

    embeddings = cache["embeddings"].float()
    labels = cache["labels"].long()

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    dataset = TensorDataset(embeddings, labels)

    # Fixed 90/10 train-validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False
    )

    model = V0Classifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 20

    checkpoint_dir = Path("results/v0")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "v0_classifier_best.pt"

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        # -------------------
        # Training
        # -------------------
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            logits = model(batch_embeddings)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_embeddings.size(0)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == batch_labels
            ).sum().item()

            total += batch_labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        # -------------------
        # Validation
        # -------------------
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_embeddings)
                predictions = logits.argmax(dim=1)

                correct += (
                    predictions == batch_labels
                ).sum().item()

                total += batch_labels.size(0)

        val_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val acc: {val_accuracy:.4f}"
        )

        # Save best model only
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"  ↳ New best model saved "
                f"(val acc = {best_val_accuracy:.4f})"
            )

    print()
    print("Training complete!")
    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", f"{best_val_accuracy:.4f}")
    print("Saved checkpoint to:", checkpoint_path)


if __name__ == "__main__":
    main()