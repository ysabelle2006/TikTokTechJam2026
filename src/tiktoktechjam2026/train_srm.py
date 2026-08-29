from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor

from tiktoktechjam2026.data.datasets import CIFAKEDataset
from tiktoktechjam2026.models.frequency_stream import FrequencyStream


def get_balanced_indices(dataset, per_class):
    real_indices = []
    fake_indices = []

    for i, (_, original_label) in enumerate(dataset.dataset.samples):
        class_name = dataset.dataset.classes[original_label].upper()

        if class_name == "REAL" and len(real_indices) < per_class:
            real_indices.append(i)

        elif class_name == "FAKE" and len(fake_indices) < per_class:
            fake_indices.append(i)

        if (
            len(real_indices) == per_class
            and len(fake_indices) == per_class
        ):
            break

    return real_indices + fake_indices


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # 1. Load CIFAKE TRAIN split
    # ---------------------------------------------------------
    dataset = CIFAKEDataset(
        "data/CIFAKE",
        split="train",
        transform=ToTensor(),
    )

    # Small development experiment:
    # 1000 REAL + 1000 FAKE
    indices = get_balanced_indices(
        dataset,
        per_class=1000,
    )

    subset = Subset(dataset, indices)

    print("Total SRM development images:", len(subset))

    # ---------------------------------------------------------
    # 2. Fixed 80/20 train-validation split
    # ---------------------------------------------------------
    generator = torch.Generator().manual_seed(42)

    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        subset,
        [train_size, val_size],
        generator=generator,
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
    # 3. SRM frequency model + classifier
    # ---------------------------------------------------------
    frequency = FrequencyStream("srm").to(device)

    classifier = nn.Linear(128, 1).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        list(frequency.parameters())
        + list(classifier.parameters()),
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
        frequency.train()
        classifier.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            embeddings, _ = frequency(images)

            logits = classifier(embeddings).squeeze(1)

            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += len(labels)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # ----- VALIDATION -----
        frequency.eval()
        classifier.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                embeddings, _ = frequency(images)

                logits = classifier(embeddings).squeeze(1)

                loss = loss_fn(logits, labels)

                val_loss += loss.item() * len(labels)

                predictions = (
                    torch.sigmoid(logits) >= 0.5
                ).float()

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += len(labels)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f}"
        )

        # Save BOTH frequency CNN and classifier
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "frequency": frequency.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                checkpoint_dir / "v1_srm_best.pt",
            )

    print("\nSRM TRAINING COMPLETE")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.3f}"
    )
    print(
        "Saved model: "
        "checkpoints/v1_srm_best.pt"
    )


if __name__ == "__main__":
    main()