from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor

from tiktoktechjam2026.data.datasets import CIFAKEDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead


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
        transform=None,
    )

    indices = get_balanced_indices(
        dataset,
        per_class=1000,
    )

    subset = Subset(dataset, indices)

    print("Total fusion development images:", len(subset))

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
        collate_fn=lambda batch: batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # ---------------------------------------------------------
    # 3. Load frozen spatial stream
    # ---------------------------------------------------------
    spatial = SpatialStream()
    spatial.model.eval()

    for param in spatial.model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # 4. Load pretrained SRM stream
    # ---------------------------------------------------------
    frequency = FrequencyStream("srm").to(device)

    checkpoint = torch.load(
        "checkpoints/v1_srm_best.pt",
        map_location=device,
    )

    frequency.load_state_dict(
        checkpoint["frequency"]
    )

    frequency.eval()

    for param in frequency.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # 5. Fusion head
    # ---------------------------------------------------------
    fusion = FusionHead().to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        fusion.parameters(),
        lr=1e-3,
    )

    # ---------------------------------------------------------
    # Helper: convert one batch of PIL images into both streams
    # ---------------------------------------------------------
    def get_features(batch):
        spatial_embeddings = []
        frequency_embeddings = []
        energies = []
        labels = []

        for image, label in batch:

            # ----- spatial CLIP input -----
            spatial_tensor = (
                spatial.preprocess(image)
                .unsqueeze(0)
            )

            with torch.no_grad():
                spatial_embedding = spatial.encode(
                    spatial_tensor
                )

                spatial_embedding = F.normalize(
                    spatial_embedding.float(),
                    dim=1,
                )

            # ----- raw [0,1] tensor for SRM -----
            image_tensor = ToTensor()(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                frequency_embedding, energy = frequency(
                    image_tensor
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

            labels.append(label)

        spatial_embeddings = torch.stack(
            spatial_embeddings
        ).to(device)

        frequency_embeddings = torch.stack(
            frequency_embeddings
        ).to(device)

        energies = torch.stack(
            energies
        ).to(device)

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

    # ---------------------------------------------------------
    # 6. Train fusion head only
    # ---------------------------------------------------------
    epochs = 20
    best_val_accuracy = 0.0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(epochs):

        # ----- TRAIN -----
        fusion.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            (
                spatial_embeddings,
                frequency_embeddings,
                energies,
                labels,
            ) = get_features(batch)

            optimizer.zero_grad()

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

            train_loss += (
                loss.item() * len(labels)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += len(labels)

        train_loss /= train_total
        train_accuracy = (
            train_correct / train_total
        )

        # ----- VALIDATION -----
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
                ) = get_features(batch)

                logits = fusion(
                    spatial_embeddings,
                    frequency_embeddings,
                    energies,
                ).squeeze(1)

                loss = loss_fn(
                    logits,
                    labels,
                )

                val_loss += (
                    loss.item() * len(labels)
                )

                predictions = (
                    torch.sigmoid(logits) >= 0.5
                ).float()

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += len(labels)

        val_loss /= val_total
        val_accuracy = (
            val_correct / val_total
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                fusion.state_dict(),
                checkpoint_dir / "v1_fusion_best.pt",
            )

    print("\nV1 FUSION TRAINING COMPLETE")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.3f}"
    )
    print(
        "Saved model: "
        "checkpoints/v1_fusion_best.pt"
    )


if __name__ == "__main__":
    main()