from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from tiktoktechjam2026.data.datasets import AIGCFolderDataset
from tiktoktechjam2026.models.frequency_stream import FrequencyStream
from tiktoktechjam2026.models.fusion import FusionHead


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# --------------------------------------------------
# V1 model
# --------------------------------------------------

class V1Model(nn.Module):
    """
    V1 = frozen/cached CLIP spatial embedding
         + trainable frequency stream
         + trainable fusion head
    """

    def __init__(self, frequency_mode):
        super().__init__()

        self.frequency_stream = FrequencyStream(
            mode=frequency_mode
        )

        self.fusion_head = FusionHead()

    def forward(
        self,
        spatial_embedding,
        frequency_input,
    ):
        frequency_embedding, residual_energy = (
            self.frequency_stream(
                frequency_input
            )
        )

        logit = self.fusion_head(
            spatial_embedding,
            frequency_embedding,
            residual_energy,
        )

        return logit


# --------------------------------------------------
# Dataset combining cached CLIP embeddings
# with the corresponding original images
# --------------------------------------------------

class CombinedDataset(Dataset):

    def __init__(
        self,
        indices,
        image_dataset,
        spatial_embeddings,
    ):
        self.indices = list(indices)
        self.image_dataset = image_dataset
        self.spatial_embeddings = spatial_embeddings

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]

        # Original image prepared for frequency stream
        frequency_image, label = (
            self.image_dataset[idx]
        )

        # Matching cached CLIP embedding
        spatial_embedding = (
            self.spatial_embeddings[idx]
        )

        label = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return (
            spatial_embedding,
            frequency_image,
            label,
        )


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_v1(
    frequency_mode="srm",
    epochs=20,
    batch_size=64,
):

    print("Device:", DEVICE)
    print("Frequency mode:", frequency_mode)

    # --------------------------------------------------
    # 1. Load clean cached CLIP embeddings
    # --------------------------------------------------

    cache = torch.load(
        "results/v0/train_embeddings_full.pt",
        map_location="cpu",
    )

    spatial_embeddings = cache["embeddings"]
    cached_labels = cache["labels"]

    print(
        "Spatial embeddings:",
        spatial_embeddings.shape
    )

    print(
        "Cached labels:",
        cached_labels.shape
    )

    # --------------------------------------------------
    # 2. Load CIFAKE images for frequency stream
    #
    # IMPORTANT:
    # Frequency branch receives ordinary RGB [0,1]
    # tensors, NOT CLIP-normalized tensors.
    # --------------------------------------------------

    frequency_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_dataset = AIGCFolderDataset(
        root_dir="data/cifake/train",
        transform=frequency_transform,
    )

    if len(image_dataset) != len(spatial_embeddings):
        raise ValueError(
            "Dataset and cached embedding sizes do not match! "
            f"Images: {len(image_dataset)}, "
            f"Embeddings: {len(spatial_embeddings)}"
        )

    # --------------------------------------------------
    # 3. Verify labels are aligned
    # --------------------------------------------------

    dataset_labels = torch.tensor(
        [
            label
            for _, label in image_dataset.samples
        ],
        dtype=cached_labels.dtype,
    )

    if not torch.equal(
        dataset_labels,
        cached_labels.cpu(),
    ):
        raise ValueError(
            "Cached CLIP embeddings and dataset labels "
            "are not aligned. Do NOT train."
        )

    print("Cache alignment check: PASSED")

    # --------------------------------------------------
    # 4. Same deterministic 90/10 split
    # --------------------------------------------------

    generator = torch.Generator().manual_seed(42)

    train_size = int(
        0.9 * len(image_dataset)
    )

    val_size = (
        len(image_dataset) - train_size
    )

    train_indices, val_indices = random_split(
        range(len(image_dataset)),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = CombinedDataset(
        indices=train_indices,
        image_dataset=image_dataset,
        spatial_embeddings=spatial_embeddings,
    )

    val_dataset = CombinedDataset(
        indices=val_indices,
        image_dataset=image_dataset,
        spatial_embeddings=spatial_embeddings,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    # --------------------------------------------------
    # 5. Create model
    # --------------------------------------------------

    model = V1Model(
        frequency_mode=frequency_mode
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    # --------------------------------------------------
    # 6. Output folder
    # --------------------------------------------------

    output_dir = Path(
        f"results/v1_{frequency_mode}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_accuracy = 0.0

    # --------------------------------------------------
    # 7. Training loop
    # --------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (
            spatial_embedding,
            frequency_image,
            label,
        ) in enumerate(train_loader):

            spatial_embedding = (
                spatial_embedding.to(DEVICE)
            )

            frequency_image = (
                frequency_image.to(DEVICE)
            )

            label = label.to(DEVICE)

            # ------------------------------
            # Forward
            # ------------------------------

            optimizer.zero_grad()

            logits = model(
                spatial_embedding,
                frequency_image,
            ).squeeze(1)

            loss = criterion(
                logits,
                label,
            )

            # ------------------------------
            # Backpropagation
            # ------------------------------

            loss.backward()

            optimizer.step()

            # ------------------------------
            # Statistics
            # ------------------------------

            running_loss += (
                loss.item()
                * label.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).long()

            total_correct += (
                predictions
                == label.long()
            ).sum().item()

            total_samples += label.size(0)

            # ------------------------------
            # Progress display
            # ------------------------------

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1:02d} | "
                    f"Batch {batch_idx + 1} / "
                    f"{len(train_loader)}"
                )

        train_loss = (
            running_loss
            / total_samples
        )

        train_accuracy = (
            total_correct
            / total_samples
        )

        # --------------------------------------------------
        # 8. Validation
        # --------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for (
                spatial_embedding,
                frequency_image,
                label,
            ) in val_loader:

                spatial_embedding = (
                    spatial_embedding.to(DEVICE)
                )

                frequency_image = (
                    frequency_image.to(DEVICE)
                )

                label = label.to(DEVICE)

                logits = model(
                    spatial_embedding,
                    frequency_image,
                ).squeeze(1)

                probabilities = torch.sigmoid(
                    logits
                )

                predictions = (
                    probabilities >= 0.5
                ).long()

                val_correct += (
                    predictions
                    == label.long()
                ).sum().item()

                val_total += label.size(0)

        val_accuracy = (
            val_correct
            / val_total
        )

        # --------------------------------------------------
        # 9. Epoch results
        # --------------------------------------------------

        print()
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_accuracy:.4f} | "
            f"Val {val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # 10. Save best checkpoint
        # --------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint = {
                "frequency_mode":
                    frequency_mode,

                "frequency_stream":
                    model
                    .frequency_stream
                    .state_dict(),

                "fusion_head":
                    model
                    .fusion_head
                    .state_dict(),

                "val_accuracy":
                    val_accuracy,

                "epoch":
                    epoch + 1,
            }

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )

            print(
                "Saved new best model!"
            )

        print()

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print("=" * 50)
    print("Training finished!")
    print("Frequency mode:", frequency_mode)
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )
    print(
        "Best checkpoint:",
        output_dir / "best.pt"
    )
    print("=" * 50)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    train_v1(
        frequency_mode="fft",
        epochs=20,
        batch_size=64,
    )