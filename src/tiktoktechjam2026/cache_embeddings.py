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

from pathlib import Path

import random 
import numpy as np 
from tiktoktechjam2026.transforms.augmentations import random_transform

import torch
from torch.utils.data import DataLoader, Subset

from tiktoktechjam2026.data.datasets import AIGCFolderDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream
from tiktoktechjam2026.transforms.preprocessing import prepare_spatial_input


def main(
    image_dir: str,
    cache_file: str,
    max_samples=None,
    augmentation=None,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = AIGCFolderDataset(
        root_dir=image_dir,
        augmentation=augmentation,
        transform=prepare_spatial_input
    )

    if max_samples is not None:
        real_indices = []
        fake_indices = []

        for i, (_, label) in enumerate(dataset.samples):
            if label == 0:
                real_indices.append(i)
            else:
                fake_indices.append(i)

        samples_per_class = max_samples // 2

        selected_indices = (
            real_indices[:samples_per_class]
            + fake_indices[:samples_per_class]
        )

        dataset = Subset(dataset, selected_indices)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    model = SpatialStream(freeze=True)

    all_embeddings = []
    all_labels = []

    for batch_index, (images, labels) in enumerate(loader):
        embeddings = model.encode(images)

        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        if (batch_index + 1) % 50 == 0 or batch_index == 0:
            print(
                f"Processed batch {batch_index + 1} / {len(loader)}"
            )

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    cache_path = Path(cache_file)
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels
        },
        cache_path
    )

    print("Done!")
    print("Saved to:", cache_path)
    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)


if __name__ == "__main__":
    main(
        image_dir="data/cifake/train",
        cache_file="results/v2/train_embeddings_augmented.pt",
        max_samples=None,
        augmentation=random_transform,
        seed=42
    )