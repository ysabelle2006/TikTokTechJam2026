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

import torch
from tqdm import tqdm

from tiktoktechjam2026.data.datasets import CIFAKEDataset
from tiktoktechjam2026.models.spatial_stream import SpatialStream


def get_balanced_indices(dataset, per_class):
    real_indices = []
    fake_indices = []

    for i, (_, original_label) in enumerate(dataset.dataset.samples):
        class_name = dataset.dataset.classes[original_label].upper()

        if class_name == "REAL" and len(real_indices) < per_class:
            real_indices.append(i)

        elif class_name == "FAKE" and len(fake_indices) < per_class:
            fake_indices.append(i)

        if len(real_indices) == per_class and len(fake_indices) == per_class:
            break

    return real_indices + fake_indices


def main():
    # For now: cache 1000 REAL + 1000 FAKE training images
    split = "test"
    per_class = 1000

    dataset = CIFAKEDataset("data/CIFAKE", split=split)
    spatial = SpatialStream()

    indices = get_balanced_indices(dataset, per_class)

    embeddings = []
    labels = []

    for i in tqdm(indices, desc=f"Caching {split} CLIP embeddings"):
        image, label = dataset[i]

        image_tensor = spatial.preprocess(image).unsqueeze(0)
        embedding = spatial.encode(image_tensor)

        embeddings.append(embedding.squeeze(0).cpu())
        labels.append(label)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype=torch.long)

    cache_dir = Path("cache/spatial_embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / f"cifake_{split}.pt"

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        output_path,
    )

    print("\nDONE")
    print("Saved to:", output_path)
    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    # CIFAKE ImageFolder ordering is FAKE=0, REAL=1
    print("FAKE:", (labels == 0).sum().item())
    print("REAL:", (labels == 1).sum().item())


if __name__ == "__main__":
    main()