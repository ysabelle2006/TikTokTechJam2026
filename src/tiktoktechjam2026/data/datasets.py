"""
Dataset loaders for the AIGC detection sources named in the brief:
  - CIFAKE (small, fast to iterate on -- good first target)
  - SID_Set
  - WildFake (translate via the ModelScope UI before use, per the brief)

The validation set (COCO val2017 vs. the DALL-E Advanced subset of
WildFake) stays entirely separate: it's for demonstrating progress
only and must never be trained on.

TODO (next step): once we know where the raw data will live locally,
implement a dataset class per source with a consistent (image, label)
interface, plus a generator_family field so we can later hold one
family out for the generalization test.
"""

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CIFAKEDataset(Dataset):
    """
    CIFAKE dataset wrapper.

    Returns:
        image: PIL image, or transformed tensor if a transform is provided
        label: 0 for REAL, 1 for AI/FAKE
    """

    def __init__(self, root, split="train", transform=None):
        split_dir = Path(root) / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Could not find CIFAKE split at: {split_dir}"
            )

        self.dataset = ImageFolder(
            root=split_dir,
            transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, original_label = self.dataset[index]

        class_name = self.dataset.classes[original_label]

        if class_name.upper() == "REAL":
            label = 0
        elif class_name.upper() == "FAKE":
            label = 1
        else:
            raise ValueError(f"Unexpected CIFAKE class: {class_name}")

        return image, label