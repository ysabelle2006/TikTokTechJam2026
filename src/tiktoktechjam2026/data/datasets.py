"""
Dataset loader for V0 / V1: SID_Set, binary real vs fully-synthetic.

SID_Set ships three classes (real / full_synthetic / tampered). V0 and V1
ask the narrow question "do the streams separate genuine photos from fully
generated images at all", so `tampered` is dropped here (config.SID_CLASS_TO_LABEL).

The split is deterministic: a stratified 80/10/10 train/val/test partition
seeded by config.SEED, materialized once to config.SPLIT_FILE so that
cache_embeddings.py, train.py and evaluate.py all see the exact same
test set. The organizer demo set (COCO val2017 + DALL-E Advanced) is never
touched here.

CLI:
    python -m tiktoktechjam2026.data.datasets --build      # (re)build the split file
    python -m tiktoktechjam2026.data.datasets --summary    # print split class balance
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from PIL import Image

from tiktoktechjam2026 import config

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_class_images(class_name: str) -> list[str]:
    folder = os.path.join(config.SID_DIR, class_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"expected SID_Set class folder at {folder!r} -- run downloads.py first"
        )
    names = sorted(
        n for n in os.listdir(folder)
        if os.path.splitext(n)[1].lower() in _IMG_EXTS
    )
    return [os.path.join(folder, n) for n in names]


def build_split(force: bool = False) -> dict:
    """
    Build (or load) the deterministic stratified train/val/test split.

    Returns a dict: {"train": [[path, label], ...], "val": [...], "test": [...]}.
    """
    if os.path.exists(config.SPLIT_FILE) and not force:
        with open(config.SPLIT_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)

    import numpy as np

    rng = np.random.default_rng(config.SEED)
    frac_train, frac_val, _ = config.SPLIT_FRACTIONS
    split: dict[str, list] = {"train": [], "val": [], "test": []}

    for class_name, label in sorted(config.SID_CLASS_TO_LABEL.items()):
        paths = _list_class_images(class_name)
        rng.shuffle(paths)
        if config.SID_PER_CLASS_CAP:
            paths = paths[: config.SID_PER_CLASS_CAP]

        n = len(paths)
        n_train = int(round(n * frac_train))
        n_val = int(round(n * frac_val))
        # Stratify by slicing each class the same way -> balanced splits.
        chunks = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        for name, chunk in chunks.items():
            split[name].extend([p, label] for p in chunk)

    # Shuffle within each split so batches mix classes.
    for name in split:
        idx = rng.permutation(len(split[name]))
        split[name] = [split[name][i] for i in idx]

    os.makedirs(os.path.dirname(config.SPLIT_FILE), exist_ok=True)
    with open(config.SPLIT_FILE, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=0)
    return split


@dataclass
class Sample:
    path: str
    label: int


class SidDataset:
    """
    Minimal indexable dataset over one split.

    __getitem__ returns (PIL.Image RGB, label int, path str). Deliberately
    not a torch Dataset subclass -- V0/V1 training reads cached CLIP
    embeddings, not raw images, and evaluate.py wants the path too.
    """

    def __init__(self, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        self.split = split
        rows = build_split()[split]
        self.samples = [Sample(path=p, label=int(y)) for p, y in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        with Image.open(s.path) as im:
            img = im.convert("RGB")
        return img, s.label, s.path

    @property
    def labels(self) -> list[int]:
        return [s.label for s in self.samples]

    @property
    def paths(self) -> list[str]:
        return [s.path for s in self.samples]


def _summary() -> None:
    split = build_split()
    print(f"split file: {config.SPLIT_FILE}")
    for name, rows in split.items():
        labels = [y for _, y in rows]
        counts = {config.LABEL_NAMES[k]: labels.count(k) for k in sorted(set(labels))}
        print(f"  {name:5s}  n={len(rows):6d}  {counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SID_Set split management for V0/V1.")
    ap.add_argument("--build", action="store_true", help="(re)build the split file")
    ap.add_argument("--summary", action="store_true", help="print split class balance")
    args = ap.parse_args()

    if args.build:
        build_split(force=True)
        print(f"wrote {config.SPLIT_FILE}")
    if args.summary or not args.build:
        _summary()


if __name__ == "__main__":
    main()
