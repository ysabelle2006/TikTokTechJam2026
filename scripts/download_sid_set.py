"""
Download a hackathon-sized slice of SID_Set from Hugging Face.

No login required. Requires the `datasets` package:
    pip install datasets pillow

Pulls a subset of the train split (default 3000 rows) and a subset of
the validation split (default 1000 rows out of the full 30k), keeps
only label 0 (real, from OpenImages) and label 1 (fully AI-generated)
-- label 2 is locally-tampered/manipulated images, a different task
we're not tackling here -- and writes:

    data/train/sid_set/
      real/*.jpg
      fake/*.jpg
      manifest.csv   (columns: filename, label, split)

Run with:  python scripts/download_sid_set.py

Not yet verified against the real dataset (no network access from the
environment this was written in) -- run it and share whatever it
prints or any error, and we'll fix it together rather than you
debugging alone. In particular: the label integers (0/1/2) and their
meaning are taken from documentation, not a live check, so this prints
a label-count summary up front -- sanity check that against what you'd
expect before trusting the output.
"""

import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/train/sid_set")
TRAIN_SUBSET_SIZE = 3000   # out of 210,000 -- plenty for V0-V3, raise later if needed
VAL_SUBSET_SIZE = 1000     # out of 30,000

LABEL_NAMES = {0: "real", 1: "fake"}  # label 2 (tampered) is intentionally skipped
OVERFETCH_FACTOR = 3       # fetch extra rows since some will be label==2 and get dropped


def save_split(split_name: str, subset_size: int, manifest_writer: "csv._writer") -> None:
    fetch_n = subset_size * OVERFETCH_FACTOR
    ds = load_dataset("saberzl/SID_Set", split=f"{split_name}[:{fetch_n}]")

    label_counts = Counter(ds["label"])
    print(f"{split_name}: fetched {len(ds)} rows, label counts: {dict(label_counts)}")

    kept = 0
    for i, row in enumerate(ds):
        if kept >= subset_size:
            break
        label = row["label"]
        if label not in LABEL_NAMES:
            continue
        out_name = f"{split_name}_{i:06d}.jpg"
        out_path = OUT_DIR / LABEL_NAMES[label] / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(out_path, "JPEG", quality=95)
        manifest_writer.writerow([str(out_path.relative_to(OUT_DIR)), label, split_name])
        kept += 1

    if kept < subset_size:
        print(f"  WARNING: only found {kept}/{subset_size} real+fake rows in the first "
              f"{fetch_n} -- raise OVERFETCH_FACTOR and re-run if you need the full amount.")
    else:
        print(f"  saved {kept} images (label 0/1 only)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "split"])
        save_split("train", TRAIN_SUBSET_SIZE, writer)
        save_split("validation", VAL_SUBSET_SIZE, writer)
    print(f"\nDone. Manifest at {OUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
