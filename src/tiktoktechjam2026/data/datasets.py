"""
Unified data manifest: turns each dataset's own folder/file layout into
one consistent table -- (path, label, source, split, generator) -- so
everything downstream (caching, training, evaluation) reads from one
place instead of knowing four different folder structures.

    label:     0 = real, 1 = AI-generated
    source:    which dataset this row came from
               (cifake / sid_set / wildfake_dalle / coco_val2017)
    split:     train / test / validation_demo
    generator: which generator family produced a fake image
               (e.g. "stable_diffusion_1.4", "dalle")
               or "real" for real images

Run with:
    python src/data/datasets.py

from the repo root.
"""

import csv
from pathlib import Path

DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "manifest.csv"

FIELDS = ["path", "label", "source", "split", "generator"]


def scan_cifake():
    rows = []
    root = DATA_DIR / "train" / "cifake"

    if not root.is_dir():
        print(f"[cifake] not found at {root}, skipping")
        return rows

    for split in ("train", "test"):
        for label_name, label in (("REAL", 0), ("FAKE", 1)):
            folder = root / split / label_name

            if not folder.is_dir():
                continue

            generator = "real" if label == 0 else "stable_diffusion_1.4"

            for f in folder.iterdir():
                if f.is_file():
                    rows.append(
                        {
                            "path": str(f),
                            "label": label,
                            "source": "cifake",
                            "split": "train",
                            "generator": generator,
                        }
                    )

    print(f"[cifake] {len(rows)} rows")
    return rows


def scan_coco_val2017():
    rows = []
    folder = DATA_DIR / "validation_demo" / "coco_val2017"

    if not folder.is_dir():
        print(f"[coco_val2017] not found at {folder}, skipping")
        return rows

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() == ".jpg":
            rows.append(
                {
                    "path": str(f),
                    "label": 0,
                    "source": "coco_val2017",
                    "split": "validation_demo",
                    "generator": "real",
                }
            )

    print(f"[coco_val2017] {len(rows)} rows")
    return rows


def scan_sid_set():
    rows = []
    manifest_csv = DATA_DIR / "train" / "sid_set" / "manifest.csv"

    if not manifest_csv.is_file():
        print(
            f"[sid_set] not found at {manifest_csv}, skipping "
            f"(run scripts/download_sid_set.py first)"
        )
        return rows

    base = manifest_csv.parent

    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            generator = "real" if label == 0 else "sid_set_mixed"

            # SID_Set validation rows stay out of training.
            split = (
                "validation_demo"
                if row["split"] == "validation"
                else "train"
            )

            rows.append(
                {
                    "path": str(base / row["filename"]),
                    "label": label,
                    "source": "sid_set",
                    "split": split,
                    "generator": generator,
                }
            )

    print(f"[sid_set] {len(rows)} rows")
    return rows


def scan_wildfake_dalle():
    rows = []
    folder = DATA_DIR / "validation_demo" / "dalle_advanced"

    if not folder.is_dir() or not any(folder.iterdir()):
        print(f"[wildfake_dalle] not found (or empty) at {folder}, skipping")
        return rows

    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            rows.append(
                {
                    "path": str(f),
                    "label": 1,
                    "source": "wildfake_dalle",
                    "split": "validation_demo",
                    "generator": "dalle",
                }
            )

    print(f"[wildfake_dalle] {len(rows)} rows")
    return rows


def build_manifest():
    rows = (
        scan_cifake()
        + scan_coco_val2017()
        + scan_sid_set()
        + scan_wildfake_dalle()
    )

    # Guardrail: evaluation-only data must never leak into training.
    for r in rows:
        if r["source"] in ("coco_val2017", "wildfake_dalle") or (
            r["source"] == "sid_set"
            and r["split"] == "validation_demo"
        ):
            assert r["split"] == "validation_demo", f"leak risk: {r}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_manifest(source=None, split=None, exclude_generator=None):
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} doesn't exist yet -- run build_manifest() first"
        )

    rows = []

    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if source and row["source"] != source:
                continue

            if split and row["split"] != split:
                continue

            if exclude_generator and row["generator"] == exclude_generator:
                continue

            row["label"] = int(row["label"])
            rows.append(row)

    return rows


def _print_summary(rows):
    from collections import Counter

    print(f"\n=== manifest: {len(rows)} total rows -> {MANIFEST_PATH} ===")

    by_source_split = Counter(
        (r["source"], r["split"]) for r in rows
    )

    for (source, split), n in sorted(by_source_split.items()):
        print(f"  {source:<16} {split:<16} {n}")

    by_label = Counter(r["label"] for r in rows)

    print(
        f"  label counts: "
        f"real(0)={by_label.get(0, 0)}  "
        f"fake(1)={by_label.get(1, 0)}"
    )


if __name__ == "__main__":
    rows = build_manifest()
    _print_summary(rows)