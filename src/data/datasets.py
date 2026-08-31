"""
Unified data manifest: turns each dataset's own folder/file layout into
one consistent table -- (path, label, source, split, generator) -- so
everything downstream (caching, training, evaluation) reads from one
place instead of knowing four different folder structures.

    label:     0 = real, 1 = AI-generated
    source:    which dataset this row came from (cifake / sid_set / wildfake_dalle / coco_val2017)
    split:     train / test / validation_demo -- validation_demo rows must
               NEVER be used for training (see build_manifest's assertion)
    generator: which generator family produced a fake image (e.g.
               "stable_diffusion_1.4", "dalle") or "real" for real images --
               this is what lets evaluate.py hold out one generator family
               for the generalization test described in the architecture doc

Each scan_*() function is independent and skips gracefully (with a
printed note, not an error) if that source's folder isn't downloaded
yet -- so build_manifest() always produces a valid manifest from
whatever's currently on disk, and picks up new sources automatically
once they're downloaded, with no code changes needed here.

Run with:  python src/data/datasets.py   (from the repo root -- paths
below are relative to the current directory, not this file's location)

Once cache_embeddings.py / train.py exist, they import build_manifest /
load_manifest directly instead of re-scanning from the command line.
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
                    rows.append({"path": str(f), "label": label, "source": "cifake",
                                 "split": "train", "generator": generator})
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
            rows.append({"path": str(f), "label": 0, "source": "coco_val2017",
                         "split": "validation_demo", "generator": "real"})
    print(f"[coco_val2017] {len(rows)} rows")
    return rows


def scan_sid_set():
    rows = []
    manifest_csv = DATA_DIR / "train" / "sid_set" / "manifest.csv"
    if not manifest_csv.is_file():
        print(f"[sid_set] not found at {manifest_csv}, skipping "
              f"(run scripts/download_sid_set.py first)")
        return rows
    base = manifest_csv.parent
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            generator = "real" if label == 0 else "sid_set_mixed"
            # SID_Set's own "validation" rows are a held-out split -- keep
            # them out of the training split so they can be used the same
            # way as coco_val2017/dalle_advanced (see build_manifest below).
            split = "validation_demo" if row["split"] == "validation" else "train"
            rows.append({"path": str(base / row["filename"]), "label": label,
                         "source": "sid_set", "split": split, "generator": generator})
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
            rows.append({"path": str(f), "label": 1, "source": "wildfake_dalle",
                         "split": "validation_demo", "generator": "dalle"})
    print(f"[wildfake_dalle] {len(rows)} rows")
    return rows


def build_manifest():
    rows = scan_cifake() + scan_coco_val2017() + scan_sid_set() + scan_wildfake_dalle()

    # Guardrail matching the brief: coco_val2017 / dalle_advanced / SID_Set's
    # own validation rows must never be trained on. This isn't just a folder
    # convention -- assert it, so a future change can't silently break it.
    for r in rows:
        if r["source"] in ("coco_val2017", "wildfake_dalle") or (
            r["source"] == "sid_set" and r["split"] == "validation_demo"
        ):
            assert r["split"] == "validation_demo", f"leak risk: {r}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_manifest(source=None, split=None, exclude_generator=None):
    """Read the manifest built above, optionally filtered. Used by
    cache_embeddings.py / train.py / evaluate.py once they exist --
    e.g. load_manifest(split="train", exclude_generator="dalle") for the
    generator-family-held-out generalization split described in the
    architecture doc."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"{MANIFEST_PATH} doesn't exist yet -- run build_manifest() first")
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
    by_source_split = Counter((r["source"], r["split"]) for r in rows)
    for (source, split), n in sorted(by_source_split.items()):
        print(f"  {source:<16} {split:<16} {n}")
    by_label = Counter(r["label"] for r in rows)
    print(f"  label counts: real(0)={by_label.get(0, 0)}  fake(1)={by_label.get(1, 0)}")


if __name__ == "__main__":
    rows = build_manifest()
    _print_summary(rows)
