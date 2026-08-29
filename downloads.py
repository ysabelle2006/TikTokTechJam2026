"""
Standalone fetcher for all three TRAINING datasets -- no package install,
no `-m` needed. Puts everything under DATA_DIR in the layout datasets.py
expects:
 
    data/
      cifake/    {train,test}/{REAL,FAKE}/*
      sid_set/   {train,validation}/{real,full_synthetic,tampered}/*
      wildfake/  <hierarchical generator folders>/*
 
The demo/eval set (COCO val2017 + DALL-E Advanced) is NOT fetched here -- it
is organizer-provided and must never be trained on. Place it yourself at
data/demo/coco_val2017/ and data/demo/dalle_advanced/.
 
Run from your project root:
    python fetch_datasets.py --all
    python fetch_datasets.py --cifake
    python fetch_datasets.py --sid --sid-cap 8000
    python fetch_datasets.py --wildfake
 
Prereqs (install into whatever interpreter you run this with):
    pip install kagglehub huggingface_hub datasets pillow modelscope
 
Credentials:
  * Kaggle : set KAGGLE_USERNAME / KAGGLE_KEY (kaggle.com -> Account -> API token).
  * HF     : usually anonymous; `huggingface-cli login` only if rate-limited.
  * ModelScope: usually anonymous for public datasets.
 
These are real, multi-GB downloads -- run from a normal terminal, not a
sandboxed shell. You can Ctrl-C a source mid-download and keep what arrived.
"""
 
from __future__ import annotations
 
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
 
DATA_DIR = Path("data")
 
 
# ---------------------------------------------------------------------------
# CIFAKE  (Kaggle -> data/cifake/{train,test}/{REAL,FAKE})
# ---------------------------------------------------------------------------
 
def fetch_cifake(data_dir: Path = DATA_DIR) -> None:
    import kagglehub
    print("\n=== CIFAKE (Kaggle) ===")
    src = Path(kagglehub.dataset_download(
        "birdy654/cifake-real-and-ai-generated-synthetic-images"))
    print("kagglehub cache:", src)
 
    dst = data_dir / "cifake"
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        cand = next((m for m in src.rglob(split)
                     if (m / "REAL").exists() or (m / "FAKE").exists()), None)
        if cand is None:
            print(f"  WARNING: couldn't locate '{split}' under {src}")
            continue
        target = dst / split
        if target.exists():
            print(f"  {target} exists, skipping")
            continue
        print(f"  copying {split} -> {target}")
        shutil.copytree(cand, target)
    print(f"  CIFAKE ready at {dst.resolve()}")
 
 
# ---------------------------------------------------------------------------
# SID_Set  (Hugging Face, streamed -> data/sid_set/{split}/{class}/*.jpg)
#
# Streaming avoids downloading all 300k images and works whether the repo is
# stored as image folders or parquet. We materialize a capped subset per class
# into the folder names datasets.py keys on (real / full_synthetic / tampered).
# ---------------------------------------------------------------------------
 
def _sid_class_folder(name: str) -> str | None:
    n = str(name).lower()
    if "mask" in n:
        return None                 # segmentation masks, not detector inputs
    if "real" in n or n in ("0", "authentic", "genuine"):
        return "real"
    if "synth" in n or n == "1":
        return "full_synthetic"
    if "tamper" in n:
        return "tampered"
    return n or None
 
 
def _first_image_field(example: dict):
    from PIL import Image as PILImage
    if "image" in example and isinstance(example["image"], PILImage.Image):
        return example["image"]
    for v in example.values():
        if isinstance(v, PILImage.Image):
            return v
    return None
 
 
def fetch_sid(data_dir: Path = DATA_DIR, cap_per_class: int = 8000) -> None:
    """cap_per_class=0 means no cap (materialize everything -- large & slow)."""
    from datasets import load_dataset
    print("\n=== SID_Set (Hugging Face, streamed) ===")
    dst = data_dir / "sid_set"
 
    want = {"real", "full_synthetic", "tampered"}
    for split_src, split_dst in (("train", "train"), ("validation", "validation")):
        try:
            ds = load_dataset("saberzl/SID_Set", split=split_src, streaming=True)
        except Exception as e:
            print(f"  [{split_src}] could not open ({e}); skipping.")
            continue
 
        # Resolve a ClassLabel feature -> string names, if present.
        label_feat = None
        try:
            label_feat = ds.features.get("label") if ds.features else None
        except Exception:
            pass
 
        counts: dict[str, int] = {}
        print(f"  [{split_src}] streaming...")
        for ex in ds:
            raw_label = ex.get("label", ex.get("category", ex.get("class")))
            if label_feat is not None and isinstance(raw_label, int):
                try:
                    raw_label = label_feat.int2str(raw_label)
                except Exception:
                    pass
            folder = _sid_class_folder(raw_label)
            if folder is None:
                continue
            if cap_per_class and counts.get(folder, 0) >= cap_per_class:
                # stop once every wanted class is full
                if all(counts.get(w, 0) >= cap_per_class for w in want):
                    break
                continue
 
            img = _first_image_field(ex)
            if img is None:
                continue
            out_dir = dst / split_dst / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = counts.get(folder, 0)
            try:
                img.convert("RGB").save(out_dir / f"{folder}_{idx:06d}.jpg", quality=95)
                counts[folder] = idx + 1
            except Exception:
                continue
        print(f"  [{split_src}] saved: {counts}")
    print(f"  SID_Set ready at {dst.resolve()}")
 
 
# ---------------------------------------------------------------------------
# WildFake  (ModelScope -> data/wildfake/<hierarchy>)
# ---------------------------------------------------------------------------
 
def fetch_wildfake(data_dir: Path = DATA_DIR) -> None:
    print("\n=== WildFake (ModelScope) ===")
    dst = data_dir / "wildfake"
    dst.mkdir(parents=True, exist_ok=True)
 
    # Preferred: the dataset snapshot API.
    try:
        from modelscope import dataset_snapshot_download
        dataset_snapshot_download("hy2628982280/WildFake", local_dir=str(dst))
        print(f"  WildFake ready at {dst.resolve()}")
        print("  -> inspect the folder tree and make sure the generator keywords "
              "in datasets.py match the real subfolder names.")
        return
    except Exception as e:
        print(f"  python API unavailable ({e}); trying the CLI...")
 
    # Fallback: the modelscope CLI.
    try:
        subprocess.run(["modelscope", "download", "--dataset",
                        "hy2628982280/WildFake", "--local_dir", str(dst)], check=True)
        print(f"  WildFake ready at {dst.resolve()}")
        return
    except Exception as e:
        print(f"  CLI failed ({e}). Run this manually once modelscope is installed:")
        print(f"    modelscope download --dataset hy2628982280/WildFake --local_dir {dst}")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch all AIGC training datasets.")
    ap.add_argument("--all", action="store_true", help="fetch all three sources")
    ap.add_argument("--cifake", action="store_true")
    ap.add_argument("--sid", action="store_true")
    ap.add_argument("--sid-cap", type=int, default=8000,
                    help="max images per SID_Set class (0 = no cap; large)")
    ap.add_argument("--wildfake", action="store_true")
    ap.add_argument("--data-dir", default="data", help="root to download into")
    args = ap.parse_args()
 
    data_dir = Path(args.data_dir)
 
    if not any([args.all, args.cifake, args.sid, args.wildfake]):
        ap.print_help()
        return
 
    if args.all or args.cifake:
        fetch_cifake(data_dir)
    if args.all or args.sid:
        fetch_sid(data_dir, cap_per_class=args.sid_cap)
    if args.all or args.wildfake:
        fetch_wildfake(data_dir)
 
    print("\nDone. Next: build the manifest and check class balance:")
    print("    python src/tiktoktechjam2026/data/datasets.py --build --cap 12000")
 

if __name__ == "__main__":
    main()