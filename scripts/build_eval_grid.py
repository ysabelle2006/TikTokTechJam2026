"""
Materializes the robustness evaluation grid: for a sample of
validation_demo images, applies every transform x severity from the
brief's grid PLUS a handful of compound (stacked) conditions, and
saves each result to disk, plus data/eval_manifest.csv that
evaluate.py will read from later to build the robustness table.

19 conditions total per sampled image: clean + 4 JPEG qualities + 3
blur sigmas + 2 resize scales + 3 noise sigmas + color jitter + center
crop (the original 15), plus 4 compound conditions chaining 2-3
transforms in sequence (the V2 addition -- see
transforms/augmentations.py's ALL_CONDITIONS for exactly what each one
does and why those four specifically). This script no longer defines
conditions itself; it iterates transforms.augmentations.ALL_CONDITIONS,
which is also what cache_embeddings.py's V2 augmented-variant cache
draws from -- one registry, so a condition name can't quietly mean two
different things in the eval grid vs. what training actually saw.

Why a sample, not every validation_demo image: the full grid is now 19
conditions. Running that against all 5,000+ validation_demo images
would mean tens of thousands of extra files and several GB --
disproportionate for a hackathon-scale robustness check, and not
something evaluate.py needs that much data for. SAMPLE_PER_SOURCE
controls how many images per source get the full treatment; raise it
later if you want tighter statistics and have the disk/time for it.

This only touches validation_demo images (never train) -- it's a
separate thing from training-time augmentation, which cache_embeddings.py
handles for the train split using a SAMPLE of these same named
conditions per image (see that module's cache_augmented_split), not
the full 19-condition treatment every validation image gets here.

Run with:  python scripts/build_eval_grid.py   (from the repo root)
"""

import csv
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.datasets import load_manifest
from transforms.augmentations import ALL_CONDITIONS

OUT_DIR = Path("data/eval_transformed")
EVAL_MANIFEST = Path("data/eval_manifest.csv")
SAMPLE_PER_SOURCE = 300  # images per source that get the FULL transform grid
SEED = 0


def conditions_for(image):
    """Yields (condition_name, make_image) for every condition in
    transforms.augmentations.ALL_CONDITIONS -- make_image is a zero-arg
    callable, not a computed image, so the caller can skip actually
    running the transform for a condition whose output file already
    exists (see main()). Needed because this script has historically
    run over the device bridge in bounded time slices -- without this,
    a rerun after a timeout redoes every already-written image before
    making any new progress. (Not needed when run locally via `uv run`,
    but harmless either way, and it's what makes adding the V2 compound
    conditions to an already-built grid cheap: the 15 pre-existing
    conditions' files already exist and get skipped, only the 4 new
    ones actually run.)
    """
    for name, fn in ALL_CONDITIONS.items():
        yield name, (lambda im=image, fn=fn: fn(im))


def main():
    random.seed(SEED)
    rows = load_manifest(split="validation_demo")
    if not rows:
        print("No validation_demo rows in the manifest -- run `python src/data/datasets.py` first.")
        return

    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    manifest_rows = []
    for source, source_rows in by_source.items():
        sample = random.sample(source_rows, min(SAMPLE_PER_SOURCE, len(source_rows)))
        print(f"[{source}] sampling {len(sample)} of {len(source_rows)} validation_demo images")
        # WildFake (and possibly other sources) stores images under
        # per-batch subfolders that can reuse the same filename -- two
        # DIFFERENT sampled images can share the same Path(...).stem.
        # Naming output files by bare stem alone would let one silently
        # overwrite the other's transformed file, leaving two manifest
        # rows pointing at one (arbitrary-winner) image. Only stems that
        # actually collide within this sample get a disambiguating
        # suffix, so the common case still gets a readable filename.
        stem_counts = Counter(Path(r["path"]).stem for r in sample)
        for r in sample:
            try:
                img = Image.open(r["path"]).convert("RGB")
            except Exception as e:
                print(f"  skipping {r['path']}: {e}")
                continue
            stem = Path(r["path"]).stem
            if stem_counts[stem] > 1:
                stem = f"{stem}_{hashlib.md5(r['path'].encode()).hexdigest()[:8]}"
            for condition, make_img in conditions_for(img):
                out_path = OUT_DIR / source / condition / f"{stem}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.exists():
                    # Write to a temp file then atomically rename into place.
                    # Without this, a run killed mid-save (this script runs in
                    # bounded time slices over the device bridge -- see
                    # conditions_for's docstring) can leave a truncated JPEG
                    # sitting at out_path. exists() alone can't tell a
                    # complete file from a truncated one, so a later resumed
                    # run would skip it forever, and evaluate.py would only
                    # discover the corruption when it tries to decode it.
                    # rename() is atomic on the same filesystem, so out_path
                    # only ever exists in a fully-written state.
                    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                    make_img().save(tmp_path, "JPEG", quality=95)
                    tmp_path.replace(out_path)
                manifest_rows.append(
                    {
                        "original_path": r["path"],
                        "transformed_path": str(out_path),
                        "condition": condition,
                        "label": r["label"],
                        "source": source,
                        "generator": r["generator"],
                    }
                )

    EVAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["original_path", "transformed_path", "condition", "label", "source", "generator"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} transformed images -> {EVAL_MANIFEST}")


if __name__ == "__main__":
    main()
