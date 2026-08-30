import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from tiktoktechjam2026.data.datasets import load_manifest
from tiktoktechjam2026.transforms.augmentations import ALL_CONDITIONS


OUT_DIR = Path("data/eval_transformed")
EVAL_MANIFEST = Path("data/eval_manifest.csv")

SAMPLE_PER_SOURCE = 300
SEED = 0


def conditions_for(image):
    for name, fn in ALL_CONDITIONS.items():
        yield name, (lambda im=image, fn=fn: fn(im))


def main():
    random.seed(SEED)

    rows = load_manifest(split="validation_demo")

    if not rows:
        print("No validation_demo rows found.")
        return

    by_source = {}

    for row in rows:
        by_source.setdefault(
            row["source"],
            [],
        ).append(row)

    manifest_rows = []

    for source, source_rows in by_source.items():

        sample = random.sample(
            source_rows,
            min(
                SAMPLE_PER_SOURCE,
                len(source_rows),
            ),
        )

        print(
            f"[{source}] sampling "
            f"{len(sample)} of "
            f"{len(source_rows)} validation images"
        )

        stem_counts = Counter(
            Path(row["path"]).stem
            for row in sample
        )

        for row in sample:

            try:
                image = Image.open(
                    row["path"]
                ).convert("RGB")

            except Exception as e:
                print(
                    f"Skipping {row['path']}: {e}"
                )
                continue

            stem = Path(
                row["path"]
            ).stem

            if stem_counts[stem] > 1:
                suffix = hashlib.md5(
                    row["path"].encode()
                ).hexdigest()[:8]

                stem = (
                    f"{stem}_{suffix}"
                )

            for condition, make_image in conditions_for(
                image
            ):

                out_path = (
                    OUT_DIR
                    / source
                    / condition
                    / f"{stem}.jpg"
                )

                out_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # Existing 15-condition images will be skipped.
                # Only missing stacked conditions should be created.
                if not out_path.exists():

                    temp_path = out_path.with_suffix(
                        out_path.suffix + ".tmp"
                    )

                    make_image().save(
                        temp_path,
                        "JPEG",
                        quality=95,
                    )

                    temp_path.replace(
                        out_path
                    )

                manifest_rows.append(
                    {
                        "original_path": row["path"],
                        "transformed_path": str(
                            out_path
                        ),
                        "condition": condition,
                        "label": row["label"],
                        "source": source,
                        "generator": row[
                            "generator"
                        ],
                    }
                )

    EVAL_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EVAL_MANIFEST,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_path",
                "transformed_path",
                "condition",
                "label",
                "source",
                "generator",
            ],
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    print(
        f"\nWrote "
        f"{len(manifest_rows)} rows "
        f"to {EVAL_MANIFEST}"
    )


if __name__ == "__main__":
    main()