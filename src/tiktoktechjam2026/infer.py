"""
Deliverable script: run the detector over an image directory and write a
JSON list of {"image_path": ..., "pred": ...} records.

`pred` is P(image is AI-generated) in [0, 1] from Detector.predict -- the
same code path training and evaluation use.

CLI:
    python -m tiktoktechjam2026.infer --images path/to/dir --out preds.json
    python -m tiktoktechjam2026.infer --images dir --variant v1 --out preds.json
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image

from tiktoktechjam2026 import config
from tiktoktechjam2026.models.detector import Detector

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(image_dir: str):
    for root, _, files in os.walk(image_dir):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in _IMG_EXTS:
                yield os.path.join(root, name)


def main(image_dir: str, output_json: str, variant: str = "v1",
         checkpoint: str = None, batch_size: int = None):
    batch_size = batch_size or config.BATCH_SIZE
    checkpoint = checkpoint or config.CHECKPOINTS[variant]
    detector = Detector.from_checkpoint(checkpoint)
    detector.eval()

    paths = list(_iter_images(image_dir))
    if not paths:
        raise SystemExit(f"no images found under {image_dir!r}")

    records = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        images = []
        for p in chunk:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        probs = detector.predict_proba(images)
        for p, prob in zip(chunk, probs):
            records.append({"image_path": p, "pred": float(prob)})
        print(f"\r  {min(start + batch_size, len(paths))}/{len(paths)}", end="", flush=True)
    print()

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    print(f"wrote {output_json}  ({len(records)} images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Directory -> {image_path, pred} JSON.")
    ap.add_argument("--images", required=True, help="image directory (recursed)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--variant", choices=["v0", "v1"], default="v1")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    main(args.images, args.out, args.variant, args.checkpoint, args.batch_size)
