"""
Deliverable inference script: image(s) in, JSON predictions out. This
is the "one function, image path to a float" entry point the
architecture doc's §03 note describes, wired to the actual Detector
class (models/detector.py) -- same frozen CLIP spatial stream +
trained frequency stream + fusion head used everywhere else in this
project, now wrapped for the "score a folder of images" deliverable
case rather than train.py/evaluate.py's batch-training/batch-eval use.

Checkpoint resolution follows evaluate.py's _evaluate_fusion_stage
convention exactly (checkpoints/<v1_fusion|v2_augmented>_<mode>/model.pt),
so the same --stage/--checkpoint/--freq-mode flags mean the same thing
here as in evaluate.py and calibrate.py -- no separate convention to
remember for the deliverable script specifically.

Calibration: Detector.load_fusion_checkpoint() auto-loads
calibration.json next to the checkpoint if calibrate.py has been run
for that stage/mode; pass --no-calibration to force the raw,
uncalibrated sigmoid instead (e.g. to compare against a report that
was written before calibration existed). If no calibration.json exists
yet, this runs fine anyway -- Detector just prints a note and returns
the uncalibrated score, same as before calibrate.py existed.

Also works unchanged against a future V3 checkpoint: V3 only changes
training (adds a consistency loss term computed from two forward
passes of the SAME shared-weight Detector), not the architecture
Detector wires together, so a V3 run only needs --checkpoint pointed
at its checkpoint file -- nothing here would need to change. See the
architecture doc's "if we stop at V2, does V3 wire back in cleanly"
discussion.

Input can be a single image file or a directory (searched
non-recursively for .jpg/.jpeg/.png/.bmp/.webp, case-insensitive).
Images that fail to open (corrupt file, unsupported format) are
skipped with a warning printed to stderr and a "error" field in their
JSON record, rather than aborting the whole run over one bad file --
appropriate for a deliverable script scoring an arbitrary folder of
images nobody has necessarily vetted, unlike train.py/evaluate.py's
manifests, which are built from images this project already knows are
readable.

Output JSON is a list of records:
    {"image_path": "...", "pred": 0.9421, "label_guess": "fake"}
or, for a file that failed to load:
    {"image_path": "...", "pred": null, "error": "..."}
label_guess is just a >0.5 threshold on pred for convenience in a
quick read-through of the output -- the actual number (pred) is what
matters, no threshold decision is baked into scoring elsewhere in this
project either.

Run with:  uv run python src/infer.py --input path/to/image_or_dir
           uv run python src/infer.py --input path/to/dir --stage v2 --output outputs/predictions.json
           uv run python src/infer.py --input path/to/dir --no-calibration
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import CHECKPOINT_DIR, FREQUENCY_MODE
from models.detector import Detector

# Same checkpoint-dir-prefix convention as evaluate.py/calibrate.py --
# kept as a small local dict rather than imported, since importing
# evaluate.py just for this would pull in its argparse/main() setup for
# no reason (same reasoning calibrate.py used).
CHECKPOINT_DIR_PREFIX = {"v1": "v1_fusion", "v2": "v2_augmented"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_OUTPUT = Path("outputs/predictions.json")


def _resolve_checkpoint_path(stage: str, checkpoint_path: str = None, freq_mode: str = None) -> Path:
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    else:
        mode_for_path = freq_mode or FREQUENCY_MODE
        checkpoint_path = Path(CHECKPOINT_DIR) / f"{CHECKPOINT_DIR_PREFIX[stage]}_{mode_for_path}" / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_path} doesn't exist -- run `uv run python src/train.py --stage {stage}` first "
            f"(add --freq-mode fft if you're pointing at the fft ablation)."
        )
    return checkpoint_path


def _iter_input_images(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"{input_path} doesn't exist")


def run_inference(
    input_path: str,
    stage: str = "v2",
    checkpoint_path: str = None,
    freq_mode: str = None,
    calibrate: bool = True,
    output_path: str = None,
):
    checkpoint_path = _resolve_checkpoint_path(stage, checkpoint_path, freq_mode)
    images = _iter_input_images(Path(input_path))
    if not images:
        print(f"No images found under {input_path} (looked for {sorted(IMAGE_EXTENSIONS)})", file=sys.stderr)
        return []
    print(f"found {len(images)} image(s) under {input_path}")

    # Peek at the raw checkpoint to resolve which frequency mode it was
    # actually trained with -- same reasoning as evaluate.py/calibrate.py:
    # --freq-mode wins if given, otherwise trust the checkpoint's own
    # recorded mode rather than guessing from config's current default,
    # which may have changed since that checkpoint was trained.
    import torch
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    resolved_mode = freq_mode or raw_ckpt.get("freq_mode") or FREQUENCY_MODE
    del raw_ckpt  # Detector.load_fusion_checkpoint() below re-reads it properly; this was just to peek at freq_mode.

    print("Loading CLIP backbone...")
    detector = Detector(device="cpu", freq_mode=resolved_mode)
    print(f"Loading frequency stream (mode={resolved_mode}) + fusion head from {checkpoint_path}...")
    detector.load_fusion_checkpoint(checkpoint_path, calibration_path=(None if calibrate else False))

    records = []
    for path in tqdm(images, desc="scoring"):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any bad-file reason should be skipped, not crash the run
            print(f"WARNING: failed to open {path}: {e}", file=sys.stderr)
            records.append({"image_path": str(path), "pred": None, "error": str(e)})
            continue
        pred = detector.predict(image)
        records.append({
            "image_path": str(path),
            "pred": pred,
            "label_guess": "fake" if pred > 0.5 else "real",
        })

    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    n_scored = sum(1 for r in records if r["pred"] is not None)
    n_failed = len(records) - n_scored
    print(f"\nscored {n_scored} image(s)" + (f", {n_failed} failed to load" if n_failed else ""))
    print(f"Wrote {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="path to a single image, or a directory of images")
    parser.add_argument("--stage", default="v2", choices=["v1", "v2"])
    parser.add_argument("--checkpoint", default=None, help="override the default checkpoint path")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"],
                         help="defaults to the checkpoint's own recorded mode")
    parser.add_argument("--no-calibration", action="store_true",
                         help="skip loading calibration.json even if present -- report the raw uncalibrated sigmoid")
    parser.add_argument("--output", default=None, help=f"where to write the JSON predictions (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    run_inference(
        input_path=args.input,
        stage=args.stage,
        checkpoint_path=args.checkpoint,
        freq_mode=args.freq_mode,
        calibrate=not args.no_calibration,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
