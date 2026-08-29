"""
Robustness evaluation harness.

Reads directly from data/eval_manifest.csv rather than re-deriving its
own sample of validation_demo images -- that file (built by
scripts/build_eval_grid.py) already has every image needed: 300 images
per validation source x all 15 conditions from the brief's transform
grid (clean + 4 JPEG qualities + 3 blur sigmas + 2 resize scales + 3
noise sigmas + color jitter + center crop). Reusing it means each
stage's evaluation is already the full robustness table, not just a
clean-data sanity check, and it can never drift out of sync with what
the eval grid actually contains.

Two AUC views are reported per condition:
  - "per_condition": pooled across ALL validation sources. This is the
    main robustness-table row.
  - "per_condition_unseen_generator": computed using ONLY
    coco_val2017 (real) + wildfake_dalle (fake, generator="dalle")
    rows. wildfake_dalle is the one generator family never seen in
    training (see data/datasets.py), so this is the actual held-out-
    generator generalization check from the architecture doc.
    sid_set is deliberately excluded from this view -- its
    "sid_set_mixed" generator label appears in BOTH the train and
    validation_demo splits, so scoring on it answers "held-out
    samples", not "held-out generator family", and mixing the two
    would misrepresent which question is being answered.

Primary metric is ROC AUC (threshold-free, robust to class imbalance),
matching the brief. Final Score = 0.50*AUC_clean + 0.50*AUC_robust,
where AUC_robust is the mean AUC across the 14 non-clean conditions.

Writes results/<stage>.json -- one file per roadmap stage, never
overwritten (see results/README.md):
    v0_spatial_only.json      (V0 -- done)
    v1_fusion_srm.json        (V1, frequency stream in "srm" mode)
    v1_fusion_fft.json        (V1, frequency stream in "fft" mode -- the
                               srm-vs-fft ablation, run with --freq-mode fft)
    v2_augmented_<mode>.json  (V2, transform-aware training -- checkpoint
                               dir is v2_augmented_<mode>, same idea as V1)

evaluate_v1 and evaluate_v2 share nearly all of their loading/scoring
logic (frozen spatial stream + trainable frequency stream + fusion head,
scored batch-by-batch over the SAME eval_manifest.csv robustness grid) --
the only real difference is which checkpoint-dir prefix to look under
and which results/ filename to write. That shared logic lives in
_evaluate_fusion_stage(); evaluate_v1/evaluate_v2 are thin wrappers so
each stage's CLI-facing name and docstring stay explicit and searchable,
without duplicating (and risking drift in) the actual scoring loop.

Run with:  uv run python src/train.py --stage v0     (first, to get a checkpoint)
           uv run python src/evaluate.py --stage v0
           uv run python src/train.py --stage v1
           uv run python src/evaluate.py --stage v1
           uv run python src/train.py --stage v2
           uv run python src/evaluate.py --stage v2
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import CHECKPOINT_DIR, FREQUENCY_MODE, RESULTS_DIR
from models.frequency_stream import FrequencyStream
from models.fusion import FusionHead
from models.spatial_stream import SpatialStream
from train import V0Head  # import the exact class that was trained, rather than redefine it here and risk drift
from transforms.preprocessing import prepare_frequency_input, residual_energy

EVAL_MANIFEST = Path("data/eval_manifest.csv")

# wildfake_dalle's "dalle" generator never appears in the train split
# (see data/datasets.py) -- that's what makes pairing it with
# coco_val2017's real images a genuine held-out-generator test, unlike
# sid_set (whose generator label does appear in train).
UNSEEN_GENERATOR_SOURCES = {"coco_val2017", "wildfake_dalle"}


def load_eval_manifest():
    if not EVAL_MANIFEST.is_file():
        raise FileNotFoundError(f"{EVAL_MANIFEST} doesn't exist -- run `python scripts/build_eval_grid.py` first.")
    with open(EVAL_MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def _summarize_and_write(stage: str, rows, preds: np.ndarray) -> dict:
    """Shared by evaluate_v0 and evaluate_v1 so the per-condition AUC /
    unseen-generator / final-score logic can't drift between stages --
    every stage's results/<stage>.json is computed exactly the same
    way, which is what makes the roadmap comparison in the architecture
    doc meaningful."""
    labels = np.array([int(r["label"]) for r in rows], dtype=np.float32)
    conditions = np.array([r["condition"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    unseen_generator_mask_all = np.isin(sources, list(UNSEEN_GENERATOR_SOURCES))

    per_condition = {}
    per_condition_unseen_generator = {}
    for condition in sorted(set(conditions)):
        mask = conditions == condition
        per_condition[condition] = {
            "auc": float(roc_auc_score(labels[mask], preds[mask])),
            "n": int(mask.sum()),
        }
        unseen_mask = mask & unseen_generator_mask_all
        if len(set(labels[unseen_mask])) > 1:  # roc_auc_score needs both classes present
            per_condition_unseen_generator[condition] = {
                "auc": float(roc_auc_score(labels[unseen_mask], preds[unseen_mask])),
                "n": int(unseen_mask.sum()),
            }

    clean_auc = per_condition["clean"]["auc"]
    robust_conditions = [c for c in per_condition if c != "clean"]
    avg_robust_auc = float(np.mean([per_condition[c]["auc"] for c in robust_conditions]))
    final_score = 0.5 * clean_auc + 0.5 * avg_robust_auc

    summary = {
        "stage": stage,
        "per_condition": per_condition,
        "per_condition_unseen_generator": per_condition_unseen_generator,
        "clean_auc": clean_auc,
        "avg_robust_auc": avg_robust_auc,
        "avg_robust_drop": clean_auc - avg_robust_auc,
        "final_score": final_score,
    }

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(RESULTS_DIR) / f"{stage}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'condition':<16}{'AUC':>8}{'n':>8}   {'unseen-gen AUC':>16}")
    for condition in sorted(per_condition):
        r = per_condition[condition]
        u = per_condition_unseen_generator.get(condition)
        u_str = f"{u['auc']:.4f}" if u else "--"
        print(f"{condition:<16}{r['auc']:>8.4f}{r['n']:>8}   {u_str:>16}")

    print(f"\nclean AUC:        {clean_auc:.4f}")
    print(f"avg robust AUC:    {avg_robust_auc:.4f}")
    print(f"avg robust drop:   {summary['avg_robust_drop']:.4f}")
    print(f"final score:       {final_score:.4f}")
    print(f"\nWrote {out_path}")
    return summary


def evaluate_v0(checkpoint_path: str = None, batch_size: int = 64):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path(CHECKPOINT_DIR) / "v0_spatial_only" / "head.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{checkpoint_path} doesn't exist -- run `uv run python src/train.py` first.")

    print("Loading CLIP backbone...")
    stream = SpatialStream()
    head = V0Head()
    head.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    head.eval()

    rows = load_eval_manifest()
    print(f"scoring {len(rows)} images from {EVAL_MANIFEST} ...")

    preds = np.empty(len(rows), dtype=np.float32)
    batch_tensors, batch_indices = [], []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors)
        with torch.no_grad():
            logits = head(stream.encode(batch))
            probs = torch.sigmoid(logits).numpy()
        for local_i, global_i in enumerate(batch_indices):
            preds[global_i] = probs[local_i]
        batch_tensors.clear()
        batch_indices.clear()

    for i, r in enumerate(tqdm(rows, desc="evaluating")):
        img = Image.open(r["transformed_path"]).convert("RGB")
        batch_tensors.append(stream.prepare(img))
        batch_indices.append(i)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    return _summarize_and_write("v0_spatial_only", rows, preds)


def _evaluate_fusion_stage(
    checkpoint_dir_prefix: str,
    result_name_fmt: str,
    checkpoint_path: str = None,
    batch_size: int = 64,
    freq_mode: str = None,
    stage_label: str = None,
):
    """Shared by evaluate_v1 and evaluate_v2 -- loads a frozen spatial
    stream + a trained frequency stream/fusion head pair, scores every
    row of the same eval_manifest.csv robustness grid, and writes
    results/<result_name_fmt.format(mode=resolved_mode)>.json via
    _summarize_and_write. See module docstring for why this is one
    function instead of two near-duplicate ones.

    checkpoint_dir_prefix: e.g. "v1_fusion" or "v2_augmented" -- the
    checkpoint is expected at checkpoints/<prefix>_<mode>/model.pt
    unless checkpoint_path overrides it directly.
    stage_label: only used in the "run this first" error message, so it
    points at the right train.py --stage.
    """
    if checkpoint_path is None:
        mode_for_path = freq_mode or FREQUENCY_MODE
        checkpoint_path = Path(CHECKPOINT_DIR) / f"{checkpoint_dir_prefix}_{mode_for_path}" / "model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_path} doesn't exist -- run `uv run python src/train.py --stage {stage_label}` first "
            f"(add --freq-mode fft if you're evaluating the fft ablation)."
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # freq_mode argument wins if given; otherwise trust what the checkpoint
    # recorded it was trained with (falls back to config's default only for
    # checkpoints saved before "freq_mode" was added to the saved dict).
    resolved_mode = freq_mode or ckpt.get("freq_mode") or FREQUENCY_MODE

    print("Loading CLIP backbone...")
    spatial_stream = SpatialStream()

    print(f"Loading frequency stream (mode={resolved_mode}) + fusion head...")
    if "best_epoch" in ckpt:
        print(f"checkpoint is from epoch {ckpt['best_epoch']}/{ckpt.get('trained_epochs', '?')} "
              f"(best val_loss={ckpt.get('best_val_loss', float('nan')):.4f}), not necessarily the last epoch trained")
    freq_stream = FrequencyStream(freeze=True, mode=resolved_mode)
    freq_stream.model.load_state_dict(ckpt["frequency_cnn"])

    fusion = FusionHead()
    fusion.load_state_dict(ckpt["fusion_head"])
    fusion.eval()

    rows = load_eval_manifest()
    print(f"scoring {len(rows)} images from {EVAL_MANIFEST} ...")

    preds = np.empty(len(rows), dtype=np.float32)
    spatial_tensors, freq_tensors, energies, batch_indices = [], [], [], []

    def flush():
        if not spatial_tensors:
            return
        spatial_batch = torch.stack(spatial_tensors)
        freq_batch = torch.stack(freq_tensors)
        energy_batch = torch.tensor(energies, dtype=torch.float32)
        with torch.no_grad():
            spatial_emb = spatial_stream.encode(spatial_batch)
            freq_emb = freq_stream.encode(freq_batch)
            logits = fusion(spatial_emb, freq_emb, energy_batch)
            probs = torch.sigmoid(logits).numpy()
        for local_i, global_i in enumerate(batch_indices):
            preds[global_i] = probs[local_i]
        spatial_tensors.clear()
        freq_tensors.clear()
        energies.clear()
        batch_indices.clear()

    for i, r in enumerate(tqdm(rows, desc="evaluating")):
        img = Image.open(r["transformed_path"]).convert("RGB")
        spatial_tensors.append(spatial_stream.prepare(img))
        freq_map = prepare_frequency_input(img, mode=resolved_mode)
        freq_tensors.append(torch.from_numpy(freq_map).unsqueeze(0))
        energies.append(residual_energy(freq_map))
        batch_indices.append(i)
        if len(spatial_tensors) >= batch_size:
            flush()
    flush()

    return _summarize_and_write(result_name_fmt.format(mode=resolved_mode), rows, preds)


def evaluate_v1(checkpoint_path: str = None, batch_size: int = 64, freq_mode: str = None):
    return _evaluate_fusion_stage(
        checkpoint_dir_prefix="v1_fusion",
        result_name_fmt="v1_fusion_{mode}",
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        freq_mode=freq_mode,
        stage_label="v1",
    )


def evaluate_v2(checkpoint_path: str = None, batch_size: int = 64, freq_mode: str = None):
    """V2 checkpoints (checkpoints/v2_augmented_<mode>/model.pt) were
    selected by train_v2's best-checkpoint tracker against a validation
    set that includes BOTH clean and augmented rows (see train.py's
    module docstring on the V1 checkpoint-selection blind spot) -- but
    the scoring here is otherwise identical to evaluate_v1: same frozen
    spatial stream, same eval_manifest.csv robustness grid, same
    per-condition AUC table. That's deliberate -- it's what makes
    results/v2_augmented_<mode>.json directly comparable to
    results/v1_fusion_<mode>.json in the architecture doc's roadmap
    table, rather than measuring something subtly different."""
    return _evaluate_fusion_stage(
        checkpoint_dir_prefix="v2_augmented",
        result_name_fmt="v2_augmented_{mode}",
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        freq_mode=freq_mode,
        stage_label="v2",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="v0", choices=["v0", "v1", "v2"])
    parser.add_argument("--checkpoint", default=None, help="override the default checkpoint path")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"],
                         help="v1/v2 only: which checkpoint/mode to evaluate (defaults to the checkpoint's own recorded mode)")
    args = parser.parse_args()
    if args.stage == "v0":
        evaluate_v0(checkpoint_path=args.checkpoint)
    elif args.stage == "v1":
        evaluate_v1(checkpoint_path=args.checkpoint, freq_mode=args.freq_mode)
    elif args.stage == "v2":
        evaluate_v2(checkpoint_path=args.checkpoint, freq_mode=args.freq_mode)


if __name__ == "__main__":
    main()
