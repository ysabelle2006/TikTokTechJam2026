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

Also reports Accuracy/FPR/FNR at a fixed decision threshold
(DECISION_THRESHOLD, 0.5) alongside AUC per condition -- AUC alone
can't answer "how often does this wrongly flag a real image," which
the brief names explicitly as a trade-off worth discussing. 0.5 is a
fair choice whether or not the checkpoint has calibration.json applied
-- see DECISION_THRESHOLD's own comment for why temperature scaling
doesn't change which side of that threshold a prediction lands on.

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

Add --dump-predictions (v1/v2 only) to also write
results/<stage>_predictions.csv -- one row per eval_manifest.csv row
plus this run's raw prediction, which src/error_analysis.py reads to
pull specific misclassified examples for a written-up error analysis.
Off by default: most runs of this script only need the aggregated
results/<stage>.json for the roadmap table.
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
from models.fusion import FusionHead, load_architecture_metadata
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


DECISION_THRESHOLD = 0.5  # only used for the Accuracy/FPR/FNR columns below -- AUC (the primary,
# threshold-free metric) is unaffected by this choice. 0.5 is also the natural choice regardless
# of whether the scoring checkpoint has calibration.json applied: temperature scaling is monotonic
# in the logit (T > 0), so which side of 0.5 a prediction falls on is identical calibrated or not --
# only the reported CONFIDENCE differs, not the accuracy/FPR/FNR numbers below.


def _confusion_counts(labels: np.ndarray, preds: np.ndarray, threshold: float = DECISION_THRESHOLD):
    """labels: 0=real, 1=fake (see data/datasets.py). Returns
    (accuracy, fpr, fnr) at `threshold` -- fpr/fnr are None if there's
    no real (resp. fake) example in this slice to divide by, rather
    than raising or silently reporting 0/0 as 0."""
    predicted_fake = preds > threshold
    actually_fake = labels > 0.5
    accuracy = float(np.mean(predicted_fake == actually_fake))

    n_real = int(np.sum(~actually_fake))
    n_fake = int(np.sum(actually_fake))
    fp = int(np.sum(predicted_fake & ~actually_fake))
    fn = int(np.sum(~predicted_fake & actually_fake))
    fpr = (fp / n_real) if n_real > 0 else None
    fnr = (fn / n_fake) if n_fake > 0 else None
    return accuracy, fpr, fnr


def _summarize_and_write(stage: str, rows, preds: np.ndarray) -> dict:
    """Shared by evaluate_v0 and evaluate_v1 so the per-condition AUC /
    unseen-generator / final-score logic can't drift between stages --
    every stage's results/<stage>.json is computed exactly the same
    way, which is what makes the roadmap comparison in the architecture
    doc meaningful.

    Also reports Accuracy/FPR/FNR at DECISION_THRESHOLD alongside AUC
    per condition -- AUC alone (rank-based, threshold-free) can't
    answer "how often would this wrongly flag a real image as AI-
    generated," which the brief names explicitly as a trade-off to
    discuss. See DECISION_THRESHOLD's comment for why 0.5 is a fair
    choice regardless of whether calibration.json is in play."""
    labels = np.array([int(r["label"]) for r in rows], dtype=np.float32)
    conditions = np.array([r["condition"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    unseen_generator_mask_all = np.isin(sources, list(UNSEEN_GENERATOR_SOURCES))

    per_condition = {}
    per_condition_unseen_generator = {}
    for condition in sorted(set(conditions)):
        mask = conditions == condition
        accuracy, fpr, fnr = _confusion_counts(labels[mask], preds[mask])
        per_condition[condition] = {
            "auc": float(roc_auc_score(labels[mask], preds[mask])),
            "accuracy": accuracy,
            "fpr": fpr,
            "fnr": fnr,
            "n": int(mask.sum()),
        }
        unseen_mask = mask & unseen_generator_mask_all
        if len(set(labels[unseen_mask])) > 1:  # roc_auc_score needs both classes present
            unseen_accuracy, unseen_fpr, unseen_fnr = _confusion_counts(labels[unseen_mask], preds[unseen_mask])
            per_condition_unseen_generator[condition] = {
                "auc": float(roc_auc_score(labels[unseen_mask], preds[unseen_mask])),
                "accuracy": unseen_accuracy,
                "fpr": unseen_fpr,
                "fnr": unseen_fnr,
                "n": int(unseen_mask.sum()),
            }

    clean_auc = per_condition["clean"]["auc"]
    robust_conditions = [c for c in per_condition if c != "clean"]
    avg_robust_auc = float(np.mean([per_condition[c]["auc"] for c in robust_conditions]))
    final_score = 0.5 * clean_auc + 0.5 * avg_robust_auc

    all_fprs = [per_condition[c]["fpr"] for c in per_condition if per_condition[c]["fpr"] is not None]
    all_fnrs = [per_condition[c]["fnr"] for c in per_condition if per_condition[c]["fnr"] is not None]
    mean_fpr = float(np.mean(all_fprs)) if all_fprs else None
    mean_fnr = float(np.mean(all_fnrs)) if all_fnrs else None

    summary = {
        "stage": stage,
        "decision_threshold": DECISION_THRESHOLD,
        "per_condition": per_condition,
        "per_condition_unseen_generator": per_condition_unseen_generator,
        "clean_auc": clean_auc,
        "avg_robust_auc": avg_robust_auc,
        "avg_robust_drop": clean_auc - avg_robust_auc,
        "final_score": final_score,
        "mean_fpr": mean_fpr,
        "mean_fnr": mean_fnr,
    }

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(RESULTS_DIR) / f"{stage}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    def _pct(x):
        return "--" if x is None else f"{100 * x:.1f}%"

    print(f"\n{'condition':<16}{'Acc.':>7}{'AUC':>8}{'FPR':>8}{'FNR':>8}{'n':>7}   {'unseen-gen AUC':>16}")
    for condition in sorted(per_condition):
        r = per_condition[condition]
        u = per_condition_unseen_generator.get(condition)
        u_str = f"{u['auc']:.4f}" if u else "--"
        print(f"{condition:<16}{100 * r['accuracy']:>6.1f}%{r['auc']:>8.4f}{_pct(r['fpr']):>8}"
              f"{_pct(r['fnr']):>8}{r['n']:>7}   {u_str:>16}")

    print(f"\nclean AUC:        {clean_auc:.4f}")
    print(f"avg robust AUC:    {avg_robust_auc:.4f}")
    print(f"avg robust drop:   {summary['avg_robust_drop']:.4f}")
    print(f"final score:       {final_score:.4f}")
    print(f"mean FPR (@ {DECISION_THRESHOLD}):  {_pct(mean_fpr)}  -- how often a REAL image gets flagged as AI-generated, averaged across conditions")
    print(f"mean FNR (@ {DECISION_THRESHOLD}):  {_pct(mean_fnr)}  -- how often an AI-generated image slips through, averaged across conditions")
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
    dump_predictions: bool = False,
    result_name_override: str = None,
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
    dump_predictions: also write results/<stage>_predictions.csv, one
    row per eval_manifest.csv row plus this run's raw prediction --
    src/error_analysis.py reads this to pull specific misclassified
    examples rather than re-deriving predictions itself (one scoring
    loop, not two that could drift apart). Off by default since most
    callers (the roadmap comparison) only need the aggregated JSON.
    result_name_override: use this exact string in place of
    result_name_fmt.format(mode=resolved_mode) for both the JSON and
    the predictions CSV filename. Needed because --checkpoint can point
    at ANY model.pt regardless of its directory name, but the result
    filename otherwise depends only on checkpoint_dir_prefix/mode -- so
    without this, evaluating a second checkpoint of the same stage/mode
    (e.g. checkpoints/v2_augmented_fft_gated/model.pt, trained with
    FusionHead(use_freq_gate=True)) would silently overwrite
    results/v2_augmented_fft.json, the already-verified baseline every
    number in this project's README traces back to.
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

    use_freq_gate = load_architecture_metadata(checkpoint_path.parent)
    fusion = FusionHead(use_freq_gate=use_freq_gate)
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

    resolved_name = result_name_override or result_name_fmt.format(mode=resolved_mode)

    if dump_predictions:
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        predictions_path = Path(RESULTS_DIR) / f"{resolved_name}_predictions.csv"
        fieldnames = ["original_path", "transformed_path", "condition", "label", "source", "generator", "pred"]
        with open(predictions_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, pred in zip(rows, preds):
                writer.writerow({**row, "pred": float(pred)})
        print(f"Wrote per-row predictions to {predictions_path}")

    return _summarize_and_write(resolved_name, rows, preds)


def evaluate_v1(checkpoint_path: str = None, batch_size: int = 64, freq_mode: str = None,
                 dump_predictions: bool = False, result_name_override: str = None):
    return _evaluate_fusion_stage(
        checkpoint_dir_prefix="v1_fusion",
        result_name_fmt="v1_fusion_{mode}",
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        freq_mode=freq_mode,
        stage_label="v1",
        dump_predictions=dump_predictions,
        result_name_override=result_name_override,
    )


def evaluate_v2(checkpoint_path: str = None, batch_size: int = 64, freq_mode: str = None,
                 dump_predictions: bool = False, result_name_override: str = None):
    """V2 checkpoints (checkpoints/v2_augmented_<mode>/model.pt) were
    selected by train_v2's best-checkpoint tracker against a validation
    set that includes BOTH clean and augmented rows (see train.py's
    module docstring on the V1 checkpoint-selection blind spot) -- but
    the scoring here is otherwise identical to evaluate_v1: same frozen
    spatial stream, same eval_manifest.csv robustness grid, same
    per-condition AUC table. That's deliberate -- it's what makes
    results/v2_augmented_<mode>.json directly comparable to
    results/v1_fusion_<mode>.json in the architecture doc's roadmap
    table, rather than measuring something subtly different.

    result_name_override: see _evaluate_fusion_stage's docstring --
    pass this (e.g. "v2_augmented_fft_gated") whenever --checkpoint
    points somewhere other than the default checkpoints/v2_augmented_
    <mode>/model.pt, so this run's results don't overwrite that
    checkpoint's already-written results/v2_augmented_<mode>.json."""
    return _evaluate_fusion_stage(
        checkpoint_dir_prefix="v2_augmented",
        result_name_fmt="v2_augmented_{mode}",
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        freq_mode=freq_mode,
        stage_label="v2",
        dump_predictions=dump_predictions,
        result_name_override=result_name_override,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="v0", choices=["v0", "v1", "v2"])
    parser.add_argument("--checkpoint", default=None, help="override the default checkpoint path")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"],
                         help="v1/v2 only: which checkpoint/mode to evaluate (defaults to the checkpoint's own recorded mode)")
    parser.add_argument("--dump-predictions", action="store_true",
                         help="v1/v2 only: also write results/<stage>_predictions.csv "
                              "(one row per eval_manifest.csv row plus this run's raw prediction) "
                              "-- src/error_analysis.py reads this")
    parser.add_argument("--result-name", default=None,
                         help="v1/v2 only: write results/<this>.json (and _predictions.csv, if requested) "
                              "under this name instead of the default v{1,2}_..._<mode> -- required when "
                              "--checkpoint points at a non-default checkpoint (e.g. a gated retrain) sharing "
                              "the same stage/mode as an existing results file you don't want overwritten")
    args = parser.parse_args()
    if args.stage == "v0":
        evaluate_v0(checkpoint_path=args.checkpoint)
    elif args.stage == "v1":
        evaluate_v1(checkpoint_path=args.checkpoint, freq_mode=args.freq_mode,
                    dump_predictions=args.dump_predictions, result_name_override=args.result_name)
    elif args.stage == "v2":
        evaluate_v2(checkpoint_path=args.checkpoint, freq_mode=args.freq_mode,
                    dump_predictions=args.dump_predictions, result_name_override=args.result_name)


if __name__ == "__main__":
    main()
