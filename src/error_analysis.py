"""
Error analysis for a fusion stage (V2 by default): pulls specific
misclassified examples, checks whether the SPATIAL stream alone (V0's
checkpoint) would have gotten each one right, and reports each
example's residual-energy value (how much high-frequency signal
survived for the frequency stream to work with) -- distinguishing two
different kinds of mistake that a single AUC number can't tell apart:

  - "shared blind spot": V0 (spatial-only) was ALSO wrong on this
    image. The frequency stream and fusion didn't introduce this
    mistake; it's a genuine hard example for the spatial signal alone,
    and V2's fusion had nothing better to fall back on.
  - "fusion-induced regression": V0 (spatial-only) was RIGHT, but V2's
    fused prediction is wrong. Something about how the frequency
    stream's output combined with the spatial embedding pulled a
    correct answer into the wrong side of 0.5. residual_energy is the
    first thing worth checking here -- an unusually low or high value
    relative to the baseline sample suggests the frequency stream's
    input was itself unreliable for that image and the fusion head
    trusted it anyway.

Requires a predictions dump from evaluate.py: run
    uv run python src/evaluate.py --stage v2 --dump-predictions
first (add --freq-mode fft/srm to match the checkpoint being
analyzed), which writes results/v2_augmented_<mode>_predictions.csv --
this script reads that rather than re-scoring the eval grid itself, so
V2's predictions here are guaranteed to be the exact same ones the
robustness table was computed from, not a second, potentially-drifted
scoring pass.

For each requested condition (--conditions, default: clean plus the
two conditions V2's own results showed the largest robust-AUC drop on
-- blur_sigma2.0 and resize_0.25, per results/v2_augmented_fft.json;
override this if analyzing a different checkpoint/mode where the worst
conditions differ):

  1. Take the --n-examples most CONFIDENTLY wrong V2 predictions
     (largest |pred - label|, so "predicted 0.95 fake, actually real"
     ranks above "predicted 0.55 fake, actually real").
  2. Sample --n-baseline correctly-classified rows from the same
     condition, for comparison -- otherwise "these misclassified
     examples have residual_energy=0.08" means nothing without knowing
     what a TYPICAL image's residual_energy looks like in the same
     condition.
  3. Score every featured example (examples + baseline) through V0's
     checkpoint (spatial stream + V0Head, see train.py's V0Head) and
     compute residual_energy (transforms.preprocessing, pure numpy --
     same function the frequency stream itself uses, see that module).

This intentionally does NOT re-run the full V0/V2 stream stack for
every eval-grid row (tens of thousands of images) -- only for the
handful of featured examples per condition, so this stays a quick,
targeted diagnostic rather than another full evaluation pass.

Writes:
    results/<stage>_error_analysis.csv   one row per featured example
    results/<stage>_error_analysis.md    the write-up: per-condition
                                          counts (shared blind spot vs.
                                          fusion-induced regression)
                                          and residual_energy comparison

Run with:  uv run python src/evaluate.py --stage v2 --dump-predictions
           uv run python src/error_analysis.py
           uv run python src/error_analysis.py --conditions clean jpeg_q30 --n-examples 15
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image

from config import CHECKPOINT_DIR, FREQUENCY_MODE, RESULTS_DIR
from models.spatial_stream import SpatialStream
from train import V0Head  # the exact class V0 was trained with -- see evaluate.py for the same reasoning
from transforms.preprocessing import prepare_frequency_input, residual_energy

CHECKPOINT_DIR_PREFIX = {"v1": "v1_fusion", "v2": "v2_augmented"}
DEFAULT_CONDITIONS = ["clean", "blur_sigma2.0", "resize_0.25"]


def _load_predictions(predictions_csv: Path):
    if not predictions_csv.is_file():
        raise FileNotFoundError(
            f"{predictions_csv} doesn't exist -- run `uv run python src/evaluate.py --stage <stage> "
            f"--dump-predictions` first (matching --freq-mode if you're not using the default)."
        )
    with open(predictions_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["pred"] = float(r["pred"])
    return rows


def _resolve_freq_mode(stage: str, freq_mode: str = None) -> str:
    if freq_mode:
        return freq_mode
    mode_for_path = FREQUENCY_MODE
    checkpoint_path = Path(CHECKPOINT_DIR) / f"{CHECKPOINT_DIR_PREFIX[stage]}_{mode_for_path}" / "model.pt"
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return ckpt.get("freq_mode") or FREQUENCY_MODE
    return FREQUENCY_MODE


def _wrongness(row: dict) -> float:
    """How confidently wrong a prediction is: distance from the correct
    endpoint of [0, 1] -- 1-pred if the true label is fake (1), pred if
    the true label is real (0). Larger = more confidently wrong."""
    return (1.0 - row["pred"]) if row["label"] == 1 else row["pred"]


def _is_correct(row: dict) -> bool:
    return (row["pred"] > 0.5) == (row["label"] == 1)


def select_examples(rows: list, condition: str, n_examples: int, n_baseline: int, seed: int):
    """Returns (featured_wrong, featured_baseline) for one condition:
    the n_examples most confidently-wrong V2 predictions, and a random
    sample of up to n_baseline correctly-classified rows from the same
    condition, for comparison."""
    condition_rows = [r for r in rows if r["condition"] == condition]
    wrong = [r for r in condition_rows if not _is_correct(r)]
    right = [r for r in condition_rows if _is_correct(r)]

    wrong_sorted = sorted(wrong, key=_wrongness, reverse=True)
    featured_wrong = wrong_sorted[:n_examples]

    rng = random.Random(seed)
    baseline_pool = right[:]
    rng.shuffle(baseline_pool)
    featured_baseline = baseline_pool[:n_baseline]

    return condition_rows, wrong, featured_wrong, featured_baseline


def _score_v0_and_energy(row: dict, spatial_stream: SpatialStream, v0_head: V0Head, freq_mode: str):
    image = Image.open(row["transformed_path"]).convert("RGB")
    with torch.no_grad():
        embedding = spatial_stream.encode(spatial_stream.prepare(image))
        v0_pred = torch.sigmoid(v0_head(embedding)).item()
    freq_map = prepare_frequency_input(image, mode=freq_mode)
    energy = residual_energy(freq_map)
    return v0_pred, energy


def run_error_analysis(
    stage: str = "v2",
    predictions_csv: str = None,
    v0_checkpoint: str = None,
    freq_mode: str = None,
    conditions=None,
    n_examples: int = 10,
    n_baseline: int = 30,
    seed: int = 0,
    output_csv: str = None,
    output_md: str = None,
):
    conditions = conditions or DEFAULT_CONDITIONS

    if predictions_csv is None:
        mode_for_path = freq_mode or FREQUENCY_MODE
        prefix = CHECKPOINT_DIR_PREFIX[stage]
        predictions_csv = Path(RESULTS_DIR) / f"{prefix}_{mode_for_path}_predictions.csv"
    else:
        predictions_csv = Path(predictions_csv)
    rows = _load_predictions(predictions_csv)
    print(f"loaded {len(rows)} predictions from {predictions_csv}")

    resolved_freq_mode = _resolve_freq_mode(stage, freq_mode)
    print(f"scoring V0 comparisons using residual_energy in mode={resolved_freq_mode}")

    v0_checkpoint = Path(v0_checkpoint) if v0_checkpoint else Path(CHECKPOINT_DIR) / "v0_spatial_only" / "head.pt"
    if not v0_checkpoint.is_file():
        raise FileNotFoundError(f"{v0_checkpoint} doesn't exist -- run `uv run python src/train.py --stage v0` first.")

    print("Loading CLIP backbone (for the V0 spatial-only comparison)...")
    spatial_stream = SpatialStream()
    v0_head = V0Head()
    v0_head.load_state_dict(torch.load(v0_checkpoint, map_location="cpu"))
    v0_head.eval()

    all_records = []
    condition_summaries = []

    for condition in conditions:
        condition_rows, wrong, featured_wrong, featured_baseline = select_examples(
            rows, condition, n_examples, n_baseline, seed
        )
        if not condition_rows:
            print(f"WARNING: no rows found for condition {condition!r} in {predictions_csv} -- skipping")
            continue
        print(f"\n{condition}: {len(condition_rows)} total rows, {len(wrong)} misclassified, "
              f"featuring {len(featured_wrong)} wrong + {len(featured_baseline)} correct-baseline")

        shared_blind_spot, fusion_regression = [], []
        baseline_energies = []

        for row in featured_baseline:
            v0_pred, energy = _score_v0_and_energy(row, spatial_stream, v0_head, resolved_freq_mode)
            baseline_energies.append(energy)
            all_records.append({
                "condition": condition, "transformed_path": row["transformed_path"], "label": row["label"],
                "v2_pred": row["pred"], "v0_pred": v0_pred, "residual_energy": energy,
                "group": "baseline_correct", "diagnosis": "correctly classified (baseline for comparison)",
            })

        for row in featured_wrong:
            v0_pred, energy = _score_v0_and_energy(row, spatial_stream, v0_head, resolved_freq_mode)
            v0_correct = (v0_pred > 0.5) == (row["label"] == 1)
            if v0_correct:
                fusion_regression.append(energy)
                diagnosis = ("spatial-only alone would have been correct here -- fusion/frequency "
                             "stream dragged this one to the wrong side of 0.5")
                group = "fusion_induced_regression"
            else:
                shared_blind_spot.append(energy)
                diagnosis = ("spatial-only alone was ALSO wrong -- a shared blind spot, not something "
                             "fusion introduced")
                group = "shared_blind_spot"
            all_records.append({
                "condition": condition, "transformed_path": row["transformed_path"], "label": row["label"],
                "v2_pred": row["pred"], "v0_pred": v0_pred, "residual_energy": energy,
                "group": group, "diagnosis": diagnosis,
            })

        def _mean(xs):
            return sum(xs) / len(xs) if xs else None

        condition_summaries.append({
            "condition": condition,
            "n_total": len(condition_rows),
            "n_wrong_total": len(wrong),
            "n_featured_wrong": len(featured_wrong),
            "n_shared_blind_spot": len(shared_blind_spot),
            "n_fusion_induced_regression": len(fusion_regression),
            "mean_energy_baseline_correct": _mean(baseline_energies),
            "mean_energy_shared_blind_spot": _mean(shared_blind_spot),
            "mean_energy_fusion_induced_regression": _mean(fusion_regression),
        })

    output_csv = Path(output_csv) if output_csv else Path(RESULTS_DIR) / f"{stage}_error_analysis.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["condition", "transformed_path", "label", "v2_pred", "v0_pred",
                      "residual_energy", "group", "diagnosis"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nWrote {len(all_records)} featured examples to {output_csv}")

    output_md = Path(output_md) if output_md else Path(RESULTS_DIR) / f"{stage}_error_analysis.md"
    _write_markdown(output_md, stage, predictions_csv, resolved_freq_mode, condition_summaries)
    print(f"Wrote {output_md}")

    return condition_summaries


def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def _write_markdown(output_md: Path, stage: str, predictions_csv: Path, freq_mode: str, summaries: list):
    lines = [
        f"# Error analysis -- {stage} ({freq_mode})",
        "",
        f"Source: {predictions_csv}",
        "",
        "For each condition below, the featured misclassified examples are split into two "
        "groups by re-scoring them through V0's spatial-only checkpoint: **shared blind spot** "
        "(V0 was also wrong -- the frequency stream and fusion head had nothing better to fall "
        "back on) vs. **fusion-induced regression** (V0 was right, but the fused prediction is "
        "wrong -- worth checking whether residual_energy looks unusual for these). "
        "mean_energy_baseline_correct is the average residual_energy over a random sample of "
        "correctly-classified rows from the same condition, for comparison.",
        "",
    ]
    for s in summaries:
        lines.append(f"## {s['condition']}")
        lines.append("")
        lines.append(
            f"{s['n_wrong_total']} of {s['n_total']} rows misclassified overall; "
            f"{s['n_featured_wrong']} featured here."
        )
        lines.append("")
        lines.append(
            f"Of the featured misclassified examples: {s['n_shared_blind_spot']} are shared blind "
            f"spots (V0 also wrong), {s['n_fusion_induced_regression']} are fusion-induced "
            f"regressions (V0 was right)."
        )
        lines.append("")
        lines.append(
            f"Mean residual_energy -- baseline (correct): {_fmt(s['mean_energy_baseline_correct'])}, "
            f"shared blind spot: {_fmt(s['mean_energy_shared_blind_spot'])}, "
            f"fusion-induced regression: {_fmt(s['mean_energy_fusion_induced_regression'])}."
        )
        if s["mean_energy_fusion_induced_regression"] is not None and s["mean_energy_baseline_correct"] is not None:
            diff = s["mean_energy_fusion_induced_regression"] - s["mean_energy_baseline_correct"]
            direction = "lower than" if diff < 0 else "higher than"
            lines.append("")
            lines.append(
                f"Fusion-induced regressions have residual_energy {direction} the correct-baseline "
                f"sample by {abs(diff):.4f} on average -- read this as a hint worth checking further "
                f"against more examples, not a conclusion on its own (n={s['n_fusion_induced_regression']} "
                f"is small by design, see this script's module docstring)."
            )
        lines.append("")
    output_md.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="v2", choices=["v1", "v2"])
    parser.add_argument("--predictions-csv", default=None, help="override the default results/<stage>_predictions.csv path")
    parser.add_argument("--v0-checkpoint", default=None, help="override the default checkpoints/v0_spatial_only/head.pt path")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"],
                         help="defaults to the analyzed checkpoint's own recorded mode")
    parser.add_argument("--conditions", nargs="+", default=None,
                         help=f"which eval_manifest.csv conditions to analyze (default: {DEFAULT_CONDITIONS})")
    parser.add_argument("--n-examples", type=int, default=10, help="most-confidently-wrong examples per condition")
    parser.add_argument("--n-baseline", type=int, default=30, help="correctly-classified comparison sample per condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()
    run_error_analysis(
        stage=args.stage,
        predictions_csv=args.predictions_csv,
        v0_checkpoint=args.v0_checkpoint,
        freq_mode=args.freq_mode,
        conditions=args.conditions,
        n_examples=args.n_examples,
        n_baseline=args.n_baseline,
        seed=args.seed,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
