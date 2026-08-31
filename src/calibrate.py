"""
Calibration: fits a single-parameter temperature scaling on a trained
checkpoint's fusion-head logits, so scores reported by src/infer.py
are calibrated probabilities rather than a raw, typically overconfident
sigmoid -- see the architecture doc's §03 "Calibration, before the
score is reported as 'confidence'" amend for why this matters: the
brief asks for a calibrated probability explicitly, and any
false-positive-rate threshold discussion only means something once the
score is actually calibrated.

Temperature scaling: calibrated_prob = sigmoid(logit / T), fit by
choosing T to minimize binary cross-entropy against true labels on a
held-out calibration split. T=1 is "no change"; T>1 softens
overconfident predictions toward 0.5, T<1 sharpens them.

The calibration split matters as much as the fitting method: it must
be distinct from BOTH the training set and the robustness eval grid
(data/eval_manifest.csv), or the reported calibration would be
measuring how well the model fits data it was already tuned against.
validation_demo is already guaranteed distinct from training (see
data/datasets.py's build_manifest assertion) -- this script additionally
excludes every image already sampled into the eval grid, sampling the
calibration set from what's left over in validation_demo. At the
default SAMPLE_PER_SOURCE=300 in scripts/build_eval_grid.py, that
leaves roughly 14,543 of validation_demo's 14,843 images untouched --
comfortably enough to draw a calibration split from without touching
anything the robustness table depends on.

Calibration is fit on CLEAN images only, not the transform grid: the
goal here is a single global temperature for the model's natural
operating point, matching how src/infer.py will actually be used
(scoring images as given, not synthetically transformed first).

Writes checkpoints/<stage>/calibration.json:
    {"temperature": T, "n_calibration_images": N, "seed": ...,
     "nll_before": ..., "nll_after": ..., "ece_before": ..., "ece_after": ...}
src/infer.py loads this file (if present) and applies it; without it,
infer.py falls back to the uncalibrated sigmoid.

Run with:  uv run python src/calibrate.py --stage v2
           uv run python src/calibrate.py --stage v2 --n-calibration 2000
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import CHECKPOINT_DIR, FREQUENCY_MODE
from data.datasets import load_manifest
from models.frequency_stream import FrequencyStream
from models.fusion import FusionHead, load_architecture_metadata
from models.spatial_stream import SpatialStream

EVAL_MANIFEST = Path("data/eval_manifest.csv")

# Same checkpoint-dir-prefix convention evaluate.py's
# _evaluate_fusion_stage uses -- kept here rather than imported, since
# importing evaluate.py just for this one dict would pull in its
# argparse/main() setup for no reason.
CHECKPOINT_DIR_PREFIX = {"v1": "v1_fusion", "v2": "v2_augmented"}


def _already_in_eval_grid() -> set:
    """The set of original image paths already sampled into
    data/eval_manifest.csv -- excluded from the calibration split so
    calibration never reuses an image the robustness table depends on."""
    if not EVAL_MANIFEST.is_file():
        print(f"WARNING: {EVAL_MANIFEST} not found -- can't exclude eval-grid images from "
              f"calibration, proceeding without that guard (run scripts/build_eval_grid.py "
              f"first if you want it).")
        return set()
    with open(EVAL_MANIFEST, newline="") as f:
        return {row["original_path"] for row in csv.DictReader(f)}


def _sample_calibration_rows(n: int, seed: int):
    rows = load_manifest(split="validation_demo")
    used = _already_in_eval_grid()
    candidates = [r for r in rows if r["path"] not in used]
    print(f"validation_demo has {len(rows)} images total, {len(used)} already used in the "
          f"eval grid, {len(candidates)} available for calibration")
    if len(candidates) < n:
        print(f"WARNING: only {len(candidates)} candidates available, requested {n} -- using all of them")
        n = len(candidates)
    rng = random.Random(seed)
    sample = rng.sample(candidates, n)
    n_real = sum(1 for r in sample if int(r["label"]) == 0)
    print(f"sampled {len(sample)} images for calibration ({n_real} real, {len(sample) - n_real} fake)")
    return sample


def _binary_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Mean, bin-size-weighted |accuracy - confidence| across n_bins
    equal-width probability bins -- the diagnostic the architecture
    doc's §03 amend asks to report alongside the AUC table. A
    perfectly calibrated model scores 0; this is meant to be compared
    before vs. after fitting T, not read as an absolute pass/fail."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Coarse-to-fine grid search over T minimizing binary NLL of
    sigmoid(logit / T) against labels. A grid search instead of
    scipy.optimize.minimize_scalar: this is a 1-D, well-behaved,
    unimodal-in-practice objective, so a grid is exactly as correct and
    has no convergence/bracketing edge cases to worry about for a
    one-off fitting script like this."""
    def nll_at(T):
        probs = 1.0 / (1.0 + np.exp(-logits / T))
        return _binary_nll(probs, labels)

    best_T, best_nll = 1.0, nll_at(1.0)
    # Coarse pass: log-spaced from 0.05 to 20.
    for T in np.geomspace(0.05, 20.0, 200):
        nll = nll_at(T)
        if nll < best_nll:
            best_nll, best_T = nll, T
    # Fine pass: linear refine around the coarse best.
    lo, hi = max(best_T * 0.5, 1e-3), best_T * 1.5
    for T in np.linspace(lo, hi, 200):
        nll = nll_at(T)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return float(best_T)


def calibrate(stage: str = "v2", checkpoint_path: str = None, n_calibration: int = 1000,
              seed: int = 0, freq_mode: str = None, batch_size: int = 64):
    prefix = CHECKPOINT_DIR_PREFIX[stage]
    if checkpoint_path is None:
        mode_for_path = freq_mode or FREQUENCY_MODE
        ckpt_dir = Path(CHECKPOINT_DIR) / f"{prefix}_{mode_for_path}"
        checkpoint_path = ckpt_dir / "model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
        ckpt_dir = checkpoint_path.parent
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{checkpoint_path} doesn't exist -- train that stage first.")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    resolved_mode = freq_mode or ckpt.get("freq_mode") or FREQUENCY_MODE

    print("Loading CLIP backbone...")
    spatial_stream = SpatialStream()
    print(f"Loading frequency stream (mode={resolved_mode}) + fusion head from {checkpoint_path}...")
    freq_stream = FrequencyStream(freeze=True, mode=resolved_mode)
    freq_stream.model.load_state_dict(ckpt["frequency_cnn"])
    use_freq_gate = load_architecture_metadata(ckpt_dir)
    fusion = FusionHead(use_freq_gate=use_freq_gate)
    fusion.load_state_dict(ckpt["fusion_head"])
    fusion.eval()

    rows = _sample_calibration_rows(n_calibration, seed)
    if not rows:
        print("No calibration rows available -- nothing to fit.")
        return

    logits = np.empty(len(rows), dtype=np.float64)
    labels = np.array([int(r["label"]) for r in rows], dtype=np.float64)

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
            logit = fusion(spatial_emb, freq_emb, energy_batch)
        for local_i, global_i in enumerate(batch_indices):
            logits[global_i] = logit[local_i].item()
        spatial_tensors.clear()
        freq_tensors.clear()
        energies.clear()
        batch_indices.clear()

    from transforms.preprocessing import prepare_frequency_input, residual_energy

    for i, r in enumerate(tqdm(rows, desc="scoring calibration split")):
        img = Image.open(r["path"]).convert("RGB")
        spatial_tensors.append(spatial_stream.prepare(img))
        freq_map = prepare_frequency_input(img, mode=resolved_mode)
        freq_tensors.append(torch.from_numpy(freq_map).unsqueeze(0))
        energies.append(residual_energy(freq_map))
        batch_indices.append(i)
        if len(spatial_tensors) >= batch_size:
            flush()
    flush()

    probs_before = 1.0 / (1.0 + np.exp(-logits))
    nll_before = _binary_nll(probs_before, labels)
    ece_before = _expected_calibration_error(probs_before, labels)

    temperature = _fit_temperature(logits, labels)

    probs_after = 1.0 / (1.0 + np.exp(-logits / temperature))
    nll_after = _binary_nll(probs_after, labels)
    ece_after = _expected_calibration_error(probs_after, labels)

    result = {
        "stage": stage,
        "checkpoint": str(checkpoint_path),
        "freq_mode": resolved_mode,
        "temperature": temperature,
        "n_calibration_images": len(rows),
        "seed": seed,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    out_path = ckpt_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\ntemperature = {temperature:.4f}  (T=1.0 means no change)")
    print(f"NLL:  before={nll_before:.4f}  after={nll_after:.4f}")
    print(f"ECE:  before={ece_before:.4f}  after={ece_after:.4f}")
    print(f"\nWrote {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="v2", choices=["v1", "v2"])
    parser.add_argument("--checkpoint", default=None, help="override the default checkpoint path")
    parser.add_argument("--freq-mode", default=None, choices=["srm", "fft"])
    parser.add_argument("--n-calibration", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    calibrate(
        stage=args.stage,
        checkpoint_path=args.checkpoint,
        n_calibration=args.n_calibration,
        seed=args.seed,
        freq_mode=args.freq_mode,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
