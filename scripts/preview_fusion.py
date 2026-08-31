"""
Smoke test for models/fusion.py, wired up end-to-end with the real
spatial + frequency streams on two real manifest images -- the last
integration check before committing to a full train_v1 run.

Verified against a live torch install, and superseded in practice by
train_v1/train_v2 actually training FusionHead end-to-end -- kept as a
fast, no-training-loop regression check to rerun after touching
models/fusion.py or either stream.

What it checks, and why each check matters:
  - FusionHead accepts the (512-d, 128-d, scalar) triple the streams
    actually produce and returns a single logit -- not a shape
    mismatch, which is the most likely integration bug given the three
    pieces were built and unit-tested separately.
  - a batch of 2 images produces a (2,) logits tensor, not (2, 1) or a
    silently-broadcast scalar -- train_v1's loss_fn(logits, label) needs
    matching shapes or BCEWithLogitsLoss will silently broadcast wrong.
  - torch.sigmoid(logits) lands in [0, 1] and is finite -- catches a
    NaN leaking in from anywhere upstream (e.g. a bad residual_energy
    division) before it reaches a real training run.
  - a full loss.backward() call reaches BOTH the frequency CNN's and
    the fusion head's parameters with nonzero gradients, while the
    frozen spatial stream's parameters get NONE -- this is the exact
    contract train_v1's optimizer depends on (it only holds
    freq_stream.model and fusion's parameters).

Run with:  python scripts/preview_fusion.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.datasets import load_manifest
from models.frequency_stream import FrequencyStream
from models.fusion import FusionHead
from models.spatial_stream import SpatialStream


def main():
    rows = load_manifest(split="train")
    if len(rows) < 2:
        print("No training images in the manifest yet -- run `python src/data/datasets.py` first.")
        return

    print("Loading CLIP backbone (frozen)...")
    spatial = SpatialStream()
    print("Building frequency stream (trainable) + fusion head (trainable)...")
    frequency = FrequencyStream(freeze=False)
    fusion = FusionHead()

    img_a = Image.open(rows[0]["path"]).convert("RGB")
    img_b = Image.open(rows[1]["path"]).convert("RGB")
    labels = torch.tensor([float(rows[0]["label"]), float(rows[1]["label"])])
    print(f"image A: {rows[0]['path']} (label={rows[0]['label']})")
    print(f"image B: {rows[1]['path']} (label={rows[1]['label']})")

    spatial_batch = torch.stack([spatial.prepare(img_a), spatial.prepare(img_b)])
    with torch.no_grad():
        spatial_emb = spatial.encode(spatial_batch)
    print(f"\nspatial embedding batch shape: {tuple(spatial_emb.shape)} (expect (2, 512))")

    freq_a, energy_a = frequency.prepare(img_a)
    freq_b, energy_b = frequency.prepare(img_b)
    freq_batch = torch.stack([freq_a, freq_b])
    energy_batch = torch.tensor([energy_a, energy_b], dtype=torch.float32)
    freq_emb = frequency.encode(freq_batch)
    print(f"frequency embedding batch shape: {tuple(freq_emb.shape)} (expect (2, 128))")
    print(f"residual_energy batch: {energy_batch.tolist()}")

    logits = fusion(spatial_emb, freq_emb, energy_batch)
    print(f"\nfusion logits shape: {tuple(logits.shape)} (expect (2,))")
    probs = torch.sigmoid(logits)
    print(f"fusion probabilities: {probs.tolist()}")
    in_range = bool(((probs >= 0) & (probs <= 1)).all())
    finite = bool(torch.isfinite(probs).all())
    print(f"all probabilities finite and in [0, 1]: {finite and in_range}")

    print("\nChecking gradients: frequency CNN + fusion head should get them, frozen spatial stream should NOT...")
    frequency.model.zero_grad()
    fusion.zero_grad()
    freq_emb_for_backward = frequency.encode(freq_batch)  # fresh forward pass -> a clean autograd graph to backward through
    logits = fusion(spatial_emb, freq_emb_for_backward, energy_batch)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    freq_grads = [p.grad for p in frequency.model.parameters() if p.requires_grad]
    fusion_grads = [p.grad for p in fusion.parameters() if p.requires_grad]
    spatial_grads = [p.grad for p in spatial.model.parameters()]

    freq_ok = any(g is not None and torch.any(g != 0) for g in freq_grads)
    fusion_ok = any(g is not None and torch.any(g != 0) for g in fusion_grads)
    spatial_untouched = all(g is None for g in spatial_grads)

    print(f"loss: {loss.item():.4f}")
    print(f"frequency CNN received nonzero gradients: {freq_ok} (expect True)")
    print(f"fusion head received nonzero gradients:   {fusion_ok} (expect True)")
    print(f"frozen spatial stream received NO gradients: {spatial_untouched} (expect True)")

    if freq_ok and fusion_ok and spatial_untouched and finite and in_range:
        print("\nPASS: the full spatial -> frequency -> fusion pipeline is wired correctly "
              "for train_v1 -- safe to start the real training run.")
    else:
        print("\nUNEXPECTED: something above didn't match -- don't start a full train_v1 "
              "run until this is fixed, it'll waste a lot of CPU time training on a bug.")


if __name__ == "__main__":
    main()
