"""
Smoke test for models/frequency_stream.py.

Needs torch installed (uv sync should handle this) -- no download
required, unlike preview_spatial_stream.py, since this network trains
from scratch rather than loading a pretrained checkpoint.

Verified against a live torch install; transforms/preprocessing.py,
which this depends on, was separately verified with plain numpy/scipy
(scripts/preview_frequency_input.py). Rerun this after touching
models/frequency_stream.py or transforms/preprocessing.py.

What it checks, and why each check matters:
  - freeze=False (the default -- this network trains from scratch, see
    the module docstring) reports ALL parameters trainable; freeze=True
    reports ZERO trainable. Catches a copy-paste bug from
    SpatialStream's freeze handling landing backwards here.
  - prepare() returns a (tensor, energy) pair -- NOT just a tensor like
    SpatialStream.prepare() -- with the tensor at the expected shape.
  - encoding a single prepared image produces a (FREQUENCY_EMBED_DIM,)
    embedding with finite values; a batch produces
    (N, FREQUENCY_EMBED_DIM).
  - gradients actually reach the network's weights after a backward
    pass on a freeze=False instance -- the one thing that actually
    matters for this stream (it's untrained, so its embeddings aren't
    meaningful yet; what matters is that training CAN happen). If this
    fails, something is silently blocking gradient flow (e.g. an
    accidental torch.no_grad(), a .detach(), or a frozen flag set
    somewhere it shouldn't be).

Run with:  python scripts/preview_frequency_stream.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import FREQUENCY_EMBED_DIM
from models.frequency_stream import FrequencyStream


def make_test_image(size=224):
    img = Image.new("RGB", (size, size), color=(120, 140, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse((size * 0.2, size * 0.2, size * 0.8, size * 0.8), outline=(0, 255, 0), width=4)
    for i in range(0, size, 10):
        draw.line((i, 0, i, size // 5), fill=(255, 255, 255), width=1)
    return img


def main():
    print("Building FrequencyStream (freeze=False, the default -- trains from scratch)...")
    stream = FrequencyStream(freeze=False)
    n_trainable = sum(p.requires_grad for p in stream.model.parameters())
    n_total = sum(1 for _ in stream.model.parameters())
    print(f"freeze=False: {n_trainable}/{n_total} parameter tensors trainable (expect {n_total}/{n_total})")

    n_params = sum(p.numel() for p in stream.model.parameters())
    print(f"total parameters: {n_params:,} (architecture doc's parameter budget: ~0.6M)")

    frozen_stream = FrequencyStream(freeze=True)
    n_trainable_frozen = sum(p.requires_grad for p in frozen_stream.model.parameters())
    print(f"freeze=True:  {n_trainable_frozen}/{n_total} parameter tensors trainable (expect 0/{n_total})")

    img_a = make_test_image()
    img_b = make_test_image()
    prepared_a, energy_a = stream.prepare(img_a)
    prepared_b, energy_b = stream.prepare(img_b)
    print(f"\nprepared tensor shape: {tuple(prepared_a.shape)} (expect (1, H, W))")
    print(f"residual_energy (a plain float, not a tensor): {energy_a!r}")

    emb_a = stream.encode(prepared_a)
    print(f"single-image embedding shape: {tuple(emb_a.shape)} (expect ({FREQUENCY_EMBED_DIM},))")
    print(f"finite values: {torch.isfinite(emb_a).all().item()}")

    batch = torch.stack([prepared_a, prepared_b])
    batch_emb = stream.encode(batch)
    print(f"batch embedding shape: {tuple(batch_emb.shape)} (expect (2, {FREQUENCY_EMBED_DIM}))")

    print("\nChecking gradients actually flow (freeze=False instance)...")
    stream.model.zero_grad()
    out = stream.encode(batch)
    out.sum().backward()
    grads = [p.grad for p in stream.model.parameters() if p.requires_grad]
    any_grad = any(g is not None and torch.any(g != 0) for g in grads)
    if any_grad:
        print("PASS: at least one parameter received a nonzero gradient -- this network can train.")
    else:
        print("UNEXPECTED: no nonzero gradients reached the network's parameters -- "
              "something is blocking gradient flow, don't trust training with this yet.")


if __name__ == "__main__":
    main()
