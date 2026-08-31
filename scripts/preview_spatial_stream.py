"""
Smoke test for models/spatial_stream.py.

Needs torch, torchvision, and open-clip-torch installed (uv sync should
handle this) and will download the CLIP ViT-B/32 "openai" checkpoint
(~350MB) on first run -- make sure you've got the bandwidth/time for
that before running.

Verified against a live torch install. Rerun this after touching
models/spatial_stream.py or config.py's SPATIAL_* settings.

What it checks, and why each check matters:
  - the model reports 0 trainable parameters (confirms freeze=True
    actually took effect, not just that the model loaded)
  - encoding a single image produces a (512,) embedding with finite
    values (catches NaNs/shape mismatches early)
  - encoding a batch produces (N, 512) (confirms the batched path works,
    which cache_embeddings.py will rely on for speed)
  - two crops of the SAME image score more similar (cosine similarity)
    than two DIFFERENT images -- a real pass/fail signal that this is
    behaving like a meaningful embedding, not just returning noise that
    happens to be the right shape

Run with:  python scripts/preview_spatial_stream.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.datasets import load_manifest
from models.spatial_stream import SpatialStream


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    print("Loading CLIP ViT-B/32 (downloads the checkpoint on first run)...")
    stream = SpatialStream()

    n_trainable = sum(p.requires_grad for p in stream.model.parameters())
    n_total = sum(1 for _ in stream.model.parameters())
    print(f"Loaded {stream.model.__class__.__name__} -- "
          f"{n_trainable}/{n_total} parameter tensors trainable "
          f"(should be 0/{n_total} since freeze={stream.frozen})")

    rows = load_manifest(split="train")
    if len(rows) < 2:
        print("No training images in the manifest yet -- run `python src/data/datasets.py` first.")
        return

    img_a = Image.open(rows[0]["path"]).convert("RGB")
    img_b = Image.open(rows[1]["path"]).convert("RGB")
    print(f"\nimage A: {rows[0]['path']} (label={rows[0]['label']})")
    print(f"image B: {rows[1]['path']} (label={rows[1]['label']})")

    emb_a = stream.encode(stream.prepare(img_a))
    emb_b = stream.encode(stream.prepare(img_b))
    print(f"\nembedding shape: {tuple(emb_a.shape)} (expect (512,))")
    print(f"finite values: {torch.isfinite(emb_a).all().item()}")

    # Two overlapping crops of image A should be more similar to each
    # other than image A is to the unrelated image B.
    w, h = img_a.size
    crop1 = img_a.crop((0, 0, int(w * 0.9), int(h * 0.9)))
    crop2 = img_a.crop((int(w * 0.1), int(h * 0.1), w, h))
    emb_crop1 = stream.encode(stream.prepare(crop1))
    emb_crop2 = stream.encode(stream.prepare(crop2))

    sim_same_image = cosine_sim(emb_crop1, emb_crop2)
    sim_different_images = cosine_sim(emb_a, emb_b)
    print(f"\ncosine similarity, two crops of the SAME image: {sim_same_image:.3f}")
    print(f"cosine similarity, two DIFFERENT images:          {sim_different_images:.3f}")
    if sim_same_image > sim_different_images:
        print("PASS: same-image crops are more similar than different images, as expected.")
    else:
        print("UNEXPECTED: different images scored more similar than same-image crops -- "
              "something's likely wrong here, don't trust this embedding yet.")

    batch = torch.stack([stream.prepare(img_a), stream.prepare(img_b)])
    batch_emb = stream.encode(batch)
    print(f"\nbatch embedding shape: {tuple(batch_emb.shape)} (expect (2, 512))")


if __name__ == "__main__":
    main()
