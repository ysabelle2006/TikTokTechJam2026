"""
Offline feature-extraction step: run the frozen CLIP backbone once over
every image (the clean version, plus a fixed set of augmented variants)
and save the resulting 512-d embeddings to disk.

Why this exists: repeatedly running a ViT-B/32 forward pass on CPU,
once per image per epoch, is the actual compute bottleneck in this
project -- not the small frequency CNN or fusion head. Precomputing
embeddings once means later training epochs read cached vectors
instead of recomputing them, which is what actually makes the
frozen-backbone version CPU-feasible.

Trade-off worth knowing: this only works because the backbone is
frozen. If V4 unfreezes even part of it, embeddings change every
training step and this caching step no longer applies for that stage
-- fall back to running CLIP live there.

Also implies a design choice: rather than sampling a fresh random
augmentation every epoch, we fix a finite set of variants per image
(one rendering per parameter value in the brief's transform grid) and
cache all of them. That's a reasonable trade for CPU feasibility, and
it conveniently matches how the robustness evaluation is already
structured around discrete severities.

TODO: implement once transforms/preprocessing.py and
models/spatial_stream.py exist.
"""


def main(image_dir: str, cache_dir: str):
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("wire up argparse once SpatialStream exists")
