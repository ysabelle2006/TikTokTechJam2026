"""
Spatial stream: frozen (or lightly fine-tuned) CLIP ViT-B/32 backbone.

Captures high-level visual and structural representations of the image
that are comparatively less dependent on individual pixel values --
object identity, scene layout, overall composition -- as opposed to
the low-level pixel/frequency statistics the frequency stream looks
at. We are NOT claiming CLIP was trained to detect AI generation, or
that it explicitly represents things like "lighting consistency" --
the defensible claim is narrower: it's a rich, general-purpose
embedding a classifier can learn useful real-vs-fake distinctions
from, and because it's high-level rather than pixel-level, it tends to
survive blur and recompression better than raw pixel statistics do.

Uses CLIP's own preprocessing transform (via open_clip) rather than a
hand-rolled resize/normalize, since CLIP was trained with a specific
normalization -- reinventing that risks a subtle mismatch that would
quietly hurt the embedding's quality.

NOT YET VERIFIED -- written without access to a torch-capable
environment. Run scripts/preview_spatial_stream.py once dependencies
are installed and report back what it prints, including any error.
"""

import torch
import open_clip

from config import FREEZE_BACKBONE, SPATIAL_BACKBONE, SPATIAL_EMBED_DIM, SPATIAL_PRETRAINED


class SpatialStream:
    def __init__(self, freeze: bool = FREEZE_BACKBONE, device: str = "cpu"):
        self.device = device
        self.frozen = freeze

        model, _, preprocess = open_clip.create_model_and_transforms(
            SPATIAL_BACKBONE, pretrained=SPATIAL_PRETRAINED
        )
        model.eval()
        model.to(device)
        if freeze:
            for p in model.parameters():
                p.requires_grad_(False)

        self.model = model
        self.preprocess = preprocess  # PIL.Image -> normalized tensor, CLIP's own transform

    def prepare(self, pil_image):
        """PIL.Image (RGB) -> a single preprocessed tensor, ready for encode()."""
        return self.preprocess(pil_image)

    @torch.no_grad()
    def encode(self, image_tensor):
        """
        image_tensor: a single preprocessed tensor from prepare() (shape
        (3, H, W)), or a batch (N, 3, H, W). Returns a (SPATIAL_EMBED_DIM,)
        embedding for a single image, or (N, SPATIAL_EMBED_DIM) for a batch.

        Always runs under no_grad, even if freeze=False -- if we ever
        fine-tune the backbone's last block (V4), that needs a separate
        method that doesn't wrap in no_grad. This one is for
        inference/caching only.
        """
        single = image_tensor.dim() == 3
        batch = image_tensor.unsqueeze(0) if single else image_tensor
        batch = batch.to(self.device)

        features = self.model.encode_image(batch)
        assert features.shape[-1] == SPATIAL_EMBED_DIM, (
            f"expected {SPATIAL_EMBED_DIM}-d embeddings from {SPATIAL_BACKBONE}, "
            f"got {features.shape[-1]} -- config.py's SPATIAL_EMBED_DIM is out of sync "
            f"with the actual backbone"
        )
        return features.squeeze(0) if single else features
