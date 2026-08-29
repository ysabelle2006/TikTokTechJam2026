"""
Spatial stream: frozen CLIP ViT-B/32 backbone.

Captures high-level visual / structural representation of the image --
object identity, scene layout, composition -- which is comparatively less
tied to individual pixel values than the frequency stream's view. We are
NOT claiming CLIP was trained to detect AI generation. The defensible
claim: it is a rich, general-purpose embedding a small classifier can
learn real-vs-fake distinctions from, and because it is high-level rather
than pixel-level it tends to survive blur and recompression better than
raw pixel statistics.

Frozen by default (config.FREEZE_BACKBONE) -- this is what makes the
offline-embedding-cache trick in cache_embeddings.py valid and the whole
pipeline CPU-feasible. V4 may unfreeze the last block; that path would
have to run CLIP live.
"""

import open_clip
import torch
import torch.nn as nn

from tiktoktechjam2026 import config


class SpatialStream(nn.Module):
    def __init__(self, freeze: bool = None):
        super().__init__()
        freeze = config.FREEZE_BACKBONE if freeze is None else freeze

        model = open_clip.create_model(
            config.SPATIAL_BACKBONE, pretrained=config.SPATIAL_PRETRAINED
        )
        # We only need the image tower.
        self.visual = model.visual
        self.frozen = freeze
        if freeze:
            self.visual.eval()
            for p in self.visual.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Keep a frozen backbone in eval mode regardless of the parent's
        # train()/eval() calls (matters for any dropout / LayerNorm stats).
        super().train(mode)
        if self.frozen:
            self.visual.eval()
        return self

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: [B, 3, 224, 224], CLIP-normalized (see
        preprocessing.prepare_spatial_input).
        Returns: [B, 512] float32 embedding.
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feats = self.visual(image_tensor)
        return feats.float()

    forward = encode
