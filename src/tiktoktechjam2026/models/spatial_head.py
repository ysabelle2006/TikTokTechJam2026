"""
V0 classification head: CLIP embedding -> one logit.

The entire V0 model is `frozen CLIP -> this`. It answers the baseline
question: does a small head on top of a general-purpose CLIP embedding
separate real from fake at all, before we add any forensic signal?

LayerNorm on the raw 512-d embedding first -- CLIP features are not unit
scaled and a bare linear layer trains poorly on them.
"""

import torch
import torch.nn as nn

from tiktoktechjam2026 import config


class SpatialHead(nn.Module):
    def __init__(self, in_dim: int = None, hidden_dims=None):
        super().__init__()
        in_dim = in_dim or config.SPATIAL_EMBED_DIM
        hidden_dims = hidden_dims or config.SPATIAL_HEAD_HIDDEN_DIMS

        layers = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, spatial_embedding: torch.Tensor) -> torch.Tensor:
        """[B, 512] -> [B] raw logit."""
        return self.net(spatial_embedding).squeeze(-1)
