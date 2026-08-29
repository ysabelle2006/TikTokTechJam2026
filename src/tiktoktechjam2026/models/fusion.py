"""
Fusion head (V1): concatenate both streams' outputs -> one logit.

Input:  spatial_embedding (512-d)
        frequency_embedding (128-d)
        residual_energy scalar (1-d)     -> 641-d  (config.FUSION_INPUT_DIM)
Output: one raw logit; sigmoid(logit) = P(image is AI-generated)

Honest framing: nothing here *forces* the network to lean on the
frequency stream when it is intact and fall back on the spatial stream
when it is not -- that is a plausible outcome of joint training, not a
designed-in mechanism. The residual-energy scalar is just an extra input
feature that makes the reliability hint explicit rather than something
the MLP has to infer from embedding statistics. If V1-V3 show the hint is
not being used well, V4's learned gate is the upgrade.
"""

import torch
import torch.nn as nn

from tiktoktechjam2026 import config


class FusionHead(nn.Module):
    def __init__(self, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or config.FUSION_HIDDEN_DIMS

        # Normalize the CLIP embedding on its own (unit-scaled features train
        # better); the 128-d frequency embedding already comes off a BatchNorm
        # stack and the scalar is small, so a plain concat is fine after that.
        self.spatial_norm = nn.LayerNorm(config.SPATIAL_EMBED_DIM)

        layers = []
        prev = config.FUSION_INPUT_DIM
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        spatial_embedding: torch.Tensor,
        frequency_embedding: torch.Tensor,
        residual_energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        spatial_embedding:   [B, 512]
        frequency_embedding: [B, 128]
        residual_energy:     [B] or [B, 1]
        -> [B] raw logit.
        """
        if residual_energy.ndim == 1:
            residual_energy = residual_energy.unsqueeze(-1)
        x = torch.cat(
            [self.spatial_norm(spatial_embedding), frequency_embedding, residual_energy],
            dim=-1,
        )
        return self.net(x).squeeze(-1)
