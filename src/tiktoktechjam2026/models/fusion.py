"""
Fusion head (V1): concatenates both streams' outputs and predicts
P(image is AI-generated).

Input:  [spatial_embedding (512-d), frequency_embedding (128-d),
         residual_energy_scalar (1-d)]  ->  641-d vector
Output: sigmoid(logit) in [0, 1]

Honest framing, worth keeping in mind while implementing: nothing here
*guarantees* the network learns to lean on the frequency stream when
it's intact and fall back on the spatial stream when it isn't -- that's
a plausible outcome of joint training, not a designed-in behavior. The
residual-energy scalar gives the MLP an explicit hint rather than
making it infer reliability purely from embedding statistics, but it's
still just an extra input feature, not a mechanism that forces that
behavior.

If V1-V3 (see README roadmap) show the network isn't actually using
that hint well, a natural upgrade (V4) is an explicit gate: a small
network conditioned on the residual-energy scalar that outputs a
weight in [0, 1] and multiplies the frequency embedding before fusion,
so the reliance on each stream becomes something we can point to
directly (e.g. "JPEG-30 detected -> frequency weight dropped to 0.18")
rather than something we hope the MLP discovered on its own. Don't
build this before V1-V3 exist -- there's nothing to diagnose it against
yet.

TODO (next step): small MLP, config.FUSION_HIDDEN_DIMS, ending in a
single logit + sigmoid.
"""


import torch
from torch import nn


class FusionHead(nn.Module):
    """
    V1 fusion head.

    Inputs:
        spatial_embedding:   [B, 512]
        frequency_embedding: [B, 128]
        residual_energy:     [B, 1]

    Combined:
        512 + 128 + 1 = 641 dimensions

    Output:
        logits: [B, 1]
    """

    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(641, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(
        self,
        spatial_embedding,
        frequency_embedding,
        residual_energy,
    ):
        x = torch.cat(
            [
                spatial_embedding,
                frequency_embedding,
                residual_energy,
            ],
            dim=1,
        )

        return self.fusion(x)