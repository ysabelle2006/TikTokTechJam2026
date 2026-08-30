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
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, hidden_dims=(128, 64)):
        super().__init__()

        input_dim = 512 + 128 + 1

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        # One output logit for binary classification
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        spatial_embedding,
        frequency_embedding,
        residual_energy,
    ):
        fused = torch.cat(
            [
                spatial_embedding,
                frequency_embedding,
                residual_energy,
            ],
            dim=1,
        )

        logit = self.mlp(fused)

        return logit
