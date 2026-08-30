import torch

from tiktoktechjam2026.models.fusion import FusionHead


model = FusionHead()

spatial_embedding = torch.randn(4, 512)
frequency_embedding = torch.randn(4, 128)
residual_energy = torch.randn(4, 1)

logits = model(
    spatial_embedding,
    frequency_embedding,
    residual_energy,
)

print("Output shape:", logits.shape)
print("Finite:", torch.isfinite(logits).all().item())

probabilities = torch.sigmoid(logits)

print("Probability shape:", probabilities.shape)
print("Example probabilities:", probabilities[:4])