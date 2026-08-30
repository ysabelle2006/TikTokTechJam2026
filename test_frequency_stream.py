import torch

from tiktoktechjam2026.models.frequency_stream import FrequencyStream


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

x = torch.randn(
    4,
    3,
    224,
    224,
).to(device)


for mode in ["srm", "fft"]:
    print()
    print("Testing:", mode)

    model = FrequencyStream(
        mode=mode
    ).to(device)

    embedding, energy = model.encode(x)

    print(
        "Embedding shape:",
        embedding.shape
    )

    print(
        "Energy shape:",
        energy.shape
    )

    print(
        "Finite embedding:",
        torch.isfinite(
            embedding
        ).all().item()
    )

    print(
        "Finite energy:",
        torch.isfinite(
            energy
        ).all().item()
    )