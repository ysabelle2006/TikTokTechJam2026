"""
Per-stream input preparation.

The two model streams see different things:
  - spatial stream:    resized + normalized RGB image (whatever the
                        pretrained vision backbone expects)
  - frequency stream:  grayscale, high-pass-filtered residual (or FFT
                        magnitude map) that exposes generator artifacts

TODO (next step): implement prepare_spatial_input and
prepare_frequency_input, plus residual_energy(residual_map) -- the
scalar reliability signal from the architecture doc (roughly: how much
high-frequency energy survived, used to help the fusion head know when
to discount the frequency stream).
"""


from torchvision import transforms


def prepare_spatial_input(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    image = image.convert("RGB")

    return preprocess(image)


def prepare_frequency_input(image):
    raise NotImplementedError


def residual_energy(residual_map):
    raise NotImplementedError

