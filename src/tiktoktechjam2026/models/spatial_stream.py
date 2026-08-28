"""
Spatial stream: frozen (or lightly fine-tuned) CLIP ViT-B/32 backbone.

Captures high-level visual and structural representations of the image
that are comparatively less dependent on individual pixel values --
object identity, scene layout, overall composition -- as opposed to
the low-level pixel/frequency statistics the frequency stream looks
at. We are NOT claiming CLIP was trained to detect AI generation, or
that it explicitly represents things like "lighting consistency" --
that's not a defensible claim to make to judges. The narrower, actually
defensible claim: it's a rich, general-purpose embedding a classifier
can learn useful real-vs-fake distinctions from, and because it's
high-level rather than pixel-level, it tends to survive blur and
recompression better than raw pixel statistics do.

TODO (next step): wrap open_clip's ViT-B/32 image encoder, freeze its
parameters by default (config.FREEZE_BACKBONE), expose
encode(image_tensor) -> 512-d embedding.
"""


class SpatialStream:
    def __init__(self, freeze: bool = True):
        raise NotImplementedError

    def encode(self, image_tensor):
        raise NotImplementedError
