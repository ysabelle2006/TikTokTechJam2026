"""
Top-level model: wires the spatial stream, frequency stream, and
fusion head into one callable that goes image -> confidence score.

This is the single entry point both training and inference should
use, so the two never drift apart.

Note on how this gets used differently by different scripts:
infer.py and evaluate.py call predict() once per image. train.py calls
it TWICE per training example -- once on the clean image, once on a
transformed copy -- reusing these exact same weights, so the
classification and consistency losses are computed against genuinely
shared-weight predictions rather than two separate models. See
train.py for the full objective.

TODO (after spatial_stream.py, frequency_stream.py, fusion.py exist):
implement Detector.predict(image) -> float end to end.
"""


class Detector:
    def __init__(self):
        raise NotImplementedError

    def predict(self, image) -> float:
        raise NotImplementedError
