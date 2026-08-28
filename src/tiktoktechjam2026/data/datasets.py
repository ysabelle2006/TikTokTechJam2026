"""
Dataset loaders for the AIGC detection sources named in the brief:
  - CIFAKE (small, fast to iterate on -- good first target)
  - SID_Set
  - WildFake (translate via the ModelScope UI before use, per the brief)

The validation set (COCO val2017 vs. the DALL-E Advanced subset of
WildFake) stays entirely separate: it's for demonstrating progress
only and must never be trained on.

TODO (next step): once we know where the raw data will live locally,
implement a dataset class per source with a consistent (image, label)
interface, plus a generator_family field so we can later hold one
family out for the generalization test.
"""
