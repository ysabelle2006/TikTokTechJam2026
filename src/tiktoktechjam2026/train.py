"""
Training entry point.

Objective (V3, our main proposed method -- see README roadmap for
V0-V2):

    L = L_classification(y_hat, y) + L_classification(y_hat_t, y)
        + lambda * L_consistency(y_hat, y_hat_t)

where y_hat = Detector.predict(image), y_hat_t = Detector.predict(transform(image))
using the SAME weights, and L_consistency penalizes the two predictions
for disagreeing (e.g. MSE or symmetric KL on the probabilities). lambda
is config.CONSISTENCY_LOSS_WEIGHT -- log L_classification and
L_consistency separately during training, not just their sum: a
consistency term weighted too heavily can collapse both predictions
toward an uninformative middle value instead of actually improving
robustness.

Roadmap (build and evaluate in this order; keep every checkpoint and
its metrics under results/, never overwrite):
    V0  spatial stream only                    (baseline)
    V1  + frequency stream                     (does fusion help?)
    V2  + transform-aware augmentation         (does robustness improve?)
    V3  + consistency loss                     (our main method)
    V4  optional: learned gating / partial backbone fine-tuning

TODO: implement once data/datasets.py and the transform pipeline exist.
Depends on cache_embeddings.py if training against precomputed spatial
embeddings rather than running CLIP live (see that file for why).
"""
