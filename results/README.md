# Results log

One file per roadmap stage (see main README's "Experiment roadmap"),
never overwritten:

    v0_spatial_only.json
    v1_spatial_plus_frequency.json
    v2_plus_augmentation.json
    v3_plus_consistency.json
    v4_plus_gating.json          (only if we get to it)

Each should hold at minimum: clean accuracy/AUC, per-transform
accuracy/AUC (matching the brief's transform grid), and average
robustness drop vs. clean. This is what makes a "24 points -> 11 -> 8
-> 5" style narrative possible for the Devpost write-up and the pitch
-- keep every stage, don't overwrite.
