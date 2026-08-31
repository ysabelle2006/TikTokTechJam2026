# Results log

One JSON file per evaluated stage (see main README's "Experiment
roadmap" and "Robustness Evaluation Summary"), written by
`evaluate.py` and never overwritten by a later stage:

    v0_spatial_only.json        V0 -- spatial stream only
    v1_fusion_srm.json          V1 -- + frequency stream ("srm" mode)
    v1_fusion_fft.json          V1 -- + frequency stream ("fft" mode, current default)
    v2_augmented_fft.json       V2 -- + transform-aware augmentation (current best single checkpoint)
    v2_augmented_fft_gated.json V2 + frequency gate + failure-condition reweighting (ablation, see main README)

Each holds clean/robust AUC per condition (both pooled and the
unseen-generator view), the avg robust AUC/drop and Final Score
roadmap-table numbers, and -- for files written by the current
evaluate.py -- Accuracy/FPR/FNR at a fixed decision threshold per
condition too (see evaluate.py's module docstring; v0/v1's files
predate that addition, so only the two v2 files currently have it).

Also here:
  - `v2_augmented_fft_predictions.csv` / `v2_augmented_fft_gated_predictions.csv`
    (both gitignored via `results/*_predictions.csv`, regenerate with
    `evaluate.py --stage v2 [--checkpoint ... --result-name ...] --dump-predictions`)
    -- one row per eval_manifest.csv row plus its raw prediction;
    error_analysis.py and the gate-vs-baseline comparison in the main
    README's ablation section both read these rather than re-scoring.
  - `v2_error_analysis.csv` / `v2_error_analysis.md` -- error_analysis.py's
    output for the baseline v2_augmented_fft checkpoint: specific
    misclassified examples, classified as shared_blind_spot vs.
    fusion_induced_regression. Not regenerated for the gated checkpoint
    (the aggregate comparison in the main README covers that ablation's
    error pattern instead).

`v2_augmented_fft_gated.json` was written by
`train.py --stage v2 --freq-mode fft --use-freq-gate` +
`evaluate.py --result-name v2_augmented_fft_gated` -- see
`evaluate.py --result-name` and `train.py`'s `_gated` checkpoint-dir
suffix for the mechanism that keeps it from ever overwriting the
baseline checkpoint or results it's compared against. It is
deliberately NOT the "current best" row in the main README's roadmap
table -- it trades a broader FNR increase and a small
held-out-generator AUC regression for a real, targeted FPR fix, so
it's reported as a documented alternative rather than a replacement.
