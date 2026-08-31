# Error analysis -- v2 (fft)

Source: results/v2_augmented_fft_predictions.csv

For each condition below, the featured misclassified examples are split into two groups by re-scoring them through V0's spatial-only checkpoint: **shared blind spot** (V0 was also wrong -- the frequency stream and fusion head had nothing better to fall back on) vs. **fusion-induced regression** (V0 was right, but the fused prediction is wrong -- worth checking whether residual_energy looks unusual for these). mean_energy_baseline_correct is the average residual_energy over a random sample of correctly-classified rows from the same condition, for comparison.

## clean

82 of 900 rows misclassified overall; 10 featured here.

Of the featured misclassified examples: 10 are shared blind spots (V0 also wrong), 0 are fusion-induced regressions (V0 was right).

Mean residual_energy -- baseline (correct): 2.2826, shared blind spot: 2.6232, fusion-induced regression: n/a.

## blur_sigma2.0

146 of 900 rows misclassified overall; 10 featured here.

Of the featured misclassified examples: 10 are shared blind spots (V0 also wrong), 0 are fusion-induced regressions (V0 was right).

Mean residual_energy -- baseline (correct): 1.8295, shared blind spot: 2.1646, fusion-induced regression: n/a.

## resize_0.25

146 of 900 rows misclassified overall; 10 featured here.

Of the featured misclassified examples: 10 are shared blind spots (V0 also wrong), 0 are fusion-induced regressions (V0 was right).

Mean residual_energy -- baseline (correct): 1.9765, shared blind spot: 2.3039, fusion-induced regression: n/a.
