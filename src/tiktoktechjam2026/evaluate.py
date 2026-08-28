"""
Robustness evaluation harness.

Runs whichever detector variant (V0-V4) on the clean test set and on
each transform x severity combination from the brief, logging
accuracy / AUC / false-positive-rate / false-negative-rate. Doubles as
both the "Robustness Evaluation Summary" deliverable and the ablation
table that makes the incremental-improvement story legible:

    Detector             | Clean | JPEG | Blur | Resize | Avg robust drop
    ----------------------+-------+------+------+--------+-----------------
    Spatial only (V0)     |       |      |      |        |
    Frequency only        |       |      |      |        |
    Spatial + frequency   |       |      |      |        |
    + augmentation (V2)   |       |      |      |        |
    + consistency (V3)    |       |      |      |        |

Write results for each stage to results/<stage>.json rather than
overwriting a single file -- see results/README.md.

TODO: build once training + a first checkpoint (V0) exist.
"""
