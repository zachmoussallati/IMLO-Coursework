"""Tweak: wd=1e-3 + longer OneCycle warmup (pct_start=0.25, was 0.17).

LR-schedule shape is an orthogonal axis I haven't explored. The locked
recipe ramps from base_lr/25 to max_lr=0.1 over the first 17% of steps,
then anneals cosine to ~0 over the remaining 83%. A longer warmup
(25%) gives the model more steps at moderate LR before the peak,
which tends to help when the model is regularised more aggressively
(wd=1e-3) and the early gradients are noisier.

Reference: Smith & Topin 2018, arXiv:1708.07120 ("Super-Convergence:
Very Fast Training of Neural Networks Using Large Learning Rates").

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_wd1e3_pct25.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_pct25",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        one_cycle_pct_start=0.25,
        baseline_test_pct=56.39,
        notes="wd=1e-3 + OneCycle pct_start=0.25 (was 0.17).",
    )
