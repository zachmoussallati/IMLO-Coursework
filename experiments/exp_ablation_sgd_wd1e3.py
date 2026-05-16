"""Tweak: SGD-Nesterov with heavier weight decay (1e-3 vs 5e-4).

The locked recipe trains the promoted model (MaxPool + full trainval) to
Q14 76.58 % vs Q15 54.13 % - a 22 pp train-test gap. That suggests some
overfit; stronger weight decay is the obvious regulariser knob.

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_sgd_wd1e3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_sgd_wd1e3",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        baseline_test_pct=54.13,
        notes=(
            "SGD-Nesterov with weight_decay=1e-3 (2x the locked 5e-4). "
            "Targeting the 22pp train-test gap on the promoted recipe."
        ),
    )
