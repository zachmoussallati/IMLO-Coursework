"""Tweak: push SGD weight decay further to 2e-3.

wd=1e-3 won +2.26pp (54.13 -> 56.39). Mixup-on-top hurt. Trying another
2x of weight decay (5e-4 -> 1e-3 -> 2e-3) to see if the regularisation
dose-response curve is still climbing or whether 1e-3 was the peak.

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_sgd_wd2e3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_sgd_wd2e3",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=2e-3,
        baseline_test_pct=56.39,
        notes="Heavier weight decay 2e-3 on the wd=1e-3 winning leaf.",
    )
