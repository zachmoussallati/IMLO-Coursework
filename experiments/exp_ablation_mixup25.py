"""Tweak: re-enable mixup / cutmix at low probability (0.25).

Earlier exp_ablation_full_trainval + maxpool combined got us to 54.13%
with mix_prob=0. The 22pp train-test gap suggests room for more
regularisation; mixup was previously dropped because it broke the
*smaller-data* recipe, but with the larger training pool and the MaxPool
capacity the additional regulariser might now help close the gap rather
than starve learning.

mix_prob=0.25 means a quarter of batches get mixed (50/50 mixup vs
cutmix when triggered), so the model still sees plenty of clean batches.

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_mixup25.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    # Stacking with wd=1e-3 since that won +2.26pp on its own.
    run_ablation(
        name="ablation_wd1e3_mixup25",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        mix_prob=0.25,
        baseline_test_pct=54.13,
        notes=(
            "Stacking the wd=1e-3 winner with mix_prob=0.25. Both target "
            "the train-test gap; hypothesis is they compound (or at least "
            "don't cancel) and push past 56.39%."
        ),
    )
