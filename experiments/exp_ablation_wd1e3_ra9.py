"""Tweak: wd=1e-3 + RandAugment magnitude 9 (up from 7).

Mixup and RandomErasing both lost when stacked on wd=1e-3 - both fight
the model's ability to fit. RandAugment is the augmentation already
in the pipeline; just turning it up to magnitude 9 (the old original
setting before I dialed it back) might apply more pressure on the
overfit without introducing a wholly new training signal.

Earlier exp_trivial_augment lost at the smaller-data recipe; that's
not directly comparable since we have MaxPool + full trainval now.

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_wd1e3_ra9.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_ra9",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        rand_augment_magnitude=9,
        baseline_test_pct=56.39,
        notes="wd=1e-3 + RandAugment magnitude 9 (was 7).",
    )
