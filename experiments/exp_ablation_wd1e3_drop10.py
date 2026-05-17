"""Tweak: wd=1e-3 + stochastic depth (drop_path_rate=0.1).

Augmentation-based regularisers (mixup, RandomErasing, RandAugment+) all
hurt when stacked on wd=1e-3. Trying an architectural regulariser
instead: drop each residual branch's output with probability p
(per-sample), where p grows linearly from 0 at the first block to
drop_path_rate at the last. This is a *forward-pass* regulariser, not a
data one, so it shouldn't compete with the existing augmentation.

Reference: Huang et al. 2016, arXiv:1603.09382 ("Deep Networks with
Stochastic Depth").

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_wd1e3_drop10.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_drop10",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        drop_path_rate=0.1,
        baseline_test_pct=56.39,
        notes="wd=1e-3 + linear stochastic depth, max drop rate 0.1.",
    )
