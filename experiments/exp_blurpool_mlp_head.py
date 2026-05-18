"""Tweak: 2-layer MLP head (Linear -> BN -> SiLU -> Linear) on BlurPool + ResNet-101.

The locked recipe uses a single Linear(512, 37) classifier directly on
pooled features. Adding a BN'd hidden layer might give the head room
to refine the 512-dim pooled representation before the 37-way cut —
the deeper model has more abstract features and the classifier may
benefit from a learned non-linear refinement step.

  baseline head:  Pool -> Dropout -> Linear(512, 37)
  mlp head:       Pool -> Dropout -> Linear(512, 256) -> BN -> SiLU -> Linear(256, 37)

Param cost: tiny (~130K added on top of 41.7M = 0.3% increase).

Run from the repo root (~30-35 min):
    python experiments/exp_blurpool_mlp_head.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="blurpool_mlp_head",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
        head_kind="mlp",
        head_hidden=256,
        baseline_test_pct=70.26,
        notes="BlurPool + ResNet-101 + 2-layer MLP head (Linear-BN-SiLU-Linear), hidden=256.",
    )
