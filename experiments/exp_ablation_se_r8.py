"""Tweak: SE reduction ratio 16 -> 8 on the deeper (3,4,6,3) recipe.

The Squeeze-and-Excitation bottleneck has been at reduction=16 since
the original architecture choice (~150K extra params at the locked
widths, <1% of model size). Halving the ratio to 8 doubles the SE
bottleneck width — more capacity to model channel-channel interactions
— at the cost of ~150K more params on top of 21.5M (~0.5% increase).

  baseline (r=16, floor 8 ch): bottlenecks {8, 8, 16, 32}
  r=8 (floor 8 ch):           bottlenecks {8, 16, 32, 64}

Bet: the deeper (3,4,6,3) network has more residual blocks each
calling SE; richer per-block gating could compound across the 16
blocks even though any single block's lift is small.

Run from the repo root (~22-25 min):
    python experiments/exp_ablation_se_r8.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_se_r8",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 6, 3),
        se_reduction=8,
        baseline_test_pct=62.09,
        notes="Deeper (3,4,6,3) + SE reduction = 8 (was 16).",
    )
