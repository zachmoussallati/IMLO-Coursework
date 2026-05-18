"""Tweak: deeper (3,4,6,3) AT widths (80, 160, 320, 640).

The deeper-blocks promotion landed at the original 64-channel ladder
(Q15 62.09 %). User flagged the zip is only ~40 MB and we have ~58 MB
of headroom. This experiment spends some of that headroom by widening
the channel ladder by 1.25× while keeping the (3,4,6,3) depth that
won.

  current locked: (64,128,256,512)  × (3,4,6,3)  21.5 M params, ~41 MB fp16
  this experiment: (80,160,320,640) × (3,4,6,3)  33.5 M params, ~64 MB fp16

Bet: at wd=1e-3, the (3,4,6,3) depth used the original-width budget
productively (+3.19pp). The 1.5× width-only variant at the *shallower*
2-block layout also won (+2.51pp). Stacking the two should not be
strictly additive — the joint move adds 50 % more parameters than
either alone — but a meaningful fraction of each could carry over.

Run from the repo root (~25-35 min, slower than (3,4,6,3) at original widths):
    python experiments/exp_ablation_deep_wide.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_deep_wide",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(80, 160, 320, 640),
        blocks_per_stage=(3, 4, 6, 3),
        baseline_test_pct=62.09,
        notes="(80,160,320,640) widths x (3,4,6,3) depth, ~33.5M params, ~64MB fp16.",
    )
