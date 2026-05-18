"""Capacity sweep: ResNet-101 layout - (64,128,256,512) widths x (3,4,23,3) depth.

The real ResNet-101 block distribution: 23 blocks in stage 3. Tests
whether depth-only at the original widths beats the
deep+wide combination. ~41.7M params, ~80MB fp16, ~82MB zip.

Note: stage 3 with 23 blocks is much deeper than anything we've tried.
Vanishing gradient risk is mitigated by the SE gating and pre-activation
ordering, but 30 epochs may not be enough optimiser budget for so deep
a network to settle.

Run from the repo root (~30-40 min):
    python experiments/exp_cap_resnet101.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="cap_resnet101",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 23, 3),
        baseline_test_pct=63.42,
        notes="Capacity sweep: ResNet-101 layout (3,4,23,3) at original widths, ~41.7M params.",
    )
