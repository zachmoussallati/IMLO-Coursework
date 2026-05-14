"""Ablation: locked recipe with ImageNet-style MaxPool after the stem.

Only knob flipped vs the locked baseline (which lands 46.74% on test):
`use_maxpool=True`. This makes the stem `Conv 3x3 stride-2 -> MaxPool 3x3
stride-2 padding-1`, so stage 1 sees a 56x56 feature map instead of
112x112. Everything else (architecture, optimiser, schedule,
augmentation, batch size, mixup, AMP, train/val split) is identical.

Run from the repo root (about 20-25 minutes):
    python experiments/exp_ablation_maxpool.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_maxpool",
        use_maxpool=True,
        notes=(
            "Locked recipe with use_maxpool=True. Stage 1 starts at 56x56 "
            "instead of 112x112. Everything else matches the locked recipe."
        ),
    )
