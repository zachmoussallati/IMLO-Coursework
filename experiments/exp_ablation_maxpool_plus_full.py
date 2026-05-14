"""Ablation: locked recipe + MaxPool + full 3680-image trainval (no val).

Combining the two winning single-knob ablations. If they stack, we expect
something around 51.87 + 2.84 = ~54.7%; if they're not perfectly additive
the realised gain will be smaller, but the combination should still beat
either alone. Everything else (architecture, optimiser, schedule,
augmentation, batch size, mixup, AMP) stays identical to the locked
recipe.

Run from the repo root (about 20 minutes; MaxPool makes per-epoch
compute much cheaper):
    python experiments/exp_ablation_maxpool_plus_full.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_maxpool_plus_full",
        use_maxpool=True,
        use_full_trainval=True,
        notes=(
            "Two winning knobs combined: stem MaxPool + full-trainval "
            "training (no val split). Architecture, optimiser, schedule, "
            "augmentation, batch size, mixup, AMP all match locked."
        ),
    )
