"""Ablation: locked recipe with AdamW instead of SGD-Nesterov.

Only knob flipped vs the locked baseline (which lands 46.74% on test):
swap SGD(lr=0.1, momentum=0.9, Nesterov, wd=5e-4) for AdamW(lr=4e-3,
wd=1e-2). OneCycle max_lr is rescaled to 4e-3 because AdamW's effective
step is much larger than SGD's at the same LR; 4e-3 is the typical
small-CNN-from-scratch operating point and matches what the alt recipe
shared with me used.

Run from the repo root (about 20-25 minutes):
    python experiments/exp_ablation_adamw.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_adamw",
        optimizer_kind="adamw",
        notes=(
            "Locked recipe with AdamW(lr=4e-3, wd=1e-2) instead of "
            "SGD-Nesterov(lr=0.1, wd=5e-4). OneCycle pct_start stays at "
            "0.17. Architecture, augmentation, batch size, mixup, AMP, "
            "train/val split all match the locked recipe."
        ),
    )
