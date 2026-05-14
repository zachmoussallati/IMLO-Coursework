"""Ablation: locked recipe trained on the full 3680-image trainval.

Only knob flipped vs the locked baseline (which lands 46.74% on test):
`use_full_trainval=True`. Drops the stratified 90/10 split and trains on
all 3680 trainval images; there's no held-out val so the per-epoch val
column reads "n/a" in the log. Q14 is now "fit on the same data I
trained on" - useful as an upper bound on capacity, no longer an
unbiased generalisation signal. Q15 (test) is the only thing that can
tell us whether the extra 368 training samples are worth it.

Run from the repo root (about 25 minutes):
    python experiments/exp_ablation_full_trainval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_full_trainval",
        use_full_trainval=True,
        notes=(
            "Locked recipe trained on all 3680 trainval images. No "
            "validation split. Architecture, optimiser, schedule, "
            "augmentation, batch size, mixup, AMP all match locked."
        ),
    )
