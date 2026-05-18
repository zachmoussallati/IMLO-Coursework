"""Tweak: BlurPool + extra depth (3,4,30,3) at original widths.

The current locked recipe is BlurPool + ResNet-101 (3,4,23,3) at
Q15=70.26%. Adding 7 more stage-3 blocks (23 -> 30) costs ~8M params
and pushes model.pth from 80MB to ~95MB fp16 (zip ~96MB - just under
the 100MB cap).

Depth has compounded at every prior step (6 -> 9 -> 23 stage-3 blocks
all won), but ResNet-101 is already under-trained at 30 epochs
(clean_train 91% at the BlurPool variant — not memorising yet). More
depth at the same training budget might over-shoot the optimisation
budget, but it might also pay off if the extra layers help the model
better disentangle the breeds.

Run from the repo root (~70-90 min):
    python experiments/exp_blurpool_deeper.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="blurpool_deeper",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 30, 3),
        use_blurpool=True,
        baseline_test_pct=70.26,
        notes="BlurPool + extra depth (3,4,30,3), ~50M params, ~95MB fp16.",
    )
