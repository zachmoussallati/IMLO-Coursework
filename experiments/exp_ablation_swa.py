"""Tweak: Stochastic Weight Averaging on the deeper (3,4,6,3) recipe.

SWA (Izmailov et al. 2018, arXiv:1803.05407): collect weight snapshots
in the tail of training, average them, and use the averaged weights
for inference. The intuition is that the optimiser's trajectory in the
late phase oscillates around a flat region of the loss landscape — the
averaged point sits closer to the centre of that flat region and
generalises better than any single snapshot.

Setup:
  - swa_start_epoch = 21 -> first snapshot at end of epoch 21
  - 10 snapshots collected (epochs 21..30)
  - After the last epoch: apply averaged weights, recalibrate BN running
    stats with 30 batches of forward passes in train mode (no grad).

Expected lift: +0.5 to +1.5 pp on Q15. SWA is essentially free at
inference time (same forward pass, just averaged weights).

Run from the repo root (~22-25 min):
    python experiments/exp_ablation_swa.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_swa",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(80, 160, 320, 640),
        blocks_per_stage=(3, 4, 6, 3),
        swa_start_epoch=21,  # last 10 epochs averaged
        swa_recalibrate_batches=30,
        baseline_test_pct=63.42,
        notes="Deep+wide (80,160,320,640)x(3,4,6,3) + SWA over epochs 21..30 + BN recalibration.",
    )
