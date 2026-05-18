"""Tweak: anti-aliased downsampling (BlurPool) on the deeper recipe.

Reference: Zhang 2019, "Making Convolutional Networks Shift-Invariant
Again", arXiv:1904.11486. A stride-2 conv samples every other pixel
without first low-passing the signal, so a 1-pixel shift in the input
can land on a different sample grid and flip the output. BlurPool
inserts a separable binomial blur kernel before the subsample so
downsampling is properly antialiased.

  baseline conv at stride=2:    Conv3x3(stride=2)
  blurpool variant:             Conv3x3(stride=1) -> BlurPool3x3(stride=2)

The blur is a fixed (non-trainable) 3x3 binomial kernel
[1, 2, 1]^T * [1, 2, 1] / 16, applied per-channel via grouped conv.
Zero parameters added; small per-block FLOP increase.

Bet: each stride-2 transition in stages 2/3/4 currently aliases; with
(3,4,6,3) blocks there are exactly 3 such transitions. Fixing them
should let the optimiser settle into a more shift-equivariant filter
bank, which generally helps on small/medium datasets.

Run from the repo root (~22-25 min):
    python experiments/exp_ablation_blurpool.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_blurpool",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 6, 3),
        use_blurpool=True,
        baseline_test_pct=62.09,
        notes="Deeper (3,4,6,3) + BlurPool antialiased stride-2 transitions.",
    )
