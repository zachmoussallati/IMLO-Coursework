"""Tweak: BlurPool antialiased downsampling on the ResNet-101 recipe.

Reference: Zhang 2019, "Making Convolutional Networks Shift-Invariant
Again", arXiv:1904.11486. Replaces each stride-2 conv with stride-1
conv + 3x3 binomial blur + stride-2 subsample, so downsampling is
properly low-passed.

The earlier (3,4,6,3) BlurPool experiment hit the chain runner's
45-min hard cap before TTA could run — BlurPool adds ~3 extra
depthwise-conv passes per epoch which roughly doubles per-epoch time.
At the ResNet-101 (3,4,23,3) depth this is even slower. This retry
uses no hard cap — it'll run as long as needed.

  baseline (3,4,23,3): plain stride-2 convs in stage 2/3/4 transitions
  blurpool variant: stride-1 conv + BlurPool3x3 stride 2 at those transitions

Zero parameter cost; small per-block FLOP increase.

Run from the repo root (~60-90 min - slower than non-blurpool variants):
    python experiments/exp_blurpool_resnet101.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="blurpool_resnet101",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
        baseline_test_pct=67.54,
        notes="ResNet-101 (3,4,23,3) + BlurPool antialiased stride-2 transitions.",
    )
