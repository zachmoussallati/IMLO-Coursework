"""Tweak: deeper network (3,4,6,3 blocks/stage) on the wd=1e-3 recipe.

Same channel widths (64,128,256,512) as the locked baseline, but more
blocks per stage in a ResNet-50-style distribution:

  baseline: (2,2,2,2) blocks per stage   8 blocks total,  11.3 M params, ~43 MB
  deeper:   (3,4,6,3) blocks per stage  16 blocks total,  21.5 M params, ~82 MB

The ResNet (3,4,6,3) pattern concentrates depth in stage 3 (256-ch),
which is the cheapest place to add blocks — every extra block there
costs ~2x the params of a stage-1 block but ~0.25x of a stage-4 block.

This is a different bet than the wider experiment: instead of more
channels (more "what to learn per pixel"), more depth gives the
network more nonlinear transforms ("how complex a feature it can
build"). At 30 epochs on 3680 images the optimisation budget is tight
either way, so the test is whether depth or width is the cheaper way
to push past 56.39%.

Run from the repo root (~25-30 min):
    python experiments/exp_ablation_wd1e3_deeper.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_deeper",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        blocks_per_stage=(3, 4, 6, 3),
        baseline_test_pct=56.39,
        notes="wd=1e-3 + deeper (3,4,6,3) ResNet-50 stage layout, ~21.5M params, ~82MB model.pth.",
    )
