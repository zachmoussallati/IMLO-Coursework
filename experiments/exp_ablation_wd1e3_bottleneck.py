"""Tweak: bottleneck blocks, ResNet-50 layout on the wd=1e-3 recipe.

Replaces every basic 3x3->3x3 PreActSEBlock with a bottleneck
1x1->3x3->1x1 BottleneckSEBlock (4x channel reduction in the middle).
Bottleneck blocks are ~3x cheaper per block, so the budget buys far
more depth at far wider channels.

  baseline: (64,128,256,512)    × (2,2,2,2) basic       11.3 M params, ~43 MB
  bottleneck: (256,512,1024,2048) × (3,4,6,3) bottleneck 26.1 M params, ~99 MB

This is essentially an SE-ResNet-50 architecture built from primitives
(no torchvision.models.resnet50 import — spec forbids that). It uses
the entire 100 MB zip headroom for capacity.

The bet vs the wider experiment: bottleneck-at-much-wider-channels
might generalise better than basic-at-1.5x-channels because the wider
representation has more room to disentangle features and the
4x-compressed middle acts as a built-in information bottleneck (mild
implicit regularisation).

Run from the repo root (~30-40 min):
    python experiments/exp_ablation_wd1e3_bottleneck.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_bottleneck",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(256, 512, 1024, 2048),
        blocks_per_stage=(3, 4, 6, 3),
        block_kind="bottleneck",
        baseline_test_pct=56.39,
        notes="wd=1e-3 + SE-ResNet-50 (bottleneck (256,512,1024,2048) x (3,4,6,3)), ~26M params, ~99MB model.pth.",
    )
