"""Tweak: MixUp p=0.10 on the BlurPool + ResNet-101 recipe.

The locked recipe has clean_train=90.84% but Q15=70.26% — a 20.58pp
generalisation gap. That's the textbook overfit signature: model has
fully memorised the 3 680 train images but its representations aren't
quite the right ones for the test distribution. A light dose of
data-level regularisation might close some of that gap.

MixUp p=0.10 means: for each batch, with 10% probability, replace the
batch with a (mixup, cutmix 50/50) blend. Beta(0.2, 0.2) for mixup and
Beta(1.0, 1.0) for cutmix, soft-target cross-entropy. p=0.25 was tried
at the shallow (3,4,6,3) layout and lost (-0.98pp), but with 33 blocks
and BlurPool the model has 4x more capacity and overfit is much more
pronounced — a smaller dose might land in the helpful regime.

Run from the repo root (~30-35 min):
    python experiments/exp_blurpool_mixup10.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="blurpool_mixup10",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
        mix_prob=0.10,
        baseline_test_pct=70.26,
        notes="BlurPool + ResNet-101 + MixUp p=0.10 light regularisation.",
    )
