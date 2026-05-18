"""Tweak: RandomErasing p=0.10 on the BlurPool + ResNet-101 recipe.

Same target as exp_blurpool_mixup10 — close the train/test gap that
the locked recipe shows (clean_train=90.84%, Q15=70.26%). MixUp blends
two images; RandomErasing zeros out a random rectangle of one image.
They're orthogonal regularisers — MixUp adds noise at the *label*
level (soft targets), RandomErasing adds noise at the *input* level.

p=0.10 means each batch sample has 10% chance of having a random
rectangle erased (after normalisation). RandomErasing(p=0.25) at the
shallow model lost -2.89pp; p=0.10 is more cautious.

Run from the repo root (~30-35 min):
    python experiments/exp_blurpool_re10.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="blurpool_re10",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 512),
        blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
        random_erasing_p=0.10,
        baseline_test_pct=70.26,
        notes="BlurPool + ResNet-101 + RandomErasing p=0.10 light regularisation.",
    )
