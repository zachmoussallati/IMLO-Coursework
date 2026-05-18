"""Tweak: widen only stage 4 on the BlurPool + ResNet-101 recipe.

Earlier width sweeps showed that uniformly widening every stage gave
noise-level gains (cap_88_346 +0.09pp). But spending the width budget
*only on stage 4* — the deepest, most abstract feature map — is a
different hypothesis. Stage 4 features are pooled directly into the
37-way head; more channels there gives the classifier richer features
to discriminate on, without adding capacity to lower stages where the
model is already over-fitting.

  baseline (BlurPool + RN-101): (64, 128, 256, 512)   41.7M params
  stage-4 widened:             (64, 128, 256, 640)   48.7M params, ~93MB fp16

If it wins, the zip goes from 77MB to ~95MB (still under the 100MB cap).

Run from the repo root (~35-45 min):
    python experiments/exp_blurpool_stage4_wider.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="blurpool_stage4_wider",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(64, 128, 256, 640),
        blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
        baseline_test_pct=70.26,
        notes="BlurPool + ResNet-101 + stage-4 widened to 640 (was 512), ~48.7M params.",
    )
