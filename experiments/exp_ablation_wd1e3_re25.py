"""Tweak: wd=1e-3 + RandomErasing(p=0.25).

wd=1e-3 alone won +2.26pp (54.13 -> 56.39). Mixup-on-top hurt. wd=2e-3
alone went too far. Trying RandomErasing as the orthogonal regulariser:
it operates on normalized tensors (vs wd which constrains weights), so
the two shouldn't fight each other the way mixup did.

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_wd1e3_re25.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_re25",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        random_erasing_p=0.25,
        baseline_test_pct=56.39,
        notes="Stacking wd=1e-3 winner with RandomErasing(p=0.25).",
    )
