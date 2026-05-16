"""Ablation: AdamW on top of the *promoted* recipe (MaxPool + full trainval).

The previous AdamW ablation (`exp_ablation_adamw.py`) was tested on the
old locked baseline (no MaxPool, 3 312 / 368 split) and lost by 10.46 pp.
Now that the recipe has changed substantially - MaxPool stem and training
on all 3 680 images, Q15 = 54.13 % - it's worth re-testing whether AdamW
is still worse, or whether the new architecture / data setup makes it
viable.

Locked-everything-else, swap only the optimiser:
  SGD-Nesterov(lr=0.1, wd=5e-4) -> AdamW(lr=4e-3, wd=1e-2)

Run from the repo root (about 15-20 min - MaxPool makes it cheap):
    python experiments/exp_ablation_adamw_on_promoted.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_adamw_on_promoted",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="adamw",
        adamw_max_lr=4e-3,
        adamw_weight_decay=1e-2,
        baseline_test_pct=54.13,
        notes=(
            "AdamW(lr=4e-3, wd=1e-2) layered on the promoted recipe "
            "(MaxPool + full trainval). Re-testing whether AdamW is still "
            "worse than SGD-Nesterov on the new architecture / data setup."
        ),
    )
