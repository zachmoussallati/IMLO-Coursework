"""Ablation tweak: AdamW with a lower max_lr (1e-3) on top of the promoted recipe.

`exp_ablation_adamw_on_promoted.py` tried AdamW(lr=4e-3, wd=1e-2) and lost
by 9.51 pp. This run drops the peak LR by 4x to the canonical
small-dataset operating point (1e-3) - in case 4e-3 was just too hot for
AdamW on this short 30-epoch schedule.

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_adamw_tune.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_adamw_tune_1e3",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="adamw",
        adamw_max_lr=1e-3,
        adamw_weight_decay=1e-2,
        baseline_test_pct=54.13,
        notes=(
            "AdamW with a tuned lower max_lr=1e-3 (vs 4e-3 in the prior "
            "AdamW-on-promoted run). Same MaxPool + full-trainval setup."
        ),
    )
