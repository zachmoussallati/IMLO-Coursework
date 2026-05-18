"""Capacity sweep: (88,176,352,704) widths x (3,4,9,3) depth.

Combined width bump + depth bump on top of the locked recipe. ~47.3M
params, ~90MB fp16, ~92MB zip — high-end of the budget. Tests whether
both axes compound or one cannibalises the other.

Run from the repo root (~28-35 min):
    python experiments/exp_cap_88_349.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="cap_88_349",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(88, 176, 352, 704),
        blocks_per_stage=(3, 4, 9, 3),
        baseline_test_pct=63.42,
        notes="Capacity sweep: (88,176,352,704)x(3,4,9,3) both axes, ~47.3M params.",
    )
