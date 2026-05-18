"""Capacity sweep: (88,176,352,704) widths x (3,4,6,3) depth.

Gentle bump on the locked (80,160,320,640) x (3,4,6,3) recipe. ~40.5M
params, ~77MB fp16, ~79MB zip — uses ~20MB more zip headroom.

Run from the repo root (~22-28 min):
    python experiments/exp_cap_88_346.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="cap_88_346",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(88, 176, 352, 704),
        blocks_per_stage=(3, 4, 6, 3),
        baseline_test_pct=63.42,
        notes="Capacity sweep: (88,176,352,704)x(3,4,6,3), ~40.5M params.",
    )
