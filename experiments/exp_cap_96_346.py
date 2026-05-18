"""Capacity sweep: (96,192,384,768) widths x (3,4,6,3) depth (max width).

Pushes the channel ladder to 1.5x baseline at the deeper layout. ~48.2M
params, ~92MB fp16, ~94MB zip — uses essentially all the zip headroom
(target buffer 2-3MB).

Run from the repo root (~25-30 min):
    python experiments/exp_cap_96_346.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="cap_96_346",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(96, 192, 384, 768),
        blocks_per_stage=(3, 4, 6, 3),
        baseline_test_pct=63.42,
        notes="Capacity sweep: (96,192,384,768)x(3,4,6,3) max width, ~48.2M params.",
    )
