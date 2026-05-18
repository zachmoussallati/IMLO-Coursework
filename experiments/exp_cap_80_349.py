"""Capacity sweep: (80,160,320,640) widths x (3,4,9,3) depth.

Holds the current widths, adds 3 more blocks in stage 3. ~39.1M params,
~75MB fp16, ~77MB zip — uses ~12MB more headroom. Tests whether more
stage-3 depth (at 320 channels, 14x14 features) helps the same way the
original (3,4,6,3) bump did.

Run from the repo root (~25-30 min):
    python experiments/exp_cap_80_349.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="cap_80_349",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(80, 160, 320, 640),
        blocks_per_stage=(3, 4, 9, 3),
        baseline_test_pct=63.42,
        notes="Capacity sweep: (80,160,320,640)x(3,4,9,3) depth-only, ~39.1M params.",
    )
