"""Tweak: AdamW with weight_decay=1e-3 (matches the SGD winner's wd).

Every prior AdamW experiment used wd=1e-2 (AdamW's typical default) and
lost decisively against SGD. The wd=1e-3 axis won big for SGD
(56.39% > 54.13%), so the natural ablation is: does AdamW also benefit
from wd=1e-3, or was the issue specifically with AdamW's step rule?

If AdamW + wd=1e-3 still loses, the conclusion is "SGD is structurally
better for this CNN" rather than "wd was the issue all along".

Run from the repo root (~15-20 min):
    python experiments/exp_ablation_adamw_wd1e3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_adamw_wd1e3",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="adamw",
        adamw_max_lr=1e-3,
        adamw_weight_decay=1e-3,
        baseline_test_pct=56.39,
        notes="AdamW with max_lr=1e-3, wd=1e-3 (matching SGD's wd winner).",
    )
