"""Tweak: wider channels (96, 192, 384, 768) on the wd=1e-3 recipe.

The wd=1e-3 recipe is at a local optimum for every single-knob flip
I've tried so far. Structural changes are the remaining axis. This
bumps every stage's channel count by 1.5x:

  baseline: (64, 128, 256, 512)   ~11.3 M params, ~43 MB model.pth
  wider:    (96, 192, 384, 768)   ~25.3 M params, ~97 MB model.pth

That sits right against the 100 MB zip cap (model.pth + the rest of
the zip ~ 99 MB). The downside is overfit risk — 2.25x more parameters
on the same 3680 training images at 30 epochs. That's exactly the
regime wd=1e-3 was tuned for, so the bet is that wider channels at
high wd extract more signal than the baseline channel count can at
the same wd.

If this run beats 56.39% by > 0.3pp it gets promoted and the live
model.pth grows to ~97 MB; if it loses or breaks even, the locked
recipe stays at ~43 MB.

Run from the repo root (~20-30 min):
    python experiments/exp_ablation_wd1e3_wider.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._ablation_base import run_ablation


if __name__ == "__main__":
    run_ablation(
        name="ablation_wd1e3_wider",
        use_maxpool=True,
        use_full_trainval=True,
        optimizer_kind="sgd",
        sgd_max_lr=1e-1,
        sgd_weight_decay=1e-3,
        widths=(96, 192, 384, 768),
        baseline_test_pct=56.39,
        notes="wd=1e-3 + wider channels (96,192,384,768), ~25.3M params, ~97MB model.pth.",
    )
