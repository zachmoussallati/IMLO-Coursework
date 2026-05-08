"""3-epoch sanity run to confirm the fixed recipe is actually learning.

I made a stripping-back fix (removed mixup/cutmix, dropped ColorJitter and
RandomErasing, softened RandAugment, removed zero-init-final-BN). This
script exercises the *real* loaders / model / optimiser / scheduler for 3
epochs on the full train split and reports clean train + val accuracy each
epoch. If clean-train climbs noticeably above 2.7% (random) the recipe
works; if it stays flat I have a deeper bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import NUM_CLASSES, build_loaders
from src.model import build_model
from src.train_loop import evaluate, train_one_epoch
from src.utils import make_worker_init, seed_all

QUICK_EPOCHS = 3
BATCH_SIZE = 128


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("verify_fix requires CUDA.")
    device = torch.device("cuda")

    seed_all(42)
    model = build_model(num_classes=NUM_CLASSES).to(device)

    generator = seed_all(42)
    worker_init = make_worker_init(42)
    loaders = build_loaders(
        batch_size=BATCH_SIZE, num_workers=4, seed=42,
        generator=generator, worker_init_fn=worker_init,
    )

    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                    nesterov=True, weight_decay=5e-4)
    steps = len(loaders["train"]) * QUICK_EPOCHS
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.1, total_steps=steps, pct_start=0.17, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, QUICK_EPOCHS + 1):
        m = train_one_epoch(
            model, loaders["train"], opt, sched, scaler,
            device=device, num_classes=NUM_CLASSES, mix_prob=0.0,
            accum_steps=1, epoch_idx=epoch, epochs=QUICK_EPOCHS,
        )
        ct = evaluate(model, loaders["clean_train"], device)
        v = evaluate(model, loaders["val"], device)
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[verify] epoch {epoch}/{QUICK_EPOCHS} | "
            f"lr={lr_now:.4f} | train_loss={m['loss']:.4f} | "
            f"clean_train={ct['acc']*100:.2f}% | val={v['acc']*100:.2f}%"
        )

    # Snapshot: what does bn_b look like after 3 epochs?
    print("[verify] bn_b weights after 3 epochs (mean / std per block):")
    for name, p in model.named_parameters():
        if "bn_b.weight" in name:
            print(f"  {name}  mean={float(p.mean()):.4f}  std={float(p.std()):.4f}")


if __name__ == "__main__":
    main()
