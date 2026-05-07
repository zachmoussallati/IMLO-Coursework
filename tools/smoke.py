"""End-to-end smoke test for the training pipeline.

Trains for one epoch on the first 200 images of the train split with all
the same components train.py uses (mixup, AMP, OneCycle, soft-target CE),
then exercises the save/reload + TTA evaluation paths so I know the full
pipeline is wired correctly before committing to a 30-epoch run.

Run from the repo root:
    python tools/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

# Repo root on path so `from src.X import Y` works when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import NUM_CLASSES, build_loaders
from src.model import build_model
from src.train_loop import evaluate, evaluate_with_tta, train_one_epoch
from src.utils import make_worker_init, seed_all


SUBSET_SIZE = 200
SMOKE_BATCH = 64
SMOKE_LR = 5e-2


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA.")
    device = torch.device("cuda")

    # First seed: model init.
    seed_all(42)
    model = build_model(num_classes=NUM_CLASSES).to(device)

    # Re-seed before building loaders so worker / shuffle order is locked.
    generator = seed_all(42)
    worker_init_fn = make_worker_init(42)
    loaders = build_loaders(
        batch_size=SMOKE_BATCH,
        num_workers=2,
        seed=42,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    train_subset = loaders["train"].dataset  # itself a Subset
    tiny_subset = Subset(train_subset, list(range(SUBSET_SIZE)))
    tiny_loader = DataLoader(
        tiny_subset,
        batch_size=SMOKE_BATCH,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    assert len(tiny_loader) > 0, "tiny train loader is empty"
    print(f"[smoke] tiny train batches: {len(tiny_loader)}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=SMOKE_LR,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=SMOKE_LR, total_steps=len(tiny_loader),
    )
    scaler = torch.amp.GradScaler("cuda")

    # 1 mini "epoch"
    metrics = train_one_epoch(
        model, tiny_loader, optimizer, scheduler, scaler,
        device=device, epoch_idx=1, epochs=1,
    )
    print(f"[smoke] train loss: {metrics['loss']:.4f}")
    assert torch.isfinite(torch.tensor(metrics["loss"])), "loss is not finite"

    val_metrics = evaluate(model, loaders["val"], device)
    print(f"[smoke] val acc: {val_metrics['acc'] * 100:.2f}%")
    assert 0.0 <= val_metrics["acc"] <= 1.0, "val acc out of [0, 1]"

    test_acc = evaluate_with_tta(model, loaders["test"], device)
    print(f"[smoke] test acc (+TTA): {test_acc * 100:.2f}%")
    assert 0.0 <= test_acc <= 1.0, "test acc out of [0, 1]"

    torch.save(model.state_dict(), "model.pth")
    saved_size_mb = Path("model.pth").stat().st_size / 1e6
    print(f"[smoke] saved model.pth ({saved_size_mb:.1f} MB)")

    fresh = build_model(num_classes=NUM_CLASSES).to(device)
    state = torch.load("model.pth", map_location=device, weights_only=True)
    fresh.load_state_dict(state)
    fresh.eval()
    test_acc_reloaded = evaluate_with_tta(fresh, loaders["test"], device)
    print(f"[smoke] test acc after reload: {test_acc_reloaded * 100:.2f}%")
    assert abs(test_acc - test_acc_reloaded) < 1e-4, "save/reload mismatched"

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
