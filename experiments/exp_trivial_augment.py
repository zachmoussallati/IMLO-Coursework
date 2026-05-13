"""Experiment: swap RandAugment for TrivialAugmentWide and retrain.

TrivialAugmentWide (Mueller & Hutter, 2021, arXiv:2103.10158) samples one
op + one magnitude per image. It needs no per-dataset tuning and reportedly
outperforms RandAugment on a number of small datasets. This experiment
keeps every other knob identical to the locked recipe and only changes the
augmentation, so the delta is attributable to the augmentation alone.

Run from the repo root (about 25 minutes on a single CUDA GPU):
    python experiments/exp_trivial_augment.py

Writes:
    experiments/trivial_augment/model.pth
    experiments/trivial_augment/result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    DATA_ROOT,
    IMAGE_SIZE,
    NUM_CLASSES,
    build_eval_transform,
    get_data_stats,
)
from src.model import build_model
from src.train_loop import evaluate, evaluate_with_tta, train_one_epoch
from src.utils import make_worker_init, seed_all


EXPERIMENT_NAME = "trivial_augment"
EXPERIMENT_DIR = Path(__file__).resolve().parent / EXPERIMENT_NAME
BASELINE_TEST_PCT = 45.60

# Match the locked recipe exactly except for the augmentation function.
SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
MAX_LR = 1e-1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
ONE_CYCLE_PCT_START = 0.17
MIX_PROB = 0.0
NUM_WORKERS = 4


def build_trivial_augment_transform(mean, std) -> T.Compose:
    """RandomResizedCrop + HFlip + TrivialAugmentWide + Normalize."""
    return T.Compose(
        [
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            # why: TrivialAugmentWide samples (op, magnitude) uniformly per
            # image with no schedule / no num_ops to tune. Same operator set
            # as RandAugment, simpler controller. Default magnitude bins=31.
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment.")
    device = torch.device("cuda")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Same seeding ritual as train.py so the only differing source of
    # randomness between this and the locked recipe is the augmentation.
    seed_all(SEED)
    model = build_model(num_classes=NUM_CLASSES).to(device)

    generator = seed_all(SEED)
    worker_init = make_worker_init(SEED)

    stats = get_data_stats(seed=SEED)
    train_t = build_trivial_augment_transform(stats["mean"], stats["std"])
    eval_t = build_eval_transform(stats["mean"], stats["std"])

    aug_trainval = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="category",
        download=True,
        transform=train_t,
    )
    eval_trainval = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="category",
        download=True,
        transform=eval_t,
    )
    test_ds = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True,
        transform=eval_t,
    )

    train_subset = Subset(aug_trainval, stats["train_indices"])
    val_subset = Subset(eval_trainval, stats["val_indices"])
    clean_train_subset = Subset(eval_trainval, stats["train_indices"])

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    eval_kwargs = dict(
        num_workers=min(NUM_WORKERS, 2),
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=worker_init, **eval_kwargs,
    )
    clean_train_loader = DataLoader(
        clean_train_subset, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=worker_init, **eval_kwargs,
    )
    full_trainval_loader = DataLoader(
        eval_trainval, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=worker_init, **eval_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=worker_init, **eval_kwargs,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=MAX_LR,
        momentum=MOMENTUM,
        nesterov=True,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=total_steps,
        pct_start=ONE_CYCLE_PCT_START,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, EPOCHS + 1):
        m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device=device, num_classes=NUM_CLASSES,
            mix_prob=MIX_PROB, accum_steps=1,
            epoch_idx=epoch, epochs=EPOCHS,
        )
        ct = evaluate(model, clean_train_loader, device)
        v = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[exp:{EXPERIMENT_NAME}] epoch {epoch:02d}/{EPOCHS} | "
            f"lr={lr_now:.4f} | train_loss={m['loss']:.4f} | "
            f"clean_train={ct['acc']*100:.2f}% | val={v['acc']*100:.2f}%"
        )

    torch.save(model.state_dict(), EXPERIMENT_DIR / "model.pth")

    full_tv = evaluate(model, full_trainval_loader, device)
    test_acc = evaluate_with_tta(model, test_loader, device)
    q14 = full_tv["acc"] * 100
    q15 = test_acc * 100

    delta = q15 - BASELINE_TEST_PCT
    margin = 0.5
    if q15 > BASELINE_TEST_PCT + margin:
        verdict = "BEATS BASELINE"
    elif q15 < BASELINE_TEST_PCT - margin:
        verdict = "WORSE THAN BASELINE"
    else:
        verdict = "within noise"

    result = {
        "experiment": EXPERIMENT_NAME,
        "q14_trainval_acc_pct": round(q14, 2),
        "q15_test_acc_pct": round(q15, 2),
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "delta_pp": round(delta, 2),
        "verdict": verdict,
    }
    (EXPERIMENT_DIR / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"[exp:{EXPERIMENT_NAME}] Q14 trainval = {q14:.2f}%")
    print(f"[exp:{EXPERIMENT_NAME}] Q15 test (+HFlip TTA) = {q15:.2f}%")
    print(
        f"[exp:{EXPERIMENT_NAME}] vs baseline {BASELINE_TEST_PCT:.2f}% : "
        f"{'+' if delta >= 0 else ''}{delta:.2f}pp  ->  {verdict}"
    )


if __name__ == "__main__":
    main()
