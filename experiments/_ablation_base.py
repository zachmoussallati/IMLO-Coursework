"""Shared helper for ablation experiments.

Each ablation experiment is "the locked recipe with ONE knob flipped".
`run_ablation` takes overrides for the knobs we vary and runs the full
30-epoch pipeline. Everything not in the overrides matches the live
recipe so any delta is attributable to the override.

Output: experiments/<name>/result.json and experiments/<name>/model.pth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

from src.data import (
    DATA_ROOT,
    NUM_CLASSES,
    build_eval_transform,
    build_train_transform,
    get_data_stats,
)
from src.model import build_model
from src.train_loop import (
    evaluate,
    evaluate_test_with_multiscale_tta,
    train_one_epoch,
)
from src.utils import make_worker_init, seed_all


# Locked-recipe defaults - any caller that doesn't pass an override gets
# bit-identical behaviour to the canonical run that landed 46.74%.
SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_LR_SGD = 1e-1
WEIGHT_DECAY_SGD = 5e-4
MOMENTUM = 0.9
MAX_LR_ADAMW = 4e-3
WEIGHT_DECAY_ADAMW = 1e-2
ONE_CYCLE_PCT_START = 0.17
MIX_PROB = 0.0


def run_ablation(
    name: str,
    *,
    use_maxpool: bool = False,
    optimizer_kind: Literal["sgd", "adamw"] = "sgd",
    use_full_trainval: bool = False,
    adamw_max_lr: float = MAX_LR_ADAMW,
    adamw_weight_decay: float = WEIGHT_DECAY_ADAMW,
    baseline_test_pct: float = 46.74,
    notes: str = "",
) -> dict:
    """Run one ablation; write experiments/<name>/result.json; return the result dict.

    why kwargs-only and no positional args: ablations should call out
    every override explicitly at the call site (`run_ablation('foo',
    use_maxpool=True)`) so reviewers can see at a glance which knob
    differs vs locked.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    experiment_dir = Path(__file__).resolve().parent / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Model init seed.
    seed_all(SEED)
    model = build_model(num_classes=NUM_CLASSES, use_maxpool=use_maxpool).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    # Re-seed for the training loop (matches the locked-recipe convention).
    generator = seed_all(SEED)
    worker_init = make_worker_init(SEED)

    stats = get_data_stats(seed=SEED)
    train_t = build_train_transform(stats["mean"], stats["std"])
    eval_t = build_eval_transform(stats["mean"], stats["std"])

    aug_trainval = OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        download=True, transform=train_t,
    )
    eval_trainval = OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        download=True, transform=eval_t,
    )

    if use_full_trainval:
        # why: alt-recipe-style — train on all 3680 trainval images, no
        # held-out val. Q14 is then "fit on the same data we trained on";
        # generalisation signal comes only from Q15 (test).
        train_subset = aug_trainval
        val_subset: Optional[torch.utils.data.Dataset] = None
        clean_train_subset = eval_trainval
        n_train = len(aug_trainval)
        n_val = 0
    else:
        train_subset = Subset(aug_trainval, stats["train_indices"])
        val_subset = Subset(eval_trainval, stats["val_indices"])
        clean_train_subset = Subset(eval_trainval, stats["train_indices"])
        n_train = len(train_subset)
        n_val = len(val_subset)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        generator=generator, worker_init_fn=worker_init,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    eval_kwargs = dict(
        num_workers=min(NUM_WORKERS, 2), pin_memory=True, persistent_workers=False,
    )
    clean_train_loader = DataLoader(
        clean_train_subset, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=worker_init, **eval_kwargs,
    )
    val_loader = (
        DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False,
            worker_init_fn=worker_init, **eval_kwargs,
        )
        if val_subset is not None
        else None
    )
    full_trainval_loader = DataLoader(
        eval_trainval, batch_size=BATCH_SIZE, shuffle=False,
        worker_init_fn=worker_init, **eval_kwargs,
    )

    if optimizer_kind == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=MAX_LR_SGD, momentum=MOMENTUM,
            nesterov=True, weight_decay=WEIGHT_DECAY_SGD,
        )
        sched_max_lr = MAX_LR_SGD
    elif optimizer_kind == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=adamw_max_lr / 25,
            weight_decay=adamw_weight_decay,
        )
        sched_max_lr = adamw_max_lr
    else:
        raise ValueError(f"unknown optimizer_kind={optimizer_kind!r}")

    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=sched_max_lr, total_steps=total_steps,
        pct_start=ONE_CYCLE_PCT_START, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda")

    print(
        f"[exp:{name}] params={param_count:,}  optimizer={optimizer_kind}  "
        f"max_lr={sched_max_lr}  train_size={n_train}  val_size={n_val}  "
        f"use_maxpool={use_maxpool}"
    )

    for epoch in range(1, EPOCHS + 1):
        m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device=device, num_classes=NUM_CLASSES,
            mix_prob=MIX_PROB, accum_steps=1,
            epoch_idx=epoch, epochs=EPOCHS,
        )
        ct = evaluate(model, clean_train_loader, device)
        if val_loader is not None:
            v = evaluate(model, val_loader, device)
            v_str = f"val={v['acc']*100:.2f}%"
        else:
            v_str = "val=n/a"
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[exp:{name}] epoch {epoch:02d}/{EPOCHS} | "
            f"lr={lr_now:.4f} | train_loss={m['loss']:.4f} | "
            f"clean_train={ct['acc']*100:.2f}% | {v_str}"
        )

    torch.save(model.state_dict(), experiment_dir / "model.pth")

    full_tv = evaluate(model, full_trainval_loader, device)
    test_acc = evaluate_test_with_multiscale_tta(
        model, data_root=DATA_ROOT, stats=stats, device=device,
        scales=(208, 224, 240, 256, 272, 288, 304),
        batch_size=BATCH_SIZE, num_workers=min(NUM_WORKERS, 2),
    )
    q14 = full_tv["acc"] * 100
    q15 = test_acc * 100

    delta = q15 - baseline_test_pct
    margin = 0.3
    if q15 > baseline_test_pct + margin:
        verdict = "BEATS BASELINE"
    elif q15 < baseline_test_pct - margin:
        verdict = "WORSE THAN BASELINE"
    else:
        verdict = "within noise"

    result = {
        "experiment": name,
        "knob_overrides": {
            "use_maxpool": use_maxpool,
            "optimizer_kind": optimizer_kind,
            "use_full_trainval": use_full_trainval,
        },
        "param_count": param_count,
        "n_train": n_train,
        "n_val": n_val,
        "q14_trainval_acc_pct": round(q14, 2),
        "q15_test_acc_pct_7scale_tta": round(q15, 2),
        "baseline_test_acc_pct": baseline_test_pct,
        "delta_pp": round(delta, 2),
        "verdict": verdict,
        "notes": notes,
    }
    (experiment_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"[exp:{name}] Q14 trainval (no TTA)            = {q14:.2f}%")
    print(f"[exp:{name}] Q15 test (7-scale + HFlip TTA)   = {q15:.2f}%")
    print(
        f"[exp:{name}] vs baseline {baseline_test_pct:.2f}% : "
        f"{'+' if delta >= 0 else ''}{delta:.2f}pp  ->  {verdict}"
    )
    return result
