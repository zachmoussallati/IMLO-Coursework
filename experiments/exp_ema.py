"""Experiment: keep an EMA (exponential moving average) copy of the weights
during training, then evaluate the EMA model at the end.

Why: EMA-averaged weights tend to fall in a flatter loss basin than the
last-step weights, which often gives a small but consistent test-accuracy
bump (typical +0.3-1.0 pp on image classifiers). No new hyperparameters
to tune beyond the decay; everything else stays identical to the locked
recipe so any delta is attributable to EMA alone.

Run (about 25 minutes on a single CUDA GPU):
    python experiments/exp_ema.py

Writes:
    experiments/ema/model.pth          (EMA model state dict)
    experiments/ema/model_raw.pth      (last-epoch raw weights, for comparison)
    experiments/ema/result.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


EXPERIMENT_NAME = "ema"
EXPERIMENT_DIR = Path(__file__).resolve().parent / EXPERIMENT_NAME
BASELINE_TEST_PCT = 46.42

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
MAX_LR = 1e-1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
ONE_CYCLE_PCT_START = 0.17
MIX_PROB = 0.0
NUM_WORKERS = 4

# why: classic image-classification EMA decay. With ~750 optimiser steps
# total, decay=0.999 has half-life ~693 steps, so the EMA "remembers"
# roughly the last half of training - exactly the cosine-decay phase where
# the model is approaching its final basin.
EMA_DECAY = 0.999


class EMAWrapper:
    """Maintains an EMA copy of a model's parameters.

    why: BN running statistics (the .running_mean / .running_var buffers)
    are not EMA-averaged - they're synced verbatim from the training model
    on each update, then re-estimated at the end via a fresh forward pass
    over the train subset (torch.optim.swa_utils.update_bn). EMA-averaging
    BN buffers as well usually gives slightly worse results because BN
    stats already do their own running average internally.
    """

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buf, buf in zip(self.module.buffers(), model.buffers()):
            ema_buf.copy_(buf)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    seed_all(SEED)
    model = build_model(num_classes=NUM_CLASSES).to(device)

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

    train_subset = Subset(aug_trainval, stats["train_indices"])
    val_subset = Subset(eval_trainval, stats["val_indices"])
    clean_train_subset = Subset(eval_trainval, stats["train_indices"])

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        generator=generator, worker_init_fn=worker_init,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    eval_kwargs = dict(
        num_workers=min(NUM_WORKERS, 2), pin_memory=True, persistent_workers=False,
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

    optimizer = optim.SGD(
        model.parameters(), lr=MAX_LR, momentum=MOMENTUM,
        nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=total_steps,
        pct_start=ONE_CYCLE_PCT_START, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda")

    ema = EMAWrapper(model, decay=EMA_DECAY)

    for epoch in range(1, EPOCHS + 1):
        # Train one epoch with manual stepping so I can call ema.update after
        # each optimiser step. Reusing train_one_epoch would skip the hook.
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            from src.mixup import maybe_mix, soft_target_cross_entropy
            mixed_images, soft_targets = maybe_mix(
                images, labels, num_classes=NUM_CLASSES,
                smoothing=0.1, mix_prob=MIX_PROB,
            )
            with torch.amp.autocast("cuda"):
                logits = model(mixed_images)
                loss = soft_target_cross_entropy(logits, soft_targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

        ct_raw = evaluate(model, clean_train_loader, device)
        ct_ema = evaluate(ema.module, clean_train_loader, device)
        v_raw = evaluate(model, val_loader, device)
        v_ema = evaluate(ema.module, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[exp:{EXPERIMENT_NAME}] epoch {epoch:02d}/{EPOCHS} | "
            f"lr={lr_now:.4f} | "
            f"raw ct/val={ct_raw['acc']*100:.2f}%/{v_raw['acc']*100:.2f}% | "
            f"ema ct/val={ct_ema['acc']*100:.2f}%/{v_ema['acc']*100:.2f}%"
        )

    # why: BN running stats of the EMA model lag the training model. Reset
    # them by running a forward pass through the clean train loader in
    # train mode - this is what torch.optim.swa_utils.update_bn does, I
    # call it explicitly so I can use my own loader.
    torch.optim.swa_utils.update_bn(clean_train_loader, ema.module, device=device)

    torch.save(ema.module.state_dict(), EXPERIMENT_DIR / "model.pth")
    torch.save(model.state_dict(), EXPERIMENT_DIR / "model_raw.pth")

    raw_tv = evaluate(model, full_trainval_loader, device)
    ema_tv = evaluate(ema.module, full_trainval_loader, device)

    raw_test = evaluate_test_with_multiscale_tta(
        model, data_root=DATA_ROOT, stats=stats, device=device,
        scales=(224, 256, 288), batch_size=BATCH_SIZE,
        num_workers=min(NUM_WORKERS, 2),
    )
    ema_test = evaluate_test_with_multiscale_tta(
        ema.module, data_root=DATA_ROOT, stats=stats, device=device,
        scales=(224, 256, 288), batch_size=BATCH_SIZE,
        num_workers=min(NUM_WORKERS, 2),
    )

    raw_test_pct = raw_test * 100
    ema_test_pct = ema_test * 100
    raw_q14 = raw_tv["acc"] * 100
    ema_q14 = ema_tv["acc"] * 100

    delta = ema_test_pct - BASELINE_TEST_PCT
    margin = 0.3
    if ema_test_pct > BASELINE_TEST_PCT + margin:
        verdict = "BEATS BASELINE"
    elif ema_test_pct < BASELINE_TEST_PCT - margin:
        verdict = "WORSE THAN BASELINE"
    else:
        verdict = "within noise"

    result = {
        "experiment": EXPERIMENT_NAME,
        "decay": EMA_DECAY,
        "raw_q14_trainval_acc_pct": round(raw_q14, 2),
        "ema_q14_trainval_acc_pct": round(ema_q14, 2),
        "raw_q15_test_acc_pct": round(raw_test_pct, 2),
        "ema_q15_test_acc_pct": round(ema_test_pct, 2),
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "delta_pp": round(delta, 2),
        "verdict": verdict,
    }
    (EXPERIMENT_DIR / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"[exp:{EXPERIMENT_NAME}] raw     Q14 = {raw_q14:.2f}%  Q15 = {raw_test_pct:.2f}%")
    print(f"[exp:{EXPERIMENT_NAME}] EMA     Q14 = {ema_q14:.2f}%  Q15 = {ema_test_pct:.2f}%")
    print(f"[exp:{EXPERIMENT_NAME}] vs baseline {BASELINE_TEST_PCT:.2f}% : "
          f"{'+' if delta >= 0 else ''}{delta:.2f}pp  ->  {verdict}")


if __name__ == "__main__":
    main()
