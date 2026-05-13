"""Experiment: smaller PetClassifier (widths 32/64/128/256) trained from scratch.

The locked baseline uses widths (64, 128, 256, 512) - 11.3 M parameters.
On 3 312 training images that's heavily over-parameterized, and the gap
between Q14 (60.84 %) and Q15 (45.60 %) of ~15 pp at the locked recipe
suggests the model is memorizing more than it generalises. A narrower
variant (~2.8 M params) might generalise better at the cost of fitting
capacity. This experiment tries that hypothesis with everything else
identical to the locked recipe.

Run (about 25 minutes on a single CUDA GPU):
    python experiments/exp_smaller_model.py

Writes:
    experiments/smaller_model/model.pth
    experiments/smaller_model/result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import optim
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


EXPERIMENT_NAME = "smaller_model"
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

# why: half the channels at every stage -> ~4x fewer params. The pattern
# still doubles channels at each stage, so the capacity ratio between
# stages is preserved.
WIDTHS = (32, 64, 128, 256)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    seed_all(SEED)
    model = build_model(num_classes=NUM_CLASSES, widths=WIDTHS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[exp:{EXPERIMENT_NAME}] widths={WIDTHS}  params={param_count:,}")

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
        num_workers=min(NUM_WORKERS, 2), pin_memory=True,
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
    test_acc = evaluate_test_with_multiscale_tta(
        model, data_root=DATA_ROOT, stats=stats, device=device,
        scales=(224, 256, 288), batch_size=BATCH_SIZE,
        num_workers=min(NUM_WORKERS, 2),
    )
    q14 = full_tv["acc"] * 100
    q15 = test_acc * 100

    delta = q15 - BASELINE_TEST_PCT
    margin = 0.3
    if q15 > BASELINE_TEST_PCT + margin:
        verdict = "BEATS BASELINE"
    elif q15 < BASELINE_TEST_PCT - margin:
        verdict = "WORSE THAN BASELINE"
    else:
        verdict = "within noise"

    result = {
        "experiment": EXPERIMENT_NAME,
        "widths": list(WIDTHS),
        "param_count": param_count,
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
    print(f"[exp:{EXPERIMENT_NAME}] Q15 test (3-scale TTA) = {q15:.2f}%")
    print(
        f"[exp:{EXPERIMENT_NAME}] vs baseline {BASELINE_TEST_PCT:.2f}% : "
        f"{'+' if delta >= 0 else ''}{delta:.2f}pp  ->  {verdict}"
    )


if __name__ == "__main__":
    main()
