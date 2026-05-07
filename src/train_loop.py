"""Per-epoch training and evaluation routines.

I keep these here (rather than inlined in train.py) so that test.py can call
the same evaluate_with_tta path. That's important for the Q15 number: the
test accuracy I report from train.py must match what test.py prints, since
test.py is what the markers actually run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from .mixup import maybe_mix, soft_target_cross_entropy
from .utils import AverageMeter


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_classes: int = 37,
    label_smoothing: float = 0.1,
    accum_steps: int = 1,
    epoch_idx: int = 0,
    epochs: int = 0,
) -> dict:
    """One training epoch with AMP + optional gradient accumulation.

    why: I always mix-then-apply-soft-target-CE, even when no mixup/cutmix
    fires (maybe_mix returns the smoothed one-hot in that case). One code
    path means fewer places for the loss to silently disagree with the
    PyTorch CrossEntropyLoss(label_smoothing) version.
    """
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"epoch {epoch_idx}/{epochs}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed_images, soft_targets = maybe_mix(
            images, labels, num_classes=num_classes,
            smoothing=label_smoothing,
        )

        with torch.amp.autocast("cuda"):
            logits = model(mixed_images)
            # why: divide by accum_steps so the *summed* gradient over an
            # accumulation window has the same scale as a single step at the
            # full effective batch.
            loss = soft_target_cross_entropy(logits, soft_targets) / accum_steps

        scaler.scale(loss).backward()
        loss_meter.update(loss.item() * accum_steps, n=images.size(0))

        # why: optimizer + scheduler step only on accumulation boundaries so
        # the effective batch is bs * accum_steps regardless of the micro-batch.
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.4f}",
            )

    return {"loss": loss_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Top-1 accuracy + un-smoothed cross-entropy loss on a deterministic loader."""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    ce_loss = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
        else:
            logits = model(images)
        loss = ce_loss(logits, labels)
        loss_meter.update(loss.item(), n=images.size(0))
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return {
        "loss": loss_meter.avg,
        "acc": correct / total if total else 0.0,
    }


@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Top-1 accuracy with horizontal-flip test-time augmentation.

    why: I run two forwards (the original image and its horizontal flip),
    softmax each, then average the probabilities. This is "free" extra
    information from the same test image - no external data, just a
    different view of the same input - which is the standard TTA setup.
    Averaging probs (not logits) is the principled choice because softmax is
    nonlinear: averaging logits and then softmax-ing gives a different (and
    in general worse) ensemble.
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        flipped = torch.flip(images, dims=[3])
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits_a = model(images)
                logits_b = model(flipped)
        else:
            logits_a = model(images)
            logits_b = model(flipped)
        probs = (F.softmax(logits_a, dim=1) + F.softmax(logits_b, dim=1)) * 0.5
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0
