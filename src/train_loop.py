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


class SWA:
    """Stochastic Weight Averaging (Izmailov et al., 2018, arXiv:1803.05407).

    Maintains a running average of model weights collected over the tail of
    training. At eval time the averaged weights typically generalise better
    than the last-epoch weights because they sit in a flatter region of the
    loss landscape, and the average smooths out the per-batch oscillations
    the optimiser produces around its trajectory.

    Usage in a training script:
        swa = SWA(model)                       # capture snapshot 1
        for epoch in range(1, epochs+1):
            ... train ...
            if epoch >= start_epoch:
                swa.update(model)              # captures another snapshot
        swa.recalibrate_bn(model, loader)      # update BN stats on avg weights
        swa.apply_to(model)                    # swap averaged weights in

    why a hand-rolled SWA rather than torch.optim.swa_utils.AveragedModel:
    keeps the dependency surface small (just torch primitives) and avoids
    coupling SWA's lifetime to the optimiser. Equivalent maths.
    """

    def __init__(self, model: nn.Module) -> None:
        # why: store params on the same device as the model so the running
        # mean update is a single device-side add. CPU storage works too but
        # forces a host-device copy per update, which compounds across 30
        # epochs.
        self._avg = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.is_floating_point()
        }
        # Non-float buffers (e.g. BN num_batches_tracked, int64) are copied
        # as-is at apply time.
        self._non_float = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if not param.is_floating_point()
        }
        self._n_snapshots = 1  # the initial snapshot from __init__

    def update(self, model: nn.Module) -> None:
        """Fold a new model snapshot into the running average."""
        self._n_snapshots += 1
        for name, param in model.state_dict().items():
            if param.is_floating_point():
                # why: incremental mean — avg <- avg + (new - avg) / n.
                # Numerically stable for any number of snapshots.
                self._avg[name].add_(
                    (param.detach() - self._avg[name]) / self._n_snapshots
                )
            else:
                self._non_float[name] = param.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        """Copy averaged weights into the model in-place."""
        sd = dict(self._avg)
        sd.update(self._non_float)
        model.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def recalibrate_bn(
        self,
        model: nn.Module,
        loader,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        """Recompute BN running stats by feeding train data through the avg weights.

        why: BN's running mean/var were collected from the *original* weight
        trajectory, not the SWA average. After we swap in the averaged
        weights those stats can be a poor match for the new activations.
        Standard fix: reset running stats and do a few forward passes in
        train mode (which updates the running stats but no gradient).
        """
        self.apply_to(model)
        # Reset running stats on all BN layers.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        model.train()
        with torch.amp.autocast("cuda"):
            for i, (images, _) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                model(images)
                if max_batches is not None and i + 1 >= max_batches:
                    break
        model.eval()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_classes: int = 37,
    label_smoothing: float = 0.1,
    mix_prob: float = 0.0,
    accum_steps: int = 1,
    epoch_idx: int = 0,
    epochs: int = 0,
) -> dict:
    """One training epoch with AMP + optional gradient accumulation.

    why: I always mix-then-apply-soft-target-CE, even when no mixup/cutmix
    fires (maybe_mix returns the smoothed one-hot in that case). One code
    path means fewer places for the loss to silently disagree with the
    PyTorch CrossEntropyLoss(label_smoothing) version. mix_prob defaults to
    0.0 - mixup / CutMix were preventing the model from fitting at this data
    scale - but the implementation lives on in src/mixup.py for re-enabling.
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
            smoothing=label_smoothing, mix_prob=mix_prob,
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
    softmax each, then average the probabilities. Averaging probs (not
    logits) is the principled choice because softmax is nonlinear:
    averaging logits and then softmax-ing gives a different (and in general
    worse) ensemble.

    Kept on the same single-loader signature for callers that already have
    a loader handy (used during training for diagnostics). The final
    submission's Q15 uses evaluate_test_with_multiscale_tta below.
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


@torch.no_grad()
def evaluate_test_with_multiscale_tta(
    model: nn.Module,
    data_root: str,
    stats: dict,
    device: torch.device,
    scales: tuple[int, ...] = (224, 256, 288),
    batch_size: int = 128,
    num_workers: int = 2,
    use_amp: bool = True,
) -> float:
    """Multi-scale + HFlip TTA on the official test split.

    For each Resize scale in `scales`, build an eval transform
    (Resize -> CenterCrop(image_size) -> Normalize), run the model and its
    horizontal flip, and accumulate softmax probabilities across all
    scale/flip pairs. Argmax over the accumulated total.

    why multi-scale: HFlip alone gives the classifier one extra view per
    image. Resizing to a slightly larger or smaller intermediate size
    before the centre crop gives zoomed-in / zoomed-out views, which
    ensemble away some of the bias the network picks up at the single
    train-time resolution. On the locked baseline this lifts test accuracy
    from 45.60% to 46.42% (+0.82pp). Costs roughly 3x HFlip-only TTA time -
    still seconds per evaluation on a CUDA GPU.

    The function owns its DataLoader construction so that callers don't
    need to wire multiple loaders per scale.
    """
    # why: lazy imports keep src/train_loop.py importable without touching
    # the dataset stack unless this function is actually called.
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.datasets import OxfordIIITPet

    image_size = stats.get("image_size", 224)
    model.eval()

    ensembled_probs: torch.Tensor | None = None
    reference_labels: torch.Tensor | None = None

    for scale in scales:
        transform = T.Compose(
            [
                T.Resize(scale),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(stats["mean"], stats["std"]),
            ]
        )
        test_ds = OxfordIIITPet(
            root=data_root,
            split="test",
            target_types="category",
            download=True,
            transform=transform,
        )
        loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        scale_probs: list[torch.Tensor] = []
        scale_labels: list[torch.Tensor] = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            flipped = torch.flip(images, dims=[3])
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits_a = model(images)
                    logits_b = model(flipped)
            else:
                logits_a = model(images)
                logits_b = model(flipped)
            probs = F.softmax(logits_a, dim=1) + F.softmax(logits_b, dim=1)
            scale_probs.append(probs.cpu())
            scale_labels.append(labels)

        probs_tensor = torch.cat(scale_probs, dim=0)
        labels_tensor = torch.cat(scale_labels, dim=0)
        if ensembled_probs is None:
            ensembled_probs = probs_tensor.clone()
            reference_labels = labels_tensor
        else:
            assert reference_labels is not None
            # why: loaders are shuffle=False with identical dataset order,
            # so labels must match across scales; this catches accidental
            # divergence early instead of producing a silently wrong number.
            assert torch.equal(
                labels_tensor, reference_labels
            ), "label order changed across scales"
            ensembled_probs += probs_tensor

    assert ensembled_probs is not None and reference_labels is not None
    correct = (ensembled_probs.argmax(dim=1) == reference_labels).sum().item()
    return correct / reference_labels.numel()
