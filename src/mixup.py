"""Mixup, CutMix, and a soft-target cross-entropy loss.

Both regularizers operate on a mini-batch *after* it leaves the DataLoader,
just before the forward pass. I produce a single (soft) target tensor either
way, so the training loop can use one loss path regardless of whether a
batch was mixed or not.

References (paper-only, my own implementation):
  * Zhang et al., 2018 - "mixup: Beyond Empirical Risk Minimization",
    arXiv:1710.09412.
  * Yun et al., 2019 - "CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features", arXiv:1905.04899.
  * Szegedy et al., 2016 - "Rethinking the Inception Architecture for
    Computer Vision" (label smoothing), arXiv:1512.00567.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def smoothed_one_hot(
    labels: torch.Tensor, num_classes: int, smoothing: float
) -> torch.Tensor:
    """Convert integer labels to label-smoothed one-hot targets.

    why: I match PyTorch's CrossEntropyLoss(label_smoothing=eps) convention -
    every class gets eps / K as a floor, and the true class gets the extra
    (1 - eps) on top. This keeps my soft-target loss numerically identical
    to nn.CrossEntropyLoss when no mixing is applied.
    """
    confidence = 1.0 - smoothing
    soft = torch.full(
        (labels.size(0), num_classes),
        smoothing / num_classes,
        dtype=torch.float32,
        device=labels.device,
    )
    soft.scatter_(1, labels.unsqueeze(1), confidence + smoothing / num_classes)
    return soft


def soft_target_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor
) -> torch.Tensor:
    """Mean-over-batch cross-entropy against a soft target distribution."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _rand_bbox(height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
    """Pick a random axis-aligned rectangle whose area is roughly (1 - lam).

    why: cut_ratio = sqrt(1 - lam) so the patch *area* matches (1 - lam)
    rather than its sides. The center is uniform over the image, and the
    patch is clipped to the image bounds (rounding the recomputed lam later).
    """
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    center_y = random.randint(0, height - 1)
    center_x = random.randint(0, width - 1)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    return y1, y2, x1, x2


def maybe_mix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
    mix_prob: float = 0.5,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """With probability ``mix_prob``, mix the batch via mixup or cutmix.

    Always returns a soft target so the caller can use a single soft-target
    cross-entropy loss for both mixed and unmixed batches.

    why split the probability 50/50 between mixup and cutmix when active:
    the two regularizers stress the model in different ways (interpolated
    pixels vs. spatial occlusion), and a 50/50 split tends to outperform
    either one alone on small fine-grained datasets like this one.
    """
    base_target = smoothed_one_hot(labels, num_classes, smoothing)
    if random.random() >= mix_prob:
        return images, base_target

    use_mixup = random.random() < 0.5
    alpha = mixup_alpha if use_mixup else cutmix_alpha
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    permuted_targets = base_target[permutation]

    if use_mixup:
        # why: linear interpolation in pixel space; I rely on Normalize having
        # already centered the inputs so the average is well-behaved.
        mixed_images = lam * images + (1.0 - lam) * images[permutation]
    else:
        height, width = images.shape[-2:]
        y1, y2, x1, x2 = _rand_bbox(height, width, lam)
        mixed_images = images.clone()
        mixed_images[..., y1:y2, x1:x2] = images[permutation][..., y1:y2, x1:x2]
        # why: recompute lam from the actual pasted area. Integer rounding in
        # _rand_bbox means the realised area can differ slightly from
        # (1 - lam_sampled), and the per-paper recipe wants the loss weight
        # to track the *actual* mix ratio.
        patch_area = max((y2 - y1) * (x2 - x1), 0)
        lam = 1.0 - patch_area / float(height * width)

    soft_target = lam * base_target + (1.0 - lam) * permuted_targets
    return mixed_images, soft_target
