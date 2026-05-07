"""Data pipeline for the Oxford-IIIT Pet dataset.

Single source of truth for: where the data lives, what transforms I apply,
how I split trainval into train/val, and the channel statistics I normalize
with. Train and test scripts both go through here so the augmentation,
normalization and split are guaranteed to match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

DATA_ROOT = "./data"
STATS_PATH = "data_stats.json"
NUM_CLASSES = 37
IMAGE_SIZE = 224
# why: 368 = exactly 10% of the 3680 trainval images. With 37 classes this
# leaves ~10 val samples per class — small but enough to track epoch-to-epoch
# trends without eating into training data.
VAL_SIZE = 368


def _stats_transform() -> T.Compose:
    """Eval-style transform used for the one-time channel-stats pass.

    why: I compute mean/std on the same shape the model will see at inference
    time. Computing on raw varying-size images would give marginally different
    numbers and a less faithful normalization.
    """
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
        ]
    )


def _compute_stats_and_split(seed: int) -> dict:
    """Single pass over trainval that produces both channel stats and labels.

    why: I'd otherwise iterate the dataset twice (once for stats, once for
    labels). Folding both into one loop halves the IO on the first run.
    """
    ds = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="category",
        download=True,
        transform=_stats_transform(),
    )
    # why: shuffle=False — the labels list I build below indexes by dataset
    # position, and the splitter below works on that same index space.
    loader = DataLoader(ds, batch_size=64, num_workers=2, shuffle=False)

    pixel_count = 0.0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq = torch.zeros(3, dtype=torch.float64)
    labels: list[int] = []
    for imgs, lbls in tqdm(loader, desc="stats"):
        imgs = imgs.to(torch.float64)
        b, _, h, w = imgs.shape
        pixel_count += b * h * w
        channel_sum += imgs.sum(dim=(0, 2, 3))
        channel_sq += (imgs ** 2).sum(dim=(0, 2, 3))
        labels.extend(lbls.tolist())

    mean = (channel_sum / pixel_count).tolist()
    # why: Var = E[X^2] - E[X]^2. Clamp non-negative to absorb the rare
    # floating-point underflow that could push a tiny variance below zero
    # before sqrt.
    var = (channel_sq / pixel_count) - torch.tensor(mean, dtype=torch.float64) ** 2
    std = var.clamp_min(0).sqrt().tolist()

    labels_np = np.asarray(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_np)), labels_np))

    return {
        "mean": mean,
        "std": std,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "seed": seed,
    }


def get_data_stats(seed: int = 42) -> dict:
    """Load cached stats / split, computing once on the first call.

    The cache file is intentionally committed to the repo so a fresh clone
    sees the exact split (and so test.py can normalize without ever touching
    the trainval set).
    """
    cache = Path(STATS_PATH)
    if cache.exists():
        with cache.open("r", encoding="utf-8") as f:
            return json.load(f)
    stats = _compute_stats_and_split(seed=seed)
    with cache.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def build_train_transform(mean: list[float], std: list[float]) -> T.Compose:
    """Aggressive train augmentation: spatial + photometric + erasing.

    Order matters — PIL-domain transforms (RandomResizedCrop, RandAugment,
    ColorJitter) run before ToTensor, tensor-domain transforms (Normalize,
    RandomErasing) run after.
    """
    return T.Compose(
        [
            # why: pets typically fill the frame, so a [0.6, 1.0] crop scale is
            # less destructive than the [0.08, 1.0] ImageNet default while
            # still giving useful translation/scale variance.
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            # why: RandAugment (Cubuk et al., 2020) — magnitude 9 / 2 ops gives
            # a strong default mix of geometric and photometric augmentations
            # without per-dataset tuning.
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.3, 0.3, 0.3),
            T.ToTensor(),
            T.Normalize(mean, std),
            # why: RandomErasing on the normalized tensor (Zhong et al., 2017).
            # Low p=0.25 because it stacks with RandAugment and CutMix later.
            T.RandomErasing(p=0.25),
        ]
    )


def build_eval_transform(mean: list[float], std: list[float]) -> T.Compose:
    """Deterministic resize + center crop + normalize for val/test/clean-train."""
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def build_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> dict:
    """Build every DataLoader the training script needs.

    Keys returned: 'train' (augmented, shuffled, drop_last), 'val' (eval
    transform, no shuffle), 'clean_train' (eval transform on the train subset
    for unaugmented train accuracy), 'full_trainval' (eval transform on all
    3680 trainval images for the Q14 number), 'test', and 'stats'.
    """
    stats = get_data_stats(seed=seed)
    train_t = build_train_transform(stats["mean"], stats["std"])
    eval_t = build_eval_transform(stats["mean"], stats["std"])

    # why: two OxfordIIITPet instances backed by the same files on disk —
    # one with augmentation for actual training, one with the eval transform
    # for clean-train accuracy and the Q14 full-trainval pass.
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

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        **common,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        **common,
    )
    clean_train_loader = DataLoader(
        clean_train_subset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        **common,
    )
    full_trainval_loader = DataLoader(
        eval_trainval,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        **common,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "clean_train": clean_train_loader,
        "full_trainval": full_trainval_loader,
        "test": test_loader,
        "stats": stats,
    }
