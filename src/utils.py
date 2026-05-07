"""Small utilities I use across training and evaluation.

I keep these out of train.py / test.py so the entrypoint scripts stay focused
on orchestration. Nothing here touches the model architecture or the training
recipe — just seeding, metric tracking, top-1 accuracy, and a thin wrapper
around torchinfo.summary so MODEL_SUMMARY.txt always reflects the current model.
"""

from __future__ import annotations

import functools
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def seed_all(seed: int = 42) -> torch.Generator:
    """Seed every RNG I touch and lock cuDNN into deterministic mode.

    The spec requires my reported accuracy to reproduce within ±3% on re-run,
    so I trade some throughput for determinism. The Generator I return is
    handed to the DataLoader so the shuffle order is also reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # why: forces conv ops to deterministic algorithms; benchmark=True would
    # pick the fastest algorithm per shape but introduces non-determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _seeded_worker_init(worker_id: int, seed: int) -> None:
    """Module-level worker init so it pickles cleanly on Windows spawn."""
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init(seed: int = 42) -> Callable[[int], None]:
    """Build a DataLoader worker_init_fn that seeds python/numpy/torch per worker.

    why: PyTorch's default worker init only seeds torch - anything that reaches
    into numpy.random or python's random (some torchvision transforms do)
    would otherwise still be non-deterministic across workers. I use
    functools.partial of a module-level function (rather than a closure) so
    the function is picklable for Windows' spawn-based DataLoader workers.
    """
    return functools.partial(_seeded_worker_init, seed=seed)


class AverageMeter:
    """Streaming average for a scalar metric (loss, accuracy, ...).

    I prefer this over collecting all batch values in a list: it gives the
    correct sample-weighted epoch average even when the last batch is smaller,
    and keeps memory flat across long training runs.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy on a batch, returned as a fraction in [0, 1]."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def write_model_summary(
    model: torch.nn.Module,
    path: str | Path = "MODEL_SUMMARY.txt",
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> None:
    """Dump torchinfo.summary to disk so MODEL_SUMMARY.txt tracks the live model.

    I call this from train.py just before training starts. Lazy-importing
    torchinfo keeps it out of the critical path for test.py.
    """
    from torchinfo import summary

    s = summary(
        model,
        input_size=input_size,
        col_names=("input_size", "output_size", "num_params"),
        depth=4,
        verbose=0,
    )
    Path(path).write_text(str(s), encoding="utf-8")
