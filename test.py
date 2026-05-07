"""Inference on the official Oxford-IIIT Pet test split.

Usage:
    python test.py

No CLI arguments. Loads model.pth (state dict only), evaluates on the
official test split with horizontal-flip test-time augmentation, and prints
the test accuracy as XX.YY%. Mirrors train.py's CUDA requirement: I fail
loudly when CUDA isn't available rather than fall through to a slow CPU run
that could produce a plausible-but-misleading number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

# why: same trick as train.py - put the repo root on sys.path so the markers
# can run `python test.py` directly from the unzipped directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import DATA_ROOT, STATS_PATH, build_eval_transform
from src.model import build_model
from src.train_loop import evaluate_with_tta


BATCH_SIZE = 128
NUM_WORKERS = 4
MODEL_PATH = "model.pth"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "test.py requires CUDA but torch.cuda.is_available() is False."
        )
    device = torch.device("cuda")

    stats_file = Path(STATS_PATH)
    if not stats_file.exists():
        raise FileNotFoundError(
            f"{STATS_PATH} is missing. It is generated on the first run of "
            "train.py and must be included in the submission alongside model.pth."
        )
    with stats_file.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    eval_transform = build_eval_transform(stats["mean"], stats["std"])
    test_dataset = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True,
        transform=eval_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model(num_classes=stats["num_classes"]).to(device)
    # why: weights_only=True - safer load when model.pth is just a state dict,
    # and silences the "weights_only=False is deprecated" warning on newer
    # PyTorch versions.
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    accuracy = evaluate_with_tta(model, test_loader, device)
    # why: 2dp percentage matches the Q15 form field on the submission system.
    print(f"{accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
