"""Experiment: alternative recipe from possible_improvement.py.

A different baseline was shared with me, reportedly gets close to 60%.
Material differences from the locked recipe (46.74% test):

  Architecture
    - Post-activation ResNet with LeakyReLU(0.1) instead of pre-activation
      with SiLU
    - 1 block per stage instead of 2
    - 7x7 stride-2 stem + MaxPool (ImageNet-style) instead of my 3x3
      stride-2 stem with no MaxPool
    - No Squeeze-and-Excitation
    - Two-layer head: 512 -> 256 (with BN) -> 37, instead of my
      Dropout(0.2) -> Linear(512, 37)
    - Dropout 0.4 vs 0.2

  Optimization
    - AdamW lr=1e-3, weight_decay=1e-2 instead of SGD-Nesterov lr=1e-1,
      weight_decay=5e-4
    - OneCycleLR max_lr=4e-3, pct_start=0.3 instead of max_lr=1e-1,
      pct_start=0.17
    - Batch size 64 (no AMP) instead of 128 with AMP
    - Label smoothing 0.05 instead of 0.1

  Data
    - ImageNet normalization stats (0.485/0.456/0.406, 0.229/0.224/0.225)
      instead of trainval-computed stats - see note below
    - Trains on the full 3680-image trainval set (no train/val split)
    - Lighter augmentation: HFlip + Rotation(10 deg) + ColorJitter(0.15)
      instead of RandAugment(magnitude=7)

Spec note: ImageNet aggregate channel stats are a much weaker form of
"borrowed knowledge" than pretrained weights, but the IMLO spec is
explicit about training from scratch with no external data. My locked
recipe computes stats from trainval. I'm faithfully replicating the
alternative recipe here (so the comparison is recipe-vs-recipe); if it
wins I'll discuss the spec implications before promoting.

Run from the repo root (about 5-10 minutes):
    python experiments/exp_possible_improvement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATA_ROOT


EXPERIMENT_NAME = "possible_improvement"
EXPERIMENT_DIR = Path(__file__).resolve().parent / EXPERIMENT_NAME
BASELINE_TEST_PCT = 46.74

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224
NUM_CLASSES = 37

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResidualBlock(nn.Module):
    """Post-activation residual block with LeakyReLU - matches the alt recipe."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.leaky_relu(out)


class AltPetClassifier(nn.Module):
    """The alternative architecture from possible_improvement.py, verbatim."""

    def __init__(self, num_classes: int = 37) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = ResidualBlock(64, 64, stride=1)
        self.stage2 = ResidualBlock(64, 128, stride=2)
        self.stage3 = ResidualBlock(128, 256, stride=2)
        self.stage4 = ResidualBlock(256, 512, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.leaky_relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


def _build_eval_transform(scale: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(scale, antialias=True),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.no_grad()
def evaluate_simple(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_multiscale_tta(
    model: nn.Module,
    device: torch.device,
    scales: tuple[int, ...] = (208, 224, 240, 256, 272, 288, 304),
    batch_size: int = 128,
) -> float:
    """Same 7-scale + HFlip TTA as the live test.py, but with ImageNet normalization."""
    model.eval()
    ensembled = None
    ref_labels = None
    for scale in scales:
        transform = _build_eval_transform(scale)
        ds = datasets.OxfordIIITPet(
            root=DATA_ROOT, split="test", target_types="category",
            download=True, transform=transform,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=min(NUM_WORKERS, 2), pin_memory=True,
        )
        scale_probs = []
        scale_labels = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            flipped = torch.flip(images, dims=[3])
            logits = model(images)
            logits_flip = model(flipped)
            probs = F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)
            scale_probs.append(probs.cpu())
            scale_labels.append(labels)
        scale_probs = torch.cat(scale_probs, dim=0)
        scale_labels = torch.cat(scale_labels, dim=0)
        if ensembled is None:
            ensembled = scale_probs.clone()
            ref_labels = scale_labels
        else:
            ensembled += scale_probs
    assert ensembled is not None and ref_labels is not None
    return (ensembled.argmax(dim=1) == ref_labels).float().mean().item()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment.")
    device = torch.device("cuda")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        download=True, transform=train_transforms,
    )
    eval_train_ds = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="trainval", target_types="category",
        download=True, transform=eval_transforms,
    )
    test_ds_simple = datasets.OxfordIIITPet(
        root=DATA_ROOT, split="test", target_types="category",
        download=True, transform=eval_transforms,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    eval_train_loader = DataLoader(
        eval_train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=min(NUM_WORKERS, 2), pin_memory=True,
    )
    test_loader_simple = DataLoader(
        test_ds_simple, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=min(NUM_WORKERS, 2), pin_memory=True,
    )

    model = AltPetClassifier(num_classes=NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[exp:{EXPERIMENT_NAME}] params={param_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=4e-3, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.3,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        train_acc = correct / total * 100
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[exp:{EXPERIMENT_NAME}] epoch {epoch:02d}/{EPOCHS} | "
            f"lr={lr_now:.4f} | loss={avg_loss:.4f} | train_acc={train_acc:.2f}%"
        )

    torch.save(model.state_dict(), EXPERIMENT_DIR / "model.pth")
    print(f"[exp:{EXPERIMENT_NAME}] saved model.pth")

    q14_acc = evaluate_simple(model, eval_train_loader, device) * 100
    q15_simple_acc = evaluate_simple(model, test_loader_simple, device) * 100
    q15_tta_acc = evaluate_multiscale_tta(model, device) * 100

    delta = q15_tta_acc - BASELINE_TEST_PCT
    margin = 0.3
    if q15_tta_acc > BASELINE_TEST_PCT + margin:
        verdict = "BEATS BASELINE"
    elif q15_tta_acc < BASELINE_TEST_PCT - margin:
        verdict = "WORSE THAN BASELINE"
    else:
        verdict = "within noise"

    result = {
        "experiment": EXPERIMENT_NAME,
        "param_count": param_count,
        "q14_full_trainval_acc_pct": round(q14_acc, 2),
        "q15_test_acc_pct_single_pass": round(q15_simple_acc, 2),
        "q15_test_acc_pct_7scale_tta": round(q15_tta_acc, 2),
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "delta_pp": round(delta, 2),
        "verdict": verdict,
        "notes": (
            "Uses ImageNet normalization stats (vs trainval-computed stats in "
            "the locked recipe). Trains on full 3680-image trainval (no val "
            "split). Architecture is post-activation ResNet with LeakyReLU, "
            "1 block per stage, no SE. AdamW + OneCycle."
        ),
    }
    (EXPERIMENT_DIR / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"[exp:{EXPERIMENT_NAME}] Q14 trainval (no TTA)            = {q14_acc:.2f}%")
    print(f"[exp:{EXPERIMENT_NAME}] Q15 test (single pass, no TTA)   = {q15_simple_acc:.2f}%")
    print(f"[exp:{EXPERIMENT_NAME}] Q15 test (7-scale + HFlip TTA)   = {q15_tta_acc:.2f}%")
    print(
        f"[exp:{EXPERIMENT_NAME}] vs baseline {BASELINE_TEST_PCT:.2f}% : "
        f"{'+' if delta >= 0 else ''}{delta:.2f}pp  ->  {verdict}"
    )


if __name__ == "__main__":
    main()
