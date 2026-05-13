"""Experiment: search a wider grid of TTA configurations on the locked model.pth.

The promoted 3-scale + HFlip TTA pushes Q15 from 45.60% -> 46.42%. This
script tries a few more configurations on the same weights to see if any
of them lift it further:

  - tight_3scale    : (240, 256, 272)  - narrower spread around train res
  - current_3scale  : (224, 256, 288)  - the current live recipe
  - dense_5scale    : (224, 240, 256, 272, 288)
  - wide_5scale     : (216, 240, 256, 272, 296)
  - dense_7scale    : (208, 224, 240, 256, 272, 288, 304)
  - 10crop_at_256   : 5 spatial crops at scale 256 + HFlip of each
  - 10crop_multi    : 10crop at three scales (224, 256, 288)

Run:
    python experiments/exp_tta_search.py

Writes experiments/tta_search/result.json with the full table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATA_ROOT, IMAGE_SIZE, STATS_PATH
from src.model import build_model


EXPERIMENT_NAME = "tta_search"
EXPERIMENT_DIR = Path(__file__).resolve().parent / EXPERIMENT_NAME
BASELINE_TEST_PCT = 46.42  # The promoted 3-scale TTA result.
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pth"

BATCH_SIZE = 128
NUM_WORKERS = 2


def _multiscale_transform(scale: int, mean, std) -> T.Compose:
    return T.Compose(
        [
            T.Resize(scale),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


class _FiveCropStack:
    """Stack a 5-crop tuple of PIL images into a normalized (5,C,H,W) tensor.

    why: torchvision's FiveCrop returns a tuple of 5 PIL crops. The standard
    idiom uses a `T.Lambda` to stack them, but lambdas don't pickle on
    Windows' spawn-based DataLoader workers - so I bind the same logic to
    a top-level class which does pickle.
    """

    def __init__(self, mean, std):
        self._to_tensor = T.ToTensor()
        self._normalize = T.Normalize(mean, std)

    def __call__(self, crops):
        return torch.stack([self._normalize(self._to_tensor(c)) for c in crops])


def _fivecrop_transform(scale: int, mean, std) -> T.Compose:
    """Resize -> FiveCrop(224) -> stack to (5, C, H, W) normalized tensor."""
    return T.Compose(
        [
            T.Resize(scale),
            T.FiveCrop(IMAGE_SIZE),
            _FiveCropStack(mean, std),
        ]
    )


@torch.no_grad()
def _accumulate_probs_multiscale(model, transform, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass + HFlip on a dataset using `transform`; returns summed probs and labels."""
    ds = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    probs_chunks, label_chunks = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            logits_flip = model(torch.flip(images, dims=[3]))
        probs = F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)
        probs_chunks.append(probs.cpu())
        label_chunks.append(labels)
    return torch.cat(probs_chunks, dim=0), torch.cat(label_chunks, dim=0)


@torch.no_grad()
def _accumulate_probs_5crop(model, scale: int, mean, std, device) -> tuple[torch.Tensor, torch.Tensor]:
    """5 spatial crops + HFlip of each = 10 forwards per image; returns summed probs."""
    transform = _fivecrop_transform(scale, mean, std)
    ds = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        target_types="category",
        download=True,
        transform=transform,
    )
    # why: each sample is (5, C, H, W); a batch of B becomes (B, 5, C, H, W).
    # I flatten the first two dims for the forward, then reshape back.
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE // 5, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    probs_chunks, label_chunks = [], []
    for images, labels in loader:
        b, ncrops = images.shape[0], images.shape[1]
        flat = images.view(b * ncrops, *images.shape[2:]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(flat)
            logits_flip = model(torch.flip(flat, dims=[3]))
        probs = F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)
        probs = probs.view(b, ncrops, -1).sum(dim=1)
        probs_chunks.append(probs.cpu())
        label_chunks.append(labels)
    return torch.cat(probs_chunks, dim=0), torch.cat(label_chunks, dim=0)


def evaluate_config(model, cfg: dict, stats: dict, device) -> tuple[float, dict]:
    """Run one TTA config; return (test_acc_pct, per-scale diagnostics dict)."""
    mean, std = stats["mean"], stats["std"]
    ensembled = None
    ref_labels = None
    diagnostics = {}

    if cfg["kind"] == "multiscale":
        for scale in cfg["scales"]:
            transform = _multiscale_transform(scale, mean, std)
            probs, labels = _accumulate_probs_multiscale(model, transform, device)
            single_acc = (probs.argmax(dim=1) == labels).float().mean().item() * 100
            diagnostics[f"scale_{scale}"] = round(single_acc, 2)
            if ensembled is None:
                ensembled, ref_labels = probs.clone(), labels
            else:
                ensembled += probs

    elif cfg["kind"] == "fivecrop_multiscale":
        for scale in cfg["scales"]:
            probs, labels = _accumulate_probs_5crop(model, scale, mean, std, device)
            single_acc = (probs.argmax(dim=1) == labels).float().mean().item() * 100
            diagnostics[f"5crop_scale_{scale}"] = round(single_acc, 2)
            if ensembled is None:
                ensembled, ref_labels = probs.clone(), labels
            else:
                ensembled += probs
    else:
        raise ValueError(f"unknown kind {cfg['kind']}")

    correct = (ensembled.argmax(dim=1) == ref_labels).sum().item()
    return correct / ref_labels.numel() * 100, diagnostics


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} missing - run python train.py first.")

    stats = json.loads(Path(STATS_PATH).read_text(encoding="utf-8"))

    model = build_model(num_classes=stats["num_classes"]).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    configs = [
        {"name": "tight_3scale", "kind": "multiscale", "scales": (240, 256, 272)},
        {"name": "current_3scale", "kind": "multiscale", "scales": (224, 256, 288)},
        {"name": "dense_5scale", "kind": "multiscale",
         "scales": (224, 240, 256, 272, 288)},
        {"name": "wide_5scale", "kind": "multiscale",
         "scales": (216, 240, 256, 272, 296)},
        {"name": "dense_7scale", "kind": "multiscale",
         "scales": (208, 224, 240, 256, 272, 288, 304)},
        {"name": "10crop_at_256", "kind": "fivecrop_multiscale", "scales": (256,)},
        {"name": "10crop_3scale", "kind": "fivecrop_multiscale",
         "scales": (224, 256, 288)},
    ]

    results = []
    for cfg in configs:
        acc, diagnostics = evaluate_config(model, cfg, stats, device)
        delta = acc - BASELINE_TEST_PCT
        margin = 0.3  # tighter than the first-pass 0.5 since we're chasing diminishing returns
        if acc > BASELINE_TEST_PCT + margin:
            verdict = "BEATS BASELINE"
        elif acc < BASELINE_TEST_PCT - margin:
            verdict = "WORSE"
        else:
            verdict = "within noise"
        results.append(
            {
                "name": cfg["name"],
                "kind": cfg["kind"],
                "scales": list(cfg["scales"]),
                "test_acc_pct": round(acc, 2),
                "delta_vs_baseline_pp": round(delta, 2),
                "verdict": verdict,
                "diagnostics": diagnostics,
            }
        )
        print(
            f"[exp:{EXPERIMENT_NAME}] {cfg['name']:<20s} "
            f"acc={acc:5.2f}%  delta={delta:+.2f}pp  {verdict}"
        )

    # Print summary sorted by accuracy desc.
    results_sorted = sorted(results, key=lambda r: r["test_acc_pct"], reverse=True)
    print(f"\n[exp:{EXPERIMENT_NAME}] best -> {results_sorted[0]['name']} "
          f"({results_sorted[0]['test_acc_pct']:.2f}%)")

    (EXPERIMENT_DIR / "result.json").write_text(
        json.dumps(
            {
                "baseline_test_acc_pct": BASELINE_TEST_PCT,
                "ranked": results_sorted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
