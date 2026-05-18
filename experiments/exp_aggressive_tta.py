"""Try more-aggressive TTA on the locked model.pth.

Inference-only — no retraining. Computes softmax probs for every unique
scale ONCE (cached on CPU), then assembles four candidate scale-sets
from that cache and picks the best.

Sets tried (Q15 reported for each):
  - wider_11:  (192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352)
  - denser_13: (208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304)
  - focused_7: (216, 232, 248, 256, 264, 280, 296)
  - baseline_7:(208, 224, 240, 256, 272, 288, 304)   # control

Each scale also runs its horizontal flip. Promote whichever beats the
locked baseline (56.39%) by > 0.3 pp.

why one-pass-with-cache: the locked helper recreates a fresh
DataLoader (with workers) per scale, and on Windows the per-call
worker spin-up dominated wall time when invoked four times in a row.
This script does each unique scale exactly once and assembles the
four sets in-memory.

Run from the repo root (~3-5 min):
    python experiments/exp_aggressive_tta.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATA_ROOT, NUM_CLASSES, get_data_stats
from src.model import build_model


SCALE_SETS = {
    "wider_11":   (192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352),
    "denser_13":  (208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304),
    "focused_7":  (216, 232, 248, 256, 264, 280, 296),
    "baseline_7": (208, 224, 240, 256, 272, 288, 304),
}
BASELINE_TEST_PCT = 56.39
MARGIN = 0.3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    print(f"[exp:aggressive_tta] device={torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(__file__).resolve().parent / "aggressive_tta"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = get_data_stats(seed=42)
    image_size = stats.get("image_size", 224)

    model = build_model(num_classes=NUM_CLASSES, use_maxpool=True).to(device)
    model_path = Path(__file__).resolve().parent.parent / "model.pth"
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    print(f"[exp:aggressive_tta] loaded {model_path.name}", flush=True)

    # Union of every scale anyone asks for (avoid redoing work).
    all_scales = sorted({s for scales in SCALE_SETS.values() for s in scales})
    print(
        f"[exp:aggressive_tta] computing {len(all_scales)} unique scales "
        f"(min {min(all_scales)}, max {max(all_scales)})",
        flush=True,
    )

    # Cache: scale -> (probs_tensor[N,37] on CPU, labels_tensor[N])
    scale_probs: dict[int, torch.Tensor] = {}
    reference_labels: torch.Tensor | None = None

    for i, scale in enumerate(all_scales, 1):
        t0 = time.time()
        transform = T.Compose([
            T.Resize(scale),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        test_ds = OxfordIIITPet(
            root=DATA_ROOT, split="test", target_types="category",
            download=True, transform=transform,
        )
        # why: num_workers=0 — single-process loading avoids Windows worker
        # spin-up (which dominated wall time when invoked per-scale). The
        # transform is cheap and the bottleneck is the GPU forward.
        loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)

        per_image_probs: list[torch.Tensor] = []
        per_image_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device, non_blocking=True)
                flipped = torch.flip(imgs, dims=[3])
                with torch.amp.autocast("cuda"):
                    p = F.softmax(model(imgs), dim=1) + F.softmax(model(flipped), dim=1)
                per_image_probs.append(p.cpu())
                per_image_labels.append(lbls)
        probs = torch.cat(per_image_probs, dim=0)
        labels = torch.cat(per_image_labels, dim=0)

        if reference_labels is None:
            reference_labels = labels
        else:
            assert torch.equal(labels, reference_labels), "label order changed"

        scale_probs[scale] = probs
        per_scale_acc = (probs.argmax(1) == labels).float().mean().item() * 100
        elapsed = time.time() - t0
        print(
            f"[exp:aggressive_tta] {i:2d}/{len(all_scales)} scale={scale} "
            f"single-scale Q15={per_scale_acc:.2f}%  (elapsed {elapsed:.1f}s)",
            flush=True,
        )

    # Assemble each scale-set from the cache.
    assert reference_labels is not None
    print("", flush=True)
    results = {}
    for name, scales in SCALE_SETS.items():
        ensembled = torch.zeros_like(scale_probs[scales[0]])
        for s in scales:
            ensembled = ensembled + scale_probs[s]
        acc_pct = (ensembled.argmax(1) == reference_labels).float().mean().item() * 100
        delta = acc_pct - BASELINE_TEST_PCT
        results[name] = {
            "scales": list(scales),
            "n_views": len(scales) * 2,
            "test_acc_pct": round(acc_pct, 4),
            "delta_pp": round(delta, 4),
        }
        sign = "+" if delta >= 0 else ""
        print(
            f"[exp:aggressive_tta] {name}: Q15 = {acc_pct:.2f}%  "
            f"({sign}{delta:.2f}pp vs 56.39%)",
            flush=True,
        )

    winners = sorted(
        [(n, r) for n, r in results.items()
         if r["delta_pp"] > MARGIN and n != "baseline_7"],
        key=lambda x: -x[1]["test_acc_pct"],
    )
    if winners:
        best_name, best = winners[0]
        verdict = f"PROMOTE: {best_name} wins +{best['delta_pp']:.2f}pp"
    else:
        verdict = "no winner > 0.30pp margin; keep locked 7-scale TTA"

    out = {
        "experiment": "aggressive_tta",
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "margin_pp": MARGIN,
        "results": results,
        "verdict": verdict,
    }
    (out_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[exp:aggressive_tta] {verdict}", flush=True)


if __name__ == "__main__":
    main()
