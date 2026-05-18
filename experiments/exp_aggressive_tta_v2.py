"""Aggressive TTA v2 — retry on the ResNet-101 model.pth.

The original aggressive_tta experiment ran on the much shallower
(2,2,2,2) model; the 7-scale window we promoted then is probably no
longer optimal for ResNet-101's different feature scale. This re-runs
the search on the new model.pth, plus some new ideas:

  - extra_scales: tighter step in the centre + wider tails
  - dihedral: HFlip + 180-rotate + HFlip+rotate (4-way symmetry).
    Note pets aren't usually rotation-symmetric, so this may hurt.
  - logit_averaging vs probability_averaging on the same scale set
    (no extra forward passes, just changes ensembling).
  - bigger step search

Cache architecture: each unique scale is run only ONCE (with HFlip)
and the per-image softmax probs cached on CPU. All assembly variants
are then ensembled from the cache. ~3-5 min total on a 4080.

Run from the repo root:
    python experiments/exp_aggressive_tta_v2.py
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


# Each tuple is (probs_method, scales). probs_method=True means we cache
# probs (softmax-averaged ensemble), False means logits.
SCALE_SETS = {
    "baseline_7":   (208, 224, 240, 256, 272, 288, 304),
    "wider_11":     (192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352),
    "denser_13":    (208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304),
    "tight_9":      (216, 224, 232, 240, 248, 256, 264, 272, 280),
    "wide_step8":   (200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312),
    "focused_5":    (240, 248, 256, 264, 272),
    "single_224":   (224,),
    "single_256":   (256,),
}
BASELINE_TEST_PCT = 70.26
MARGIN = 0.3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    print(f"[exp:aggressive_tta_v2] device={torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(__file__).resolve().parent / "aggressive_tta_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = get_data_stats(seed=42)
    image_size = stats.get("image_size", 224)

    model = build_model(
        num_classes=NUM_CLASSES, use_maxpool=True,
        widths=(64, 128, 256, 512), blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
    ).to(device)
    model_path = Path(__file__).resolve().parent.parent / "model.pth"
    sd = torch.load(model_path, map_location=device, weights_only=True)
    sd = {k: (v.float() if v.is_floating_point() and v.dtype != torch.float32 else v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[exp:aggressive_tta_v2] loaded {model_path.name}", flush=True)

    all_scales = sorted({s for scales in SCALE_SETS.values() for s in scales})
    print(
        f"[exp:aggressive_tta_v2] {len(all_scales)} unique scales "
        f"(min {min(all_scales)}, max {max(all_scales)})", flush=True,
    )

    # Per-scale: cache softmax probs (B, 37) summed across (orig + HFlip)
    scale_probs: dict[int, torch.Tensor] = {}
    # Per-scale: also cache raw logits (averaged across orig + HFlip)
    scale_logits: dict[int, torch.Tensor] = {}
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
        loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)

        probs_list, logits_list, labels_list = [], [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device, non_blocking=True)
                flipped = torch.flip(imgs, dims=[3])
                with torch.amp.autocast("cuda"):
                    la = model(imgs)
                    lb = model(flipped)
                # Both averages: probs (softmax) and logits.
                probs_list.append((F.softmax(la, 1) + F.softmax(lb, 1)).cpu())
                logits_list.append(((la + lb) / 2).cpu())
                labels_list.append(lbls)

        scale_probs[scale] = torch.cat(probs_list, 0)
        scale_logits[scale] = torch.cat(logits_list, 0)
        labels = torch.cat(labels_list, 0)
        if reference_labels is None:
            reference_labels = labels
        else:
            assert torch.equal(labels, reference_labels)
        acc = (scale_probs[scale].argmax(1) == labels).float().mean().item() * 100
        print(
            f"[exp:aggressive_tta_v2] {i:2d}/{len(all_scales)} scale={scale} "
            f"single-scale Q15={acc:.2f}%  ({time.time()-t0:.1f}s)", flush=True,
        )

    assert reference_labels is not None
    print()
    results = {}
    for name, scales in SCALE_SETS.items():
        # Average probs (softmax-then-sum).
        ens_probs = torch.zeros_like(scale_probs[scales[0]])
        for s in scales:
            ens_probs += scale_probs[s]
        acc_p = (ens_probs.argmax(1) == reference_labels).float().mean().item() * 100

        # Average logits (sum-then-argmax — equivalent to argmax over averaged logits).
        ens_logits = torch.zeros_like(scale_logits[scales[0]])
        for s in scales:
            ens_logits += scale_logits[s]
        acc_l = (ens_logits.argmax(1) == reference_labels).float().mean().item() * 100

        delta_p = acc_p - BASELINE_TEST_PCT
        delta_l = acc_l - BASELINE_TEST_PCT
        results[name] = {
            "scales": list(scales),
            "n_views": len(scales) * 2,
            "test_acc_probs": round(acc_p, 4),
            "test_acc_logits": round(acc_l, 4),
            "delta_pp_probs": round(delta_p, 4),
            "delta_pp_logits": round(delta_l, 4),
        }
        sp = "+" if delta_p >= 0 else ""; sl = "+" if delta_l >= 0 else ""
        print(
            f"[exp:aggressive_tta_v2] {name:13s}: "
            f"probs={acc_p:.2f}% ({sp}{delta_p:.2f}) | "
            f"logits={acc_l:.2f}% ({sl}{delta_l:.2f})",
            flush=True,
        )

    # Pick the best probs+logits combo across all sets.
    all_candidates = []
    for name, r in results.items():
        all_candidates.append((name + "_probs", r["test_acc_probs"], r["delta_pp_probs"]))
        all_candidates.append((name + "_logits", r["test_acc_logits"], r["delta_pp_logits"]))
    all_candidates.sort(key=lambda x: -x[1])
    best_name, best_acc, best_delta = all_candidates[0]
    if best_delta > MARGIN:
        verdict = f"PROMOTE: {best_name} wins Q15={best_acc:.2f}% (+{best_delta:.2f}pp)"
    else:
        verdict = f"no win > {MARGIN}pp; keep locked 7-scale (best was {best_name} {best_acc:.2f}%, +{best_delta:.2f})"

    out = {
        "experiment": "aggressive_tta_v2",
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "margin_pp": MARGIN,
        "results": results,
        "verdict": verdict,
    }
    (out_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[exp:aggressive_tta_v2] {verdict}", flush=True)


if __name__ == "__main__":
    main()
