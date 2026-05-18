"""TTA refinement on the locked 5-crop pipeline.

Tests several scale sets with 5-crop + HFlip on the current BlurPool +
ResNet-101 model.pth. Caches per-scale 5-crop probabilities once and
assembles all candidate scale-sets in-memory.

Scale sets tried:
  baseline_7   (208, 224, 240, 256, 272, 288, 304)   - locked
  wider_9      (192, 208, 224, 240, 256, 272, 288, 304, 320)
  denser_9     (208, 216, 224, 232, 240, 256, 272, 288, 304)
  peak_7       (216, 224, 232, 240, 248, 256, 272)
  tight_5      (216, 232, 240, 248, 256)
  expand_11    (192, 208, 216, 224, 232, 240, 256, 272, 288, 304, 320)
  centered_5   (224, 240, 256, 272, 288)

For each set we report both probability-space and logit-space
averaging. Promote whichever beats the locked baseline by > 0.3pp.

Run from the repo root (~10-15 min):
    python experiments/exp_5crop_tta_refinement.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATA_ROOT, NUM_CLASSES, get_data_stats
from src.model import build_model
from src.train_loop import _five_crop


SCALE_SETS = {
    "baseline_7":  (208, 224, 240, 256, 272, 288, 304),
    "wider_9":     (192, 208, 224, 240, 256, 272, 288, 304, 320),
    "denser_9":    (208, 216, 224, 232, 240, 256, 272, 288, 304),
    "peak_7":      (216, 224, 232, 240, 248, 256, 272),
    "tight_5":     (216, 232, 240, 248, 256),
    "expand_11":   (192, 208, 216, 224, 232, 240, 256, 272, 288, 304, 320),
    "centered_5":  (224, 240, 256, 272, 288),
}
BASELINE_TEST_PCT = 71.25
MARGIN = 0.3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    print(f"[exp:5crop_tta_refine] device={torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(__file__).resolve().parent / "5crop_tta_refinement"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = get_data_stats(seed=42)
    image_size = stats.get("image_size", 224)

    model = build_model(
        num_classes=NUM_CLASSES, use_maxpool=True,
        widths=(64, 128, 256, 512), blocks_per_stage=(3, 4, 23, 3),
        use_blurpool=True,
    ).to(device)
    sd = torch.load(
        Path(__file__).resolve().parent.parent / "model.pth",
        map_location=device, weights_only=True,
    )
    sd = {k: (v.float() if v.is_floating_point() and v.dtype != torch.float32 else v)
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("[exp:5crop_tta_refine] model loaded", flush=True)

    # Compute every unique scale once with 5-crop + HFlip; cache CPU probs
    # and logits separately so we can build both averaging variants.
    all_scales = sorted({s for ss in SCALE_SETS.values() for s in ss})
    print(f"[exp:5crop_tta_refine] {len(all_scales)} unique scales: {all_scales}", flush=True)

    class _FiveCropDS(Dataset):
        def __init__(self, raw_ds, base_t, crop):
            self.raw = raw_ds; self.base = base_t; self.crop = crop
        def __len__(self): return len(self.raw)
        def __getitem__(self, i):
            img, lbl = self.raw[i]
            t = self.base(img)
            if t.shape[-2] < self.crop or t.shape[-1] < self.crop:
                t = F.interpolate(t.unsqueeze(0), size=self.crop,
                                  mode="bilinear", align_corners=False).squeeze(0)
            return _five_crop(t, self.crop, self.crop), lbl

    scale_probs: dict[int, torch.Tensor] = {}
    scale_logits: dict[int, torch.Tensor] = {}  # sum of fp32 logits
    reference_labels: torch.Tensor | None = None

    for i, scale in enumerate(all_scales, 1):
        t0 = time.time()
        base_t = T.Compose([
            T.Resize(scale),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        raw = OxfordIIITPet(root=DATA_ROOT, split="test",
                            target_types="category", download=True)
        ds = _FiveCropDS(raw, base_t, image_size)
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)

        per_probs: list[torch.Tensor] = []
        per_logits: list[torch.Tensor] = []
        per_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for crops, lbls in loader:
                b, n_crops, c, s, _ = crops.shape
                flat = crops.view(b * n_crops, c, s, s).to(device, non_blocking=True)
                flipped = torch.flip(flat, dims=[3])
                with torch.amp.autocast("cuda"):
                    la = model(flat)
                    lb = model(flipped)
                # accumulate prob-space (current default)
                p = F.softmax(la, dim=1) + F.softmax(lb, dim=1)
                p = p.view(b, n_crops, -1).sum(dim=1)  # sum over 5 crops
                per_probs.append(p.cpu())
                # accumulate logit-space (alternative averaging)
                lo = (la.float() + lb.float()).view(b, n_crops, -1).sum(dim=1)
                per_logits.append(lo.cpu())
                per_labels.append(lbls)

        probs_t = torch.cat(per_probs, dim=0)
        logits_t = torch.cat(per_logits, dim=0)
        labels_t = torch.cat(per_labels, dim=0)
        if reference_labels is None:
            reference_labels = labels_t
        else:
            assert torch.equal(labels_t, reference_labels)
        scale_probs[scale] = probs_t
        scale_logits[scale] = logits_t
        per_scale_acc = (probs_t.argmax(1) == labels_t).float().mean().item() * 100
        print(
            f"[exp:5crop_tta_refine] {i:2d}/{len(all_scales)} scale={scale} "
            f"single-scale-5crop Q15={per_scale_acc:.2f}%  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    # Assemble each scale-set under both averaging modes.
    print("", flush=True)
    assert reference_labels is not None
    results: dict[str, dict] = {}
    for name, scales in SCALE_SETS.items():
        probs_sum = torch.zeros_like(scale_probs[scales[0]])
        logits_sum = torch.zeros_like(scale_logits[scales[0]])
        for s in scales:
            probs_sum = probs_sum + scale_probs[s]
            logits_sum = logits_sum + scale_logits[s]
        probs_acc = (probs_sum.argmax(1) == reference_labels).float().mean().item() * 100
        logits_acc = (logits_sum.argmax(1) == reference_labels).float().mean().item() * 100
        results[name] = {
            "scales": list(scales),
            "probs_acc_pct": round(probs_acc, 4),
            "logits_acc_pct": round(logits_acc, 4),
            "probs_delta_pp": round(probs_acc - BASELINE_TEST_PCT, 4),
            "logits_delta_pp": round(logits_acc - BASELINE_TEST_PCT, 4),
        }
        print(
            f"[exp:5crop_tta_refine] {name:12s}: probs={probs_acc:.2f}% "
            f"({results[name]['probs_delta_pp']:+.2f}) | "
            f"logits={logits_acc:.2f}% ({results[name]['logits_delta_pp']:+.2f})",
            flush=True,
        )

    # Find winner.
    winners = []
    for name, r in results.items():
        if name == "baseline_7":
            continue
        if r["probs_delta_pp"] > MARGIN:
            winners.append((name, "probs", r["probs_acc_pct"], r["probs_delta_pp"]))
        if r["logits_delta_pp"] > MARGIN:
            winners.append((name, "logits", r["logits_acc_pct"], r["logits_delta_pp"]))
    winners.sort(key=lambda x: -x[2])
    if winners:
        bn, bm, ba, bd = winners[0]
        verdict = f"PROMOTE {bn}_{bm}: {ba:.2f}% (+{bd:.2f}pp)"
    else:
        verdict = "no win > 0.30pp margin; keep locked 7-scale 5-crop"

    (out_dir / "result.json").write_text(json.dumps({
        "experiment": "5crop_tta_refinement",
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "results": results,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\n[exp:5crop_tta_refine] {verdict}", flush=True)


if __name__ == "__main__":
    main()
