"""4-crop + multi-scale + HFlip TTA on the locked BlurPool+ResNet-101.

Inference-only. The locked 7-scale TTA already averages one centre
crop per scale + its HFlip = 14 views per test image. This experiment
adds 4 corner crops at each scale:

  per scale, per image: 5 crops (centre + 4 corners) x 2 (HFlip) = 10 views
  total per image: 7 scales x 10 = 70 views

If the model's discriminative signal lives mostly in the centre crop
(true for typical pet photos where the subject is centred), the
corner crops add weak/wrong votes and the ensemble hurts. If it lives
in different image regions (rare for pets), corner crops would help.

The earlier 10-crop test on the shallow model lost; running this on
the deeper net once to confirm or upset that finding.

Run from the repo root (~10-15 min):
    python experiments/exp_4crop_tta.py
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


SCALES = (208, 224, 240, 256, 272, 288, 304)
BASELINE_TEST_PCT = 70.26
MARGIN = 0.3


def five_crop(t: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """Return [5, C, crop_h, crop_w]: centre + 4 corners of t [C, H, W]."""
    _, h, w = t.shape
    assert h >= crop_h and w >= crop_w, f"{t.shape} too small for {crop_h}x{crop_w}"
    tl = t[:, :crop_h, :crop_w]
    tr = t[:, :crop_h, w - crop_w:]
    bl = t[:, h - crop_h:, :crop_w]
    br = t[:, h - crop_h:, w - crop_w:]
    cy = (h - crop_h) // 2
    cx = (w - crop_w) // 2
    ct = t[:, cy:cy + crop_h, cx:cx + crop_w]
    return torch.stack([ct, tl, tr, bl, br], dim=0)  # [5, C, crop_h, crop_w]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    print(f"[exp:4crop_tta] device={torch.cuda.get_device_name(0)}", flush=True)

    out_dir = Path(__file__).resolve().parent / "4crop_tta"
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
    sd = {k: (v.float() if v.is_floating_point() and v.dtype != torch.float32 else v)
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[exp:4crop_tta] loaded {model_path.name}", flush=True)

    # ensembled prob accumulators
    centre_probs: torch.Tensor | None = None
    five_probs: torch.Tensor | None = None
    reference_labels: torch.Tensor | None = None

    for i, scale in enumerate(SCALES, 1):
        t0 = time.time()
        # Resize -> ToTensor -> Normalize (no centre crop here; we crop after)
        base_t = T.Compose([
            T.Resize(scale),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        # Use the eval transform PLUS our own 5-crop step on the resized tensor.
        # OxfordIIITPet hands us PIL images via transform; we apply base_t to
        # get the resized normalised tensor, then five_crop in the loop.
        # But the DataLoader wants fixed-shape tensors per sample; we apply
        # both base_t and crop inside collate by writing a tiny custom
        # dataset wrapper inline.
        class _CroppedDS(torch.utils.data.Dataset):
            def __init__(self, raw_ds, base_transform, crop_size):
                self.raw_ds = raw_ds
                self.base_transform = base_transform
                self.crop_size = crop_size
            def __len__(self): return len(self.raw_ds)
            def __getitem__(self, idx):
                img, lbl = self.raw_ds[idx]
                t = self.base_transform(img)
                # If the image's shorter side is < crop_size, fall back to
                # interpolating up. (Pet images are typically much larger.)
                if t.shape[-2] < self.crop_size or t.shape[-1] < self.crop_size:
                    t = F.interpolate(t.unsqueeze(0), size=self.crop_size,
                                     mode="bilinear", align_corners=False).squeeze(0)
                crops = five_crop(t, self.crop_size, self.crop_size)  # [5, C, S, S]
                return crops, lbl

        raw = OxfordIIITPet(
            root=DATA_ROOT, split="test", target_types="category",
            download=True, transform=None,
        )
        # transform=None -> raw_ds returns PIL; we apply base_t in our wrapper
        # Need PIL-mode dataset for the wrapper. OxfordIIITPet with transform=None
        # returns PIL images by default.
        wrapped = _CroppedDS(
            OxfordIIITPet(root=DATA_ROOT, split="test",
                          target_types="category", download=True),
            base_t, image_size,
        )
        loader = DataLoader(wrapped, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)

        scale_centre_probs: list[torch.Tensor] = []
        scale_five_probs: list[torch.Tensor] = []
        scale_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for crops, lbls in loader:
                # crops: [B, 5, C, S, S]
                b, n_crops, c, s, _ = crops.shape
                flat = crops.view(b * n_crops, c, s, s).to(device, non_blocking=True)
                flipped = torch.flip(flat, dims=[3])
                with torch.amp.autocast("cuda"):
                    logits = model(flat)
                    logits_f = model(flipped)
                probs = F.softmax(logits, dim=1) + F.softmax(logits_f, dim=1)
                probs = probs.view(b, n_crops, -1)  # [B, 5, 37]
                # centre-only path: just crop 0
                scale_centre_probs.append(probs[:, 0].cpu())
                # 5-crop path: sum over the 5 crops
                scale_five_probs.append(probs.sum(dim=1).cpu())
                scale_labels.append(lbls)
        centre_t = torch.cat(scale_centre_probs, dim=0)
        five_t = torch.cat(scale_five_probs, dim=0)
        labels_t = torch.cat(scale_labels, dim=0)

        if centre_probs is None:
            centre_probs = centre_t.clone()
            five_probs = five_t.clone()
            reference_labels = labels_t
        else:
            assert torch.equal(labels_t, reference_labels)
            centre_probs = centre_probs + centre_t
            five_probs = five_probs + five_t

        centre_acc = (centre_t.argmax(1) == labels_t).float().mean().item() * 100
        five_acc = (five_t.argmax(1) == labels_t).float().mean().item() * 100
        elapsed = time.time() - t0
        print(
            f"[exp:4crop_tta] {i}/{len(SCALES)} scale={scale} "
            f"centre={centre_acc:.2f}% 5crop={five_acc:.2f}%  ({elapsed:.1f}s)",
            flush=True,
        )

    assert centre_probs is not None and reference_labels is not None
    centre_ensemble = (centre_probs.argmax(1) == reference_labels).float().mean().item() * 100
    five_ensemble = (five_probs.argmax(1) == reference_labels).float().mean().item() * 100

    delta_centre = centre_ensemble - BASELINE_TEST_PCT
    delta_five = five_ensemble - BASELINE_TEST_PCT
    print(flush=True)
    print(f"[exp:4crop_tta] 7-scale centre-only (control): "
          f"{centre_ensemble:.2f}% ({delta_centre:+.2f} vs baseline)", flush=True)
    print(f"[exp:4crop_tta] 7-scale 5-crop ensemble: "
          f"{five_ensemble:.2f}% ({delta_five:+.2f} vs baseline)", flush=True)

    if delta_five > MARGIN:
        verdict = f"PROMOTE 5-crop: +{delta_five:.2f}pp"
    else:
        verdict = "no win > 0.3pp margin; keep locked centre-only 7-scale TTA"

    (out_dir / "result.json").write_text(json.dumps({
        "experiment": "4crop_tta",
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "centre_only_test_acc_pct": round(centre_ensemble, 4),
        "five_crop_test_acc_pct": round(five_ensemble, 4),
        "delta_five_crop_pp": round(delta_five, 4),
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\n[exp:4crop_tta] {verdict}", flush=True)


if __name__ == "__main__":
    main()
