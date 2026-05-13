"""Experiment: multi-scale horizontal-flip TTA against the locked model.pth.

The locked baseline does 2-pass TTA: original + HFlip, averaged softmax.
This experiment runs the same model under three Resize scales (224 / 256 /
288 then CenterCrop 224) with HFlips of each, so 6 forward passes per
image, and averages the softmax probabilities. No retraining - just a
different inference path on the same weights.

If the resulting Q15 beats the 45.60 % baseline by more than the spec's
+/-3 % reproducibility envelope (i.e. > ~46.5 %), it's worth promoting the
TTA function into `src/train_loop.py:evaluate_with_tta` so test.py picks it
up. See `experiments/README.md` for the promotion checklist.

Run from the repo root:
    python experiments/exp_multi_scale_tta.py

Writes:
    experiments/multi_scale_tta/result.json
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

from src.data import DATA_ROOT, STATS_PATH, IMAGE_SIZE
from src.model import build_model


EXPERIMENT_NAME = "multi_scale_tta"
EXPERIMENT_DIR = Path(__file__).resolve().parent / EXPERIMENT_NAME
BASELINE_TEST_PCT = 45.60
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pth"

# why: three Resize sizes around the train/eval default of 256.
# 224 = no centre crop trimming, full-image view at native train size.
# 256 = matches train-time eval transform (this is what the model "knows").
# 288 = ~13% zoomed-in centre crop, gives the classifier a closer look.
SCALES = (224, 256, 288)
BATCH_SIZE = 128
NUM_WORKERS = 2


def _build_eval_transform_for_scale(scale: int, mean, std) -> T.Compose:
    """Resize(scale) -> CenterCrop(224) -> Normalize."""
    return T.Compose(
        [
            T.Resize(scale),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


@torch.no_grad()
def _accumulate_probs(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one pass of the model and its HFlip; return summed softmax probs + labels."""
    sum_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            logits_flip = model(torch.flip(images, dims=[3]))
        probs = F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)
        sum_probs.append(probs.cpu())
        all_labels.append(labels)
    return torch.cat(sum_probs, dim=0), torch.cat(all_labels, dim=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} missing — run python train.py first.")

    stats = json.loads(Path(STATS_PATH).read_text(encoding="utf-8"))

    model = build_model(num_classes=stats["num_classes"]).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # Accumulate softmax-prob ensemble across scales.
    ensembled = None
    reference_labels = None
    per_scale_acc: dict[str, float] = {}
    for scale in SCALES:
        eval_t = _build_eval_transform_for_scale(scale, stats["mean"], stats["std"])
        test_ds = OxfordIIITPet(
            root=DATA_ROOT,
            split="test",
            target_types="category",
            download=True,
            transform=eval_t,
        )
        loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        probs_sum, labels = _accumulate_probs(model, loader, device)
        # why: probs_sum is the sum of (original_softmax + flip_softmax) per
        # image at this scale. I keep summing across scales so the final
        # argmax is over the total accumulated probability.
        if ensembled is None:
            ensembled = probs_sum.clone()
            reference_labels = labels
        else:
            assert torch.equal(labels, reference_labels), \
                "label order changed across loaders"
            ensembled += probs_sum

        # per-scale accuracy (already includes HFlip): just for diagnostics
        per_scale_correct = (probs_sum.argmax(dim=1) == labels).sum().item()
        per_scale_acc[str(scale)] = per_scale_correct / labels.numel() * 100
        print(
            f"[exp:{EXPERIMENT_NAME}] scale={scale}  "
            f"HFlip-pair acc = {per_scale_acc[str(scale)]:.2f}%"
        )

    assert ensembled is not None and reference_labels is not None
    correct = (ensembled.argmax(dim=1) == reference_labels).sum().item()
    multiscale_acc_pct = correct / reference_labels.numel() * 100

    delta = multiscale_acc_pct - BASELINE_TEST_PCT
    margin = 0.5  # I treat anything below this as noise
    if multiscale_acc_pct > BASELINE_TEST_PCT + margin:
        verdict = "BEATS BASELINE"
    elif multiscale_acc_pct < BASELINE_TEST_PCT - margin:
        verdict = "WORSE THAN BASELINE"
    else:
        verdict = "within noise"

    result = {
        "experiment": EXPERIMENT_NAME,
        "scales": list(SCALES),
        "per_scale_test_acc_pct": per_scale_acc,
        "ensembled_test_acc_pct": round(multiscale_acc_pct, 2),
        "baseline_test_acc_pct": BASELINE_TEST_PCT,
        "delta_pp": round(delta, 2),
        "verdict": verdict,
    }
    (EXPERIMENT_DIR / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(f"[exp:{EXPERIMENT_NAME}] ensembled test acc = {multiscale_acc_pct:.2f}%")
    print(
        f"[exp:{EXPERIMENT_NAME}] vs baseline {BASELINE_TEST_PCT:.2f}% : "
        f"{'+' if delta >= 0 else ''}{delta:.2f}pp  ->  {verdict}"
    )


if __name__ == "__main__":
    main()
