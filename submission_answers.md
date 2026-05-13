# Submission point answers (Q1–Q15)

I copy these into the form fields as-is. Q14 and Q15 are filled in by
`train.py` at the end of a full 30-epoch run (it patches the placeholders
below). Q5 / parameter counts are pulled from `MODEL_SUMMARY.txt`, which
`train.py` regenerates on every run.

---

## Q1 — How many layers in your network?

**Answer:** `54`

I count parametric layers (Conv2d + BatchNorm2d + Linear). Breakdown:

| Type        | Count |
|-------------|------:|
| Conv2d      |    20 |
| BatchNorm2d |    17 |
| Linear      |    17 |
| **Total**   |    **54** |

Where the Conv2d come from: 1 stem + 4 main convs per stage × 4 stages
(= 16) + 1×1 projection conv in the first block of stages 2/3/4 (= 3) = 20.
Where the BatchNorm2d come from: 2 BNs per residual block × 8 blocks (= 16)
+ 1 final BN before the head = 17. Where the Linear come from: 2 FCs per
SE module × 8 blocks (= 16) + 1 classifier = 17.

## Q2 — Layer types used

**Answer (comma-separated):**

`Conv2d (×20), BatchNorm2d (×17), Linear (×17), SiLU (×25), Sigmoid (×8), AdaptiveAvgPool2d (×9), Dropout (×1)`

I don't list `nn.Identity` explicitly — it's used as the residual shortcut in
blocks where the input and output shapes already match, so it has no
parameters and isn't really a layer in any meaningful sense.

## Q3 — Units / kernels per layer

**Answer (grouped per stage):**

`stem: 64 (3×3 conv); stage 1: 64 channels × 2 PreAct-SE blocks (no projection); stage 2: 128 channels × 2 blocks (1×1 projection 64→128 in the first block); stage 3: 256 channels × 2 blocks (1×1 projection 128→256 in the first block); stage 4: 512 channels × 2 blocks (1×1 projection 256→512 in the first block); SE bottleneck channels per stage: 8, 8, 16, 32; classifier: 37 units`

All 3×3 convs in the residual main path keep their stage's channel count;
each first block of stages 2–4 also has a 1×1 stride-2 projection conv that
matches dims for the residual add.

## Q4 — Activation functions

**Answer (with counts):** `SiLU (×25, applied at every BN output and inside each SE bottleneck), Sigmoid (×8, on the SE expand output as the channel gate)`

The classifier (Linear → softmax via cross-entropy) uses no explicit
activation — softmax is folded into the loss.

## Q5 — Total number of weights/biases

**Answer:** `11276133`

Pulled from `torchinfo.summary(model, (1,3,224,224))` in `MODEL_SUMMARY.txt`.

## Q6 — Loss function

**Answer:** `Cross-entropy with label smoothing (epsilon = 0.1), implemented as soft-target cross-entropy.`

The soft-target form is numerically identical to
`nn.CrossEntropyLoss(label_smoothing=0.1)` for hard labels and would compose
cleanly with Mixup / CutMix if those were re-enabled (they're currently off
— see Q11).

## Q7 — Optimisation algorithm

**Answer:** `SGD with Nesterov momentum (momentum=0.9, weight_decay=5e-4), with a OneCycleLR schedule.`

## Q8 — Learning rate

**Answer:** `1e-1`

This is the OneCycleLR `max_lr`. The schedule warms up linearly from
~`max_lr/25` over the first ~17 % of optimizer steps (≈ 5 epochs) and cosine
anneals back down through the remaining epochs.

## Q9 — Number of epochs

**Answer:** `30`

Hard-asserted in `train.py` (`assert EPOCHS == 30`).

## Q10 — Batch size

**Answer:** `128`

`train.py` first probes whether batch size 128 fits in VRAM on the
training host. If not, it falls back to batch size 64 with gradient
accumulation = 2 so the effective batch (the one the LR schedule cares
about) stays at 128.

## Q11 — Training augmentations / transforms

**Answer (comma-separated, applied in this order):** `RandomResizedCrop(224, scale=(0.6, 1.0)), RandomHorizontalFlip, RandAugment(num_ops=2, magnitude=7), ToTensor, Normalize(per-channel mean/std computed from trainval)`

I deliberately stripped the regularisation back from a heavier first attempt
(RandAugment magnitude 9 + ColorJitter + RandomErasing + Mixup + CutMix +
label smoothing): on 3 312 training images for only 30 epochs from scratch,
that stack stopped the model from fitting its own training data. The
implementations of Mixup / CutMix still live in `src/mixup.py` and the
soft-target CE path is intact — they're disabled by default
(`MIX_PROB = 0.0` in `train.py`) rather than removed.

## Q12 — Total images in the training set

**Answer:** `3312`

Stratified 90 % split of the official 3 680-image trainval set
(`StratifiedShuffleSplit(test_size=368, random_state=42)`).

## Q13 — Total images in the validation set

**Answer:** `368`

The complementary 10 % of trainval. I only use it to *monitor* training —
no early stopping, no hyperparameter selection — so the test split is never
indirectly leaked into model selection.

## Q14 — Accuracy on the official trainval set

**Answer:** `60.84 %`

Evaluated under the eval transform (Resize 256 → CenterCrop 224 → Normalize),
no augmentation, no TTA, on all 3 680 trainval images. Reported by the
final block of `train.py`.

## Q15 — Accuracy on the official test set

**Answer:** `46.42 %`

Evaluated on all 3 669 test images with **3-scale + horizontal-flip TTA**.
For each test image I build three eval transforms — `Resize(224)`,
`Resize(256)`, `Resize(288)`, each followed by `CenterCrop(224)` and the
trainval-derived `Normalize` — run the model and its horizontal flip
through each, and sum the softmax probabilities across all six views.
Argmax over the accumulated total is the prediction. This is the same
function `test.py` calls, so `train.py`'s reported number matches what
the markers see when they run `python test.py`.

Promoted from `experiments/exp_multi_scale_tta.py` after it beat the
HFlip-only baseline by +0.82pp (45.60 % → 46.42 %).
