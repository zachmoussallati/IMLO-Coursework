# Experiments

A sandbox for trying variants on top of the locked 45.60 % test baseline.

## Ground rules

- Everything outside `experiments/` is **frozen** at the verified-reproducible
  recipe (`model.pth` scores 45.60 % via `test.py`). Nothing in `src/`,
  `train.py`, or `test.py` should be edited from here.
- Each experiment is one self-contained Python file under `experiments/`.
- Outputs go to `experiments/<experiment_name>/` (model weights, result JSON,
  any logs). That directory is created on demand.
- Each experiment prints a comparison against the baseline at the end and
  writes a `result.json` so I can scan all of them later.

## Workflow

1. **Add** an experiment file under `experiments/` (see existing examples).
2. **Run** it from the repo root, e.g.
   `python experiments/exp_multi_scale_tta.py`.
3. **Read** `experiments/<name>/result.json` — it has `q14_trainval`,
   `q15_test`, `baseline_test`, `delta_pp`, and a `verdict` field.
4. **Promote** the winner (only if it actually beats baseline by a margin
   larger than the spec's ±3 % reproducibility envelope). See "Promotion"
   below.

## Promotion (only when an experiment beats baseline)

If an experiment's test accuracy is convincingly above 45.60 %:

1. Edit `src/` and/or `test.py` to embed the change (e.g. swap the train
   transform, swap the TTA function).
2. If retraining was involved, copy `experiments/<name>/model.pth` to the
   repo root as the new `model.pth`.
3. Run `python train.py` to refresh `submission_answers.md`'s Q14/Q15 and
   `MODEL_SUMMARY.txt` for the new recipe (skip this if no retrain happened —
   just edit Q15 in `submission_answers.md`).
4. Update `ARCHITECTURE.md` to reflect the new recipe + new accuracy band,
   and keep the failed-experiment narrative as part of §3 / §4.
5. Commit with a `promote: <experiment name>` message. The previous
   baseline stays in git history so we can always revert.

## Experiments tried

Run `python experiments/summarise.py` to regenerate the leaderboard.

| File | What it tries | Cost | Q15 | Δ vs locked | Verdict |
|---|---|---|---|---|---|
| `exp_multi_scale_tta.py` | 3-scale + HFlip TTA on the existing `model.pth` (no retraining) | ~30 s | 46.42 % | +0.82 vs 45.60 % | **promoted** |
| `exp_tta_search.py` | Grid over 7 TTA configurations on the same weights | ~5 min | 46.74 % | +0.32 vs 46.42 % | **promoted** (dense_7scale winner) |
| `exp_ema.py` | EMA weights, decay=0.999 | ~25 min | — | — | aborted at epoch 8 (decay too slow for a 30-epoch budget — EMA still near random) |
| `exp_ema_fast.py` | EMA weights, decay=0.99 | ~25 min | 43.17 % | −3.57 | worse than baseline |
| `exp_trivial_augment.py` | `RandAugment` → `TrivialAugmentWide` | ~25 min | 42.76 % * | −3.98 | worse than baseline |
| `exp_smaller_model.py` | Widths `(32, 64, 128, 256)`, ~2.8 M params | ~25 min | 38.89 % | −7.53 | worse than baseline |
| `exp_possible_improvement.py` | Faithful replay of an alt recipe shared with me (post-act ResNet, LeakyReLU, AdamW, ImageNet stats, no train/val split) | ~10 min | 41.40 % * | −5.34 | worse than baseline |

\*\* The alt recipe scored **71.01 %** on the unaugmented full trainval
(higher than the locked 60.84 %) but **43.25 %** single-pass on test and
**41.40 %** with the live 7-scale + HFlip TTA. The "close to 60 %" claim
that came with this recipe was almost certainly the *training*
accuracy in its print log (it reaches `train_acc=60.76 %` at epoch 30),
not test accuracy. The 28 pp train-test gap is a classic small-data
overfit driven by the lighter augmentation (just HFlip + Rotation +
ColorJitter), lower label smoothing (0.05), and training on the full
3 680-image trainval with no val signal. Our 7-scale TTA actively
*hurts* this model (41.40 % vs 43.25 % single-pass) because the alt
recipe trains with `Resize((224, 224))` which stretches images to a
square — TTA's aspect-preserving `Resize(scale) → CenterCrop(224)`
gives the model inputs it has never seen.

\* TrivialAugment's `result.json` records the HFlip-only Q15 (42.25 %) the
script measured directly. I followed up by evaluating
`experiments/trivial_augment/model.pth` through the live 7-scale + HFlip
TTA path, which gave 42.76 % — the apples-to-apples number against the
46.74 % locked baseline.

## What I learned

- **TTA is the cheapest improvement axis.** The two TTA promotions stack
  for +1.14 pp over the original HFlip-only baseline with zero
  retraining cost. Each step is fully deterministic.
- **EMA needs more training steps than 30 epochs gives.** Both
  reasonable decays failed: 0.999 has half-life ~700 steps so EMA still
  carries random-init weight at evaluation time, and 0.99 lags the
  training model by ~3 epochs which is enough to land outside the final
  basin.
- **TrivialAugmentWide is too aggressive untuned for this dataset.**
  `RandAugment(magnitude=7)` is an already-tuned operating point;
  swapping in untuned TrivialAugment costs 4 pp.
- **The model isn't overparameterised.** Cutting channels by 2× (and
  therefore params by ~4×) cost 7.5 pp on test. The 30-epoch / 3.3 K
  image budget bottlenecks on data, not on capacity.
- **10-crop hurts on this dataset.** Pets are centred subjects, so
  corner crops cut the subject. Multi-scale centre-crop ensemble is the
  right TTA shape here.

## Candidate experiments I considered but didn't run

Most of these would likely move the needle by ≤ 1 pp on top of 46.74 %
based on the priors built up from the experiments above:

- **Stochastic depth** in residual branches.
- **AdamW** with cosine LR.
- **Cosine annealing warm restarts** (2-3 hot cycles in 30 epochs).
- **Lower / higher weight decay** (1e-4, 1e-3).
- **Different label smoothing** (0.05, 0.2).
- **Bigger model** (`(96, 192, 384, 768)` widths) — counter to the
  smaller-model finding, but worth trying given the underfit signal.
- **Channels-last memory format** — pure runtime, no accuracy change.
