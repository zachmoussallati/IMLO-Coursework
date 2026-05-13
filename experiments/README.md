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

## Current experiments

| File | What it tries | Cost | Result |
|---|---|---|---|
| `exp_multi_scale_tta.py` | 3-scale + HFlip TTA on the existing `model.pth` (no retraining) | ~30 s | see `multi_scale_tta/result.json` |
| `exp_trivial_augment.py` | Swap `RandAugment` for `TrivialAugmentWide` and retrain 30 epochs | ~25 min | see `trivial_augment/result.json` |

## Candidate experiments I might add later

- **Smaller model variant**. 11 M params is heavy for 3.3 K training images;
  a `[32, 64, 128, 256]` channel ladder cuts to ~3 M params and could
  generalise better.
- **EMA model weights**. Maintain a slow exponential moving average of
  weights during training and evaluate that copy at the end. Often a free
  +0.5–1 pp.
- **Cosine annealing warm restarts** (`CosineAnnealingWarmRestarts`).
  Multiple "hot" learning phases in 30 epochs.
- **Channels-last memory format**. PyTorch's NHWC layout on CUDA tends to
  speed training up but does not change accuracy — useful only for runtime,
  not score.

Each of these gets its own `exp_<name>.py` when I get to it.
