#!/usr/bin/env bash
# Run a sequence of ablation experiments in order. Each writes its own
# result.json under experiments/<name>/; this script just orchestrates
# the order and captures stdout to a single log per experiment.
#
# Usage:
#   bash tools/chain_sweep.sh
#
# Order is from cheap-and-most-likely-to-help to expensive-and-speculative.
# Bail out early on the first non-zero exit (so user can ctrl-C between runs).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="C:/Users/zacha/anaconda3/envs/imlo-coursework-gpu/python.exe"
export PYTHONUNBUFFERED=1

run() {
    local script="$1"
    local log="experiments/$(basename "$script" .py).log"
    echo "=========================================="
    echo "[chain_sweep] $(date) starting $script"
    echo "=========================================="
    "$PY" "$script" > "$log" 2>&1
    echo "[chain_sweep] $(date) finished $script"
}

run experiments/exp_ablation_se_r8.py
run experiments/exp_ablation_blurpool.py
run experiments/exp_cap_88_346.py
run experiments/exp_cap_80_349.py
run experiments/exp_cap_resnet101.py
run experiments/exp_cap_88_349.py
run experiments/exp_cap_96_346.py

echo "[chain_sweep] All experiments done."
