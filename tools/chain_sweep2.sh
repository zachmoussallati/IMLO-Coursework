#!/usr/bin/env bash
# Chain runner v2 with hang detection.
#
# For each experiment:
#   1. Launch and start a wall-clock timer
#   2. Poll the log file every minute for "epoch N/30" lines
#   3. If no progress in 5 minutes (likely DataLoader hang on Windows),
#      kill the process tree and move to the next experiment
#   4. Hard cap at 45 minutes per experiment.

set -uo pipefail
cd "$(dirname "$0")/.."

PY="C:/Users/zacha/anaconda3/envs/imlo-coursework-gpu/python.exe"
export PYTHONUNBUFFERED=1

# Skip SE-r=8 (hung previously, was looking weak).
EXPERIMENTS=(
    experiments/exp_ablation_blurpool.py
    experiments/exp_cap_88_346.py
    experiments/exp_cap_80_349.py
    experiments/exp_cap_resnet101.py
    experiments/exp_cap_88_349.py
    experiments/exp_cap_96_346.py
)

# Hard cap removed - BlurPool reached ep22/30 of training before being killed
# at 45 min because per-epoch was ~2 min not the ~17s of the basic blocks.
# The hang detection alone is enough: if the log stops growing for 7 min,
# the process really is hung (Windows DataLoader deadlock).
HARD_CAP_SEC=14400     # 4 hours - effectively no cap, just a safety net
HANG_SEC=420           # if log unchanged for 7 min, kill

for script in "${EXPERIMENTS[@]}"; do
    log="experiments/$(basename "$script" .py).log"
    echo "[chain] $(date) starting $script"
    rm -f "$log"
    touch "$log"

    "$PY" "$script" > "$log" 2>&1 &
    pid=$!

    start_t=$(date +%s)
    last_size=0
    last_size_t=$start_t

    while kill -0 "$pid" 2>/dev/null; do
        now=$(date +%s)
        elapsed=$((now - start_t))
        if [ $elapsed -gt $HARD_CAP_SEC ]; then
            echo "[chain] $(date) HARD_CAP hit for $script; killing tree"
            taskkill //F //T //PID $pid 2>/dev/null || kill -9 $pid 2>/dev/null
            break
        fi
        sz=$(stat -c %s "$log" 2>/dev/null || echo 0)
        if [ "$sz" -gt "$last_size" ]; then
            last_size=$sz
            last_size_t=$now
        fi
        idle=$((now - last_size_t))
        if [ $idle -gt $HANG_SEC ]; then
            echo "[chain] $(date) HANG_DETECT ($idle s idle) for $script; killing tree"
            taskkill //F //T //PID $pid 2>/dev/null || kill -9 $pid 2>/dev/null
            break
        fi
        sleep 30
    done
    wait $pid 2>/dev/null || true
    echo "[chain] $(date) finished $script  (elapsed $((($(date +%s) - start_t)/60))m)"
done

echo "[chain] All done."
