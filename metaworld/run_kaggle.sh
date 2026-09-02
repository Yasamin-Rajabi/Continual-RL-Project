#!/usr/bin/env bash
# run_kaggle.sh -- one entrypoint for the continual Meta-World benchmark,
# designed to be split across TWO Kaggle accounts over two days.
#
# Kaggle gives 30 GPU-hours/week per account and caps a session at ~12h.
# The full plan is ~16-40 GPU-hours depending on throughput (see README), so it
# is split by CONDITION: each account runs two of the four conditions. Both
# accounts need the same pilot + scratch baselines, which are cheap.
#
#   ACCOUNT A                          ACCOUNT B
#   ---------                          ---------
#   bash run_kaggle.sh setup           bash run_kaggle.sh setup
#   bash run_kaggle.sh pilot           (skip -- copy A's verdict)
#   bash run_kaggle.sh baselines       bash run_kaggle.sh baselines
#   bash run_kaggle.sh pretrain        bash run_kaggle.sh pretrain
#   bash run_kaggle.sh run A           bash run_kaggle.sh run B
#
# Then merge: download both accounts' agents/ + runs/ into one directory and
# run `bash run_kaggle.sh report`.
#
# Everything is resumable: run_continual_benchmark.py and scratch_baselines.py
# both check checkpoint_complete() and skip finished work, so if a session
# drops, re-run the identical command.

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
mkdir -p "$OUT"/{pretrained_encoders,agents,plots,logs,runs}

# --------------------------------------------------------------------------
# Configuration -- edit here, not on the command line
# --------------------------------------------------------------------------
SUITE="mw_easy4"
SEEDS="1 2 3"
TOTAL_TIMESTEPS=150000
SEQUENCE="0 2 3 1 0 3 2 1"       # see tasks.py for the design rationale
POOL_SIZE=4                      # = number of distinct tasks; see tasks.py

PRETRAIN_STEPS=60000
PRETRAIN_EPOCHS=30

ENC="$OUT/pretrained_encoders/mw/fc.pt"

# --------------------------------------------------------------------------
step_setup() {
    echo ">>> setup"
    pip install -q -r requirements.txt --break-system-packages 2>/dev/null || \
    pip install -q -r requirements.txt
    # Meta-World is not on PyPI; install from source.
    python3 -c "import metaworld" 2>/dev/null || \
    pip install -q "git+https://github.com/Farama-Foundation/Metaworld.git@master" \
        --break-system-packages 2>/dev/null || \
    pip install -q "git+https://github.com/Farama-Foundation/Metaworld.git@master"
    python3 tasks.py
    echo "OK"
}

step_check() {
    echo ">>> check: builds every task and asserts constant shapes + info keys"
    python3 sanity_check_pool.py
    python3 tasks.py --check
    echo "OK"
}

step_pilot() {
    # ~1 GPU-hour. Decides whether the whole plan is viable -- do not skip it
    # on the first account. Its from-scratch runs double as FT baselines.
    echo ">>> pilot: is 150k enough for these four tasks?"
    python3 pilot_check.py \
        --task-suite "$SUITE" --steps "$TOTAL_TIMESTEPS" --seed 1 \
        --runs-root "$OUT/runs" \
        --save-dir "$OUT/agents/pilot" \
        --analysis-root "$OUT/logs/pilot" \
        2>&1 | tee "$OUT/logs/pilot.log"
}

step_pretrain() {
    if [ -f "$ENC" ]; then
        echo ">>> pretrain: fc.pt already exists, skipping"
        return 0
    fi
    echo ">>> pretrain: TD-JEPA shared encoder on held-out tasks"
    python3 tdjepa_pretrain.py \
        --heldout-suite "$SUITE" \
        --steps-per-task "$PRETRAIN_STEPS" --epochs "$PRETRAIN_EPOCHS" \
        --out "$OUT/pretrained_encoders/mw" \
        2>&1 | tee "$OUT/logs/pretrain.log"
}

step_baselines() {
    echo ">>> baselines: from-scratch SAC per task (denominator for FT)"
    python3 scratch_baselines.py \
        --task-suites "$SUITE" \
        --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --runs-root "$OUT/runs" \
        --save-root "$OUT/agents/scratch" \
        --analysis-root "$OUT/logs/scratch" \
        2>&1 | tee -a "$OUT/logs/baselines.log"
    echo "OK"
}

# Conditions are split across accounts. --condition-index is 1-BASED in
# run_continual_benchmark.py (0 would mean "run all four"):
#   1 baseline, 2 distil_only, 3 weight_only, 4 combined.
step_run() {
    local part="${1:-A}"
    local idx
    case "$part" in
        A) idx="1 2" ;;
        B) idx="3 4" ;;
        all) idx="1 2 3 4" ;;
        *) echo "part must be A, B or all"; exit 1 ;;
    esac

    for i in $idx; do
        echo ">>> continual: condition index $i"
        python3 run_continual_benchmark.py \
            --task-suites "$SUITE" \
            --seeds $SEEDS \
            --task-sequence $SEQUENCE \
            --total-timesteps "$TOTAL_TIMESTEPS" \
            --pool-size "$POOL_SIZE" \
            --condition-index "$i" \
            --runs-root "$OUT/runs" \
            --save-root "$OUT/agents/continual" \
            --plots-root "$OUT/plots" \
            2>&1 | tee -a "$OUT/logs/continual_cond${i}.log"
    done
    echo "OK"
}

# Same, but with the TD-JEPA encoder. Run only after `run` has finished, and
# only if the pilot said the budget is sufficient.
step_run_tdjepa() {
    local part="${1:-A}"
    local idx
    case "$part" in
        A) idx="1 2" ;;
        B) idx="3 4" ;;
        all) idx="1 2 3 4" ;;
        *) echo "part must be A, B or all"; exit 1 ;;
    esac
    [ -f "$ENC" ] || { echo "ERROR: run 'pretrain' first"; exit 1; }

    for i in $idx; do
        echo ">>> continual + TD-JEPA encoder: condition index $i"
        python3 run_continual_benchmark.py \
            --task-suites "$SUITE" \
            --seeds $SEEDS \
            --task-sequence $SEQUENCE \
            --total-timesteps "$TOTAL_TIMESTEPS" \
            --pool-size "$POOL_SIZE" \
            --condition-index "$i" \
            --pretrained-encoder "$ENC" --encoder-linear-out \
            --runs-root "$OUT/runs" \
            --save-root "$OUT/agents/continual_tdjepa" \
            --plots-root "$OUT/plots_tdjepa" \
            2>&1 | tee -a "$OUT/logs/continual_tdjepa_cond${i}.log"
    done
    echo "OK"
}

# After merging both accounts' runs/ and agents/ into one directory.
step_report() {
    echo ">>> report: metrics + plots over all conditions"
    python3 run_continual_benchmark.py \
        --task-suites "$SUITE" \
        --seeds $SEEDS \
        --task-sequence $SEQUENCE \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pool-size "$POOL_SIZE" \
        --skip-training \
        --runs-root "$OUT/runs" \
        --save-root "$OUT/agents/continual" \
        --plots-root "$OUT/plots" \
        2>&1 | tee "$OUT/logs/report.log"
    python3 estimate_timing.py --plots-root "$OUT/plots" || true
    echo "OK -> $OUT/plots"
}

case "${1:-}" in
    setup)       step_setup ;;
    check)       step_check ;;
    pilot)       step_pilot ;;
    pretrain)    step_pretrain ;;
    baselines)   step_baselines ;;
    run)         step_run "${2:-A}" ;;
    run-tdjepa)  step_run_tdjepa "${2:-A}" ;;
    report)      step_report ;;
    *)
        echo "Usage: bash run_kaggle.sh {setup|check|pilot|pretrain|baselines|run|run-tdjepa|report} [A|B|all]"
        exit 1
        ;;
esac
