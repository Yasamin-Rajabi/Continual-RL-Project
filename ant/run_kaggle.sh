#!/usr/bin/env bash
# run_kaggle.sh — one entrypoint for the whole pipeline, meant to be called
# from Kaggle notebook cells.
#
# Why a single script: Kaggle's weekly GPU quota is limited (30 hrs) and
# sessions get interrupted. Both run_continual_benchmark.py and
# scratch_baselines.py already check checkpoint_complete() and skip finished
# work, so re-running this script just resumes whatever wasn't finished.
#
# Usage from a notebook cell:
#   !bash run_kaggle.sh setup
#   !bash run_kaggle.sh sanity
#   !bash run_kaggle.sh pretrain hc
#   !bash run_kaggle.sh baselines hc
#   !bash run_kaggle.sh continual hc
#   !bash run_kaggle.sh all hc          # the four steps above, back to back
#
# For Ant, calibrate the velocity band FIRST -- absolute HalfCheetah targets
# are unreachable on Ant and collapse several tasks into one:
#   !bash run_kaggle.sh calibrate ant     # then paste the result into tasks.py
#   !bash run_kaggle.sh all ant
#
# If a session drops mid-step, just re-run the same command.

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
mkdir -p "$OUT/pretrained_encoders" "$OUT/agents" "$OUT/plots" "$OUT/logs"

# --------------------------------------------------------------------------
# Configuration — edit values here, not on the command line
# --------------------------------------------------------------------------
SEEDS="1 2 3"
TOTAL_TIMESTEPS=50000          # vs. the 300000 default; see README section 6
PRETRAIN_STEPS_PER_TASK=100000
PRETRAIN_EPOCHS=30

HC_SUITE="halfcheetah_vel"
ANT_SUITE="ant_vel"

# --------------------------------------------------------------------------
suite_of() { [ "$1" = "ant" ] && echo "$ANT_SUITE" || echo "$HC_SUITE"; }
enc_dir()  { echo "$OUT/pretrained_encoders/$1"; }

step_setup() {
    echo ">>> setup: installing dependencies"
    pip install -q -r requirements.txt --break-system-packages 2>/dev/null || \
    pip install -q -r requirements.txt
    echo "OK"
}

step_sanity() {
    echo ">>> sanity: structural checks for pool and tasks (no GPU needed)"
    python3 sanity_check_pool.py
    python3 tasks.py
    python3 tasks.py --check
    echo "OK"
}

step_calibrate() {
    local fam="$1"
    if [ "$fam" != "ant" ]; then
        echo "calibrate only applies to ant (HalfCheetah targets are already set)"
        return 0
    fi
    echo ">>> calibrate[ant]: measuring reachable forward velocity"
    echo "    After this finishes, paste the printed _ANT_V_MAX into tasks.py"
    echo "    and re-run 'sanity' before continuing."
    python3 calibrate_ant.py \
        --total-timesteps 150000 \
        --save-dir "$OUT/agents/ant_calibrate" \
        --analysis-root "$OUT/logs/ant_calibrate" \
        2>&1 | tee "$OUT/logs/calibrate_ant.log"
}

step_pretrain() {
    local fam="$1" suite; suite="$(suite_of "$fam")"
    local out; out="$(enc_dir "$fam")"
    if [ -f "$out/fc.pt" ]; then
        echo ">>> pretrain[$fam]: fc.pt already exists, skipping ($out)"
        return 0
    fi
    echo ">>> pretrain[$fam]: TD-JEPA on $suite"
    if [ "$fam" = "ant" ]; then
        # These assume the DEFAULT _ANT_V_MAX = 3.3. If you calibrated a
        # different v_max, rescale them: pretrain velocities must fall BETWEEN
        # the benchmark targets (never equal to one), held-out ones must BE
        # benchmark targets.
        python3 tdjepa_pretrain.py \
            --task-suite "$suite" \
            --pretrain-velocities 0.66 1.98 3.30 \
            --heldout-velocities 0.33 1.65 2.97 \
            --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
            --out "$out" 2>&1 | tee "$OUT/logs/pretrain_${fam}.log"
    else
        python3 tdjepa_pretrain.py \
            --task-suite "$suite" \
            --pretrain-velocities 0.75 1.75 2.75 \
            --heldout-velocities 0.5 2.0 3.0 \
            --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
            --out "$out" 2>&1 | tee "$OUT/logs/pretrain_${fam}.log"
    fi
    echo "OK -> $out/fc.pt"
}

step_baselines() {
    local fam="$1" suite; suite="$(suite_of "$fam")"
    local enc; enc="$(enc_dir "$fam")/fc.pt"
    [ -f "$enc" ] || { echo "ERROR: run 'pretrain $fam' first"; exit 1; }
    echo ">>> baselines[$fam]: scratch baselines (same encoder as the continual runs)"
    python3 scratch_baselines.py \
        --task-suites "$suite" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pretrained-encoder "$enc" --encoder-linear-out \
        --analysis-root "$OUT/logs/scratch_${fam}" \
        2>&1 | tee -a "$OUT/logs/baselines_${fam}.log"
    echo "OK"
}

step_continual() {
    local fam="$1" suite; suite="$(suite_of "$fam")"
    local enc; enc="$(enc_dir "$fam")/fc.pt"
    [ -f "$enc" ] || { echo "ERROR: run 'pretrain $fam' first"; exit 1; }
    echo ">>> continual[$fam]: full benchmark run (S0 baseline + S4 TD-JEPA)"

    # S0 -- baseline, no pretrained encoder
    python3 run_continual_benchmark.py \
        --task-suites "$suite" --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$OUT/agents/${fam}_S0" \
        --plots-root "$OUT/plots/${fam}_S0" \
        2>&1 | tee -a "$OUT/logs/continual_${fam}_S0.log"

    # S4 -- TD-JEPA
    python3 run_continual_benchmark.py \
        --task-suites "$suite" --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pretrained-encoder "$enc" --encoder-linear-out \
        --save-root "$OUT/agents/${fam}_S4" \
        --plots-root "$OUT/plots/${fam}_S4" \
        2>&1 | tee -a "$OUT/logs/continual_${fam}_S4.log"
    echo "OK"
}

step_all() {
    local fam="$1"
    step_pretrain "$fam"
    step_baselines "$fam"
    step_continual "$fam"
}

case "${1:-}" in
    setup)      step_setup ;;
    sanity)     step_sanity ;;
    calibrate)  step_calibrate "${2:?need hc or ant}" ;;
    pretrain)   step_pretrain "${2:?need hc or ant}" ;;
    baselines)  step_baselines "${2:?need hc or ant}" ;;
    continual)  step_continual "${2:?need hc or ant}" ;;
    all)        step_all "${2:?need hc or ant}" ;;
    *)
        echo "Usage: bash run_kaggle.sh {setup|sanity|calibrate|pretrain|baselines|continual|all} [hc|ant]"
        exit 1
        ;;
esac
