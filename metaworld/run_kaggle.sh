#!/usr/bin/env bash
# Kaggle entrypoint for the continual Meta-World pipeline.
#
# DESIGN GOAL: every idea is a SEPARATE stage, so no single Kaggle session has
# to fit the whole plan inside the 12 h wall, and stages can be split freely
# across two accounts.
#
#   smoke        ~10 min   does the pipeline run end to end at all
#   pilot        ~1 h      do these tasks learn anything in 150k steps
#   baselines    ~1.5 h    from-scratch SAC per task (denominator for FT)
#   pretrain     ~0.5 h    TD-JEPA shared encoder (idea: shared layer)
#   cond N       varies    ONE of the four CKA conditions (idea: fusion/merge)
#   report       minutes   metrics + plots over whatever is present
#
# Conditions (--condition-index is 1-BASED; 0 would mean all four):
#   1 baseline     classic_cka + cosine + arithmetic merge
#   2 distil_only  classic_cka + symmetric-KL + policy distillation
#   3 weight_only  weight_delta + cosine + arithmetic merge
#   4 combined     weight_delta + symmetric-KL + policy distillation
#
# S0 vs S4 is the ENCODER ablation, orthogonal to the four conditions:
#   S0 = no TD-JEPA pretraining, root encoder learned on task 0 then frozen
#   S4 = TD-JEPA pretrained encoder, frozen from task 0
# Both use the same linear-output encoder/critic so pretraining is the only
# difference between them.
#
# SUGGESTED SPLIT ACROSS TWO ACCOUNTS
#   Account A: smoke, pilot, baselines s0, cond s0 1, cond s0 2
#   Account B: baselines s0, cond s0 3, cond s0 4
#   Then (optional, second day): pretrain, baselines s4, cond s4 {1..4}
#
# Everything is resumable: run_continual_benchmark.py and scratch_baselines.py
# check run manifests and skip completed work, so re-running a stage after a
# session drop continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
SEEDS="${SEEDS:-1 2 3}"
SCRATCH_SEEDS="${SCRATCH_SEEDS:-101 102 103}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-150000}"
SUITE="${SUITE:-mw_easy4}"
SEQUENCE="${SEQUENCE:-0 2 3 1 0 3 2 1}"
POOL_SIZE="${POOL_SIZE:-4}"
PRETRAIN_STEPS_PER_TASK="${PRETRAIN_STEPS_PER_TASK:-60000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
ENC_DIR="$OUT/pretrained_encoders/mw_tdjepa"

# Pinned Meta-World commit. This is the revision that exposes
# ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, which the v2-era task ids in tasks.py
# rely on. Newer master has moved on and would silently change task semantics.
MW_COMMIT="${MW_COMMIT:-c822f28f582ba1ad49eb5dcf61016566f28003ba}"

mkdir -p "$OUT"/{pretrained_encoders,agents,plots,logs,runs,analysis,scratch_models,analysis_scratch}

cuda_check() {
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
    # is_available() alone can be True on a P100 even when the wheel has no
    # sm_60 kernels. An actual kernel launch catches that configuration.
    x = torch.randn(256, 256, device="cuda"); y = x @ x
    torch.cuda.synchronize()
    print("real CUDA kernel test: OK", float(y.mean()))
PY
}

step_setup() {
    echo ">>> setup"
    # Order matters. Meta-World is installed with --no-deps against a pinned
    # commit, so pip never resolves its (stale) dependency pins and never
    # downgrades torch/numpy/gymnasium underneath us. That means MuJoCo has to
    # be present first, and everything else comes from requirements.txt with
    # the metaworld/mujoco lines filtered out so they cannot fight the pin.
    python3 -m pip install -q "mujoco>=3.0,<4"
    python3 -c "import metaworld" 2>/dev/null || \
        python3 -m pip install -q --no-deps \
            "git+https://github.com/Farama-Foundation/Metaworld.git@${MW_COMMIT}"
    grep -v -E "^\s*(metaworld|mujoco)" requirements.txt > /tmp/reqs_clean.txt
    python3 -m pip install -q -r /tmp/reqs_clean.txt

    python3 -c "import metaworld; print('metaworld OK:', metaworld.__file__)"
    cuda_check
    python3 tasks.py
    # Which Meta-World API actually resolved. metaworld_envs supports three;
    # this is the first thing to check if env construction ever fails.
    python3 -c "from metaworld_envs import make_env; e=make_env('window-close-v2',0); print('API:', e.env.metaworld_api); e.close()"
    echo "OK"
}

step_sanity() {
    echo ">>> sanity (no GPU time)"
    python3 sanity_check_pool.py
    python3 tasks.py --check
    echo "OK"
}

roots_for() {
    local stage="$1"
    echo "$OUT/agents/mw_${stage}|$OUT/runs/mw_${stage}|$OUT/analysis/mw_${stage}|$OUT/scratch_models/mw_${stage}|$OUT/analysis_scratch/mw_${stage}|$OUT/plots/mw_${stage}"
}

encoder_flags() {
    local stage="$1"
    local -n _out=$2
    _out=(--encoder-linear-out --no-train-shared --no-freeze-root-encoder)
    if [ "$stage" = "s4" ]; then
        [ -f "$ENC_DIR/fc.pt" ] || { echo "Missing $ENC_DIR/fc.pt; run 'pretrain' first" >&2; exit 1; }
        _out+=(--pretrained-encoder "$ENC_DIR/fc.pt")
    elif [ "$stage" != "s0" ]; then
        echo "stage must be s0 or s4" >&2; exit 2
    fi
}

# --------------------------------------------------------------------------
# smoke: the cheapest possible end-to-end exercise. 3 positions, 2 tasks,
# 3k steps each, pool_size 2 so a merge actually fires. Catches wiring bugs
# (env API, info keys, merge, retention, metrics) in minutes rather than after
# an hour of real training.
# --------------------------------------------------------------------------
step_smoke() {
    echo ">>> smoke: full pipeline, tiny budget"
    local values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for smoke)"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags s0 extra

    python3 scratch_baselines.py \
        --task-suites mw_smoke2 --seeds 101 \
        --total-timesteps 3000 --eval-every 1000 --num-evals 2 \
        --learning-starts 500 --random-actions-end 1000 \
        --save-root "$scratch" --runs-root "$runs" --analysis-root "$scratch_analysis" \
        "${extra[@]}" 2>&1 | tee "$OUT/logs/mw_smoke_baselines.log"

    python3 run_continual_benchmark.py \
        --task-suites mw_smoke2 --seeds 1 \
        --task-sequence 0 1 0 \
        --total-timesteps 3000 --eval-every 1000 --num-evals 2 \
        --learning-starts 500 --random-actions-end 1000 \
        --pool-size 2 --distill-extra-steps 1000 --test-adapt-steps 200 \
        --scratch-seeds 101 \
        --save-root "$agents" --runs-root "$runs" --analysis-root "$analysis" \
        --scratch-save-root "$scratch" --plots-root "$plots" \
        "${extra[@]}" 2>&1 | tee "$OUT/logs/mw_smoke_continual.log"
    echo "SMOKE OK -> if this passed, the pipeline is wired correctly."
}

# --------------------------------------------------------------------------
# pilot: do the four real tasks learn anything at 150k? One seed, no continual
# machinery. Its from-scratch runs are reusable as FT baselines.
# --------------------------------------------------------------------------
step_pilot() {
    echo ">>> pilot: is $TOTAL_TIMESTEPS enough for $SUITE?"
    python3 pilot_check.py \
        --task-suite "$SUITE" --steps "$TOTAL_TIMESTEPS" --seed 101 \
        --runs-root "$OUT/runs/mw_s0" \
        --save-dir "$OUT/scratch_models/mw_s0" \
        --analysis-root "$OUT/analysis_scratch/mw_s0" \
        2>&1 | tee "$OUT/logs/mw_pilot.log"
}

step_pretrain() {
    if [ -f "$ENC_DIR/fc.pt" ] && [ "${FORCE_PRETRAIN:-0}" != "1" ]; then
        echo ">>> TD-JEPA encoder already exists: $ENC_DIR/fc.pt"
        return
    fi
    rm -rf "$ENC_DIR"
    echo ">>> pretrain: TD-JEPA shared encoder on held-out tasks"
    python3 tdjepa_pretrain.py \
        --heldout-suite "$SUITE" \
        --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
        --out "$ENC_DIR" 2>&1 | tee "$OUT/logs/mw_pretrain.log"
}

step_baselines() {
    local stage="${1:-s0}" values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags "$stage" extra
    echo ">>> baselines [$stage]: from-scratch SAC per task"
    python3 scratch_baselines.py \
        --task-suites "$SUITE" --seeds $SCRATCH_SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$scratch" --runs-root "$runs" --analysis-root "$scratch_analysis" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/mw_${stage}_baselines.log"
}

# --------------------------------------------------------------------------
# cond: ONE condition at a time. This is the unit of work you schedule.
#   bash run_kaggle.sh cond s0 3
# --------------------------------------------------------------------------
step_cond() {
    local stage="${1:-s0}" idx="${2:-}"
    [ -n "$idx" ] || { echo "usage: run_kaggle.sh cond [s0|s4] <1..4>"; exit 2; }
    local values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags "$stage" extra

    echo ">>> condition $idx [$stage] on $SUITE, seeds: $SEEDS"
    python3 run_continual_benchmark.py \
        --task-suites "$SUITE" --seeds $SEEDS \
        --task-sequence $SEQUENCE \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pool-size "$POOL_SIZE" \
        --condition-index "$idx" \
        --scratch-seeds $SCRATCH_SEEDS \
        --save-root "$agents" --runs-root "$runs" --analysis-root "$analysis" \
        --scratch-save-root "$scratch" --plots-root "$plots" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/mw_${stage}_cond${idx}.log"
}

# Report over whatever conditions are already present. Training is skipped, so
# this is safe to run after merging outputs from both accounts.
step_report() {
    local stage="${1:-s0}" values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags "$stage" extra
    echo ">>> report [$stage]"
    python3 run_continual_benchmark.py \
        --task-suites "$SUITE" --seeds $SEEDS \
        --task-sequence $SEQUENCE \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pool-size "$POOL_SIZE" \
        --skip-training \
        --scratch-seeds $SCRATCH_SEEDS \
        --save-root "$agents" --runs-root "$runs" --analysis-root "$analysis" \
        --scratch-save-root "$scratch" --plots-root "$plots" \
        "${extra[@]}" 2>&1 | tee "$OUT/logs/mw_${stage}_report.log"
    python3 estimate_timing.py --plots-root "$plots" || true
    echo "OK -> $plots"
}

case "${1:-}" in
    setup)     step_setup ;;
    sanity)    step_sanity ;;
    smoke)     step_smoke ;;
    pilot)     step_pilot ;;
    pretrain)  step_pretrain ;;
    baselines) step_baselines "${2:-s0}" ;;
    cond)      step_cond "${2:-s0}" "${3:-}" ;;
    report)    step_report "${2:-s0}" ;;
    *)
        cat <<'USAGE'
Usage: bash run_kaggle.sh <stage> [args]

  setup                     install deps (pinned Meta-World) + CUDA check
  sanity                    structural checks, no GPU time
  smoke                     ~10 min full-pipeline test at a tiny budget
  pilot                     ~1 h: do the real tasks learn in 150k steps?
  pretrain                  TD-JEPA shared encoder (needed only for s4)
  baselines [s0|s4]         from-scratch SAC per task (FT denominator)
  cond [s0|s4] <1..4>       ONE condition: 1 baseline 2 distil_only
                            3 weight_only 4 combined
  report [s0|s4]            metrics + plots over whatever is present

Env overrides: SEEDS, SCRATCH_SEEDS, TOTAL_TIMESTEPS, SUITE, SEQUENCE,
               POOL_SIZE, MW_COMMIT
USAGE
        exit 2
        ;;
esac
