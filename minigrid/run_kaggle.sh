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
#   cond N       varies    ONE of the four CKA conditions (idea: fusion/merge)
#   report       minutes   metrics + plots over whatever is present
#
# Conditions (--condition-index is 1-BASED; 0 would mean all four):
#   1 baseline     classic_cka + cosine + arithmetic merge
#   2 distil_only  classic_cka + symmetric-KL + policy distillation
#   3 weight_only  weight_delta + cosine + arithmetic merge
#   4 combined     weight_delta + symmetric-KL + policy distillation
#
# SUGGESTED SPLIT ACROSS TWO ACCOUNTS
#   Account A: smoke, pilot, baselines, cond 1, cond 2
#   Account B: baselines, cond 3, cond 4
#   (the whole plan is only a few GPU-hours, so one account is also fine)
#
# Everything is resumable: run_continual_benchmark.py and scratch_baselines.py
# check run manifests and skip completed work, so re-running a stage after a
# session drop continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
SEEDS="${SEEDS:-1 2 3}"
SCRATCH_SEEDS="${SCRATCH_SEEDS:-101 102 103}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
SUITE="${SUITE:-doorkey4}"
SEQUENCE="${SEQUENCE:-0 1 2 3 0 2 1 3}"
POOL_SIZE="${POOL_SIZE:-4}"
PRETRAIN_STEPS_PER_TASK="${PRETRAIN_STEPS_PER_TASK:-60000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"

# Pinned Meta-World commit. This is the revision that exposes
# ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, which the v2-era task ids in tasks.py
# rely on. Newer master has moved on and would silently change task semantics.
UNUSED_COMMIT="${UNUSED_COMMIT:-c822f28f582ba1ad49eb5dcf61016566f28003ba}"

mkdir -p "$OUT"/{pretrained_encoders,agents,plots,logs,runs,analysis,scratch_models,analysis_scratch}

cuda_check() {
python3 - <<'PYEOF'
import sys, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("NO GPU -- runs would fall back to CPU and be far too slow for this plan.")
    sys.exit(3)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print(f"gpu: {name} | capability: sm_{cap[0]}{cap[1]} | count: {torch.cuda.device_count()}")
print("wheel supports:", arches)

# torch.cuda.is_available() returns True on a P100 even when the installed
# wheel has no sm_60 kernels; the failure only surfaces much later as
# "no kernel image is available for execution on the device", deep inside
# training. Launch a real kernel here so it is caught during setup instead.
try:
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("real CUDA kernel test: OK", float(y.mean()))
except Exception as exc:
    print()
    print("=" * 72)
    print("CUDA KERNEL TEST FAILED -- do not start any training run")
    print("=" * 72)
    print(f"  {type(exc).__name__}: {exc}")
    print()
    print(f"  This GPU is sm_{cap[0]}{cap[1]}, but the installed PyTorch was built for:")
    print(f"    {arches}")
    print()
    print("  FIX: in the Kaggle right-hand panel set Accelerator = 'GPU T4 x2'")
    print("       (T4 is sm_75). P100 is sm_60 and this wheel has no kernels for it.")
    print("=" * 72)
    sys.exit(4)
PYEOF
}

step_setup() {
    echo ">>> setup"
    python3 -m pip install -q -r requirements.txt
    python3 -c "import minigrid, gymnasium; print('minigrid', minigrid.__version__, '| gymnasium', gymnasium.__version__)"
    cuda_check
    python3 tasks.py
    python3 -c "from tasks import get_task; e=get_task(0); o,_=e.reset(seed=0); print('obs', o.shape, '| actions', e.action_space.n); e.close()"
    echo "OK"
}

step_sanity() {
    echo ">>> sanity (no GPU time)"
    python3 sanity_check_pool.py
    python3 tasks.py --check
    echo "OK"
}

roots_for() {
    local stage="${1:-s0}"
    echo "$OUT/agents/mg_${stage}|$OUT/runs/mg_${stage}|$OUT/analysis/mg_${stage}|$OUT/scratch_models/mg_${stage}|$OUT/analysis_scratch/mg_${stage}|$OUT/plots/mg_${stage}"
}

encoder_flags() {
    # The shared encoder is trained on the root task and then frozen, exactly as
    # in the CKA-RL protocol. linear_out drops the encoder's trailing ReLU,
    # which costs nothing here (the heads apply their own) and keeps features
    # signed.
    local -n _out=$1
    _out=(--encoder-linear-out --no-train-shared --no-freeze-root-encoder)
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
    local extra; encoder_flags extra

    # --pool-size MUST match what the continual run below uses. The run manifest
    # in experiment_identity.py includes pool_size in the training signature, so
    # a baseline trained at the default 4 is rejected as a config mismatch and
    # the survey metrics (FT/BWT/A_N) are silently skipped -- with only a
    # warning buried in the log. Same rule for every real stage: baselines and
    # continual runs must agree on every training-signature key.
    python3 scratch_baselines.py \
        --task-suites mg_smoke2 --seeds 101 \
        --total-timesteps 4000 --eval-every 1000 --num-evals 3 \
        --learning-starts 400 --random-actions-end 800 \
        --pool-size 2 \
        --save-root "$scratch" --runs-root "$runs" --analysis-root "$scratch_analysis" \
        "${extra[@]}" 2>&1 | tee "$OUT/logs/mg_smoke_baselines.log"

    python3 run_continual_benchmark.py \
        --task-suites mg_smoke2 --seeds 1 \
        --task-sequence 0 1 0 \
        --total-timesteps 4000 --eval-every 1000 --num-evals 3 \
        --learning-starts 400 --random-actions-end 800 \
        --pool-size 2 --distill-extra-steps 1000 --test-adapt-steps 200 \
        --scratch-seeds 101 \
        --save-root "$agents" --runs-root "$runs" --analysis-root "$analysis" \
        --scratch-save-root "$scratch" --plots-root "$plots" \
        "${extra[@]}" 2>&1 | tee "$OUT/logs/mg_smoke_continual.log"
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
        --runs-root "$OUT/runs/mg_s0" \
        --save-dir "$OUT/scratch_models/mg_s0" \
        --analysis-root "$OUT/analysis_scratch/mg_s0" \
        2>&1 | tee "$OUT/logs/mg_pilot.log"
}

step_baselines() {
    local stage="s0" values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags extra
    echo ">>> baselines [$stage]: from-scratch SAC per task"
    python3 scratch_baselines.py \
        --task-suites "$SUITE" --seeds $SCRATCH_SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pool-size "$POOL_SIZE" \
        --save-root "$scratch" --runs-root "$runs" --analysis-root "$scratch_analysis" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/mg_${stage}_baselines.log"
}

# --------------------------------------------------------------------------
# cond: ONE condition at a time. With the measured throughput on a Kaggle T4
# (~250-800 steps/s: MiniGrid steps are cheap and the SAC update dominates), one condition at 150k x 8
# positions x 3 seeds is well under an hour, so a whole condition with all
# seeds fits comfortably in one session:
#
#   bash run_kaggle.sh cond 3
#
# Run the same command with SEEDS=2, then SEEDS=3, in later sessions. Results
# accumulate under the same roots and `report` averages whatever is present.
# --------------------------------------------------------------------------
step_cond() {
    local idx="${1:-}" stage="s0"
    [ -n "$idx" ] || { echo "usage: run_kaggle.sh cond <1..4>"; exit 2; }
    local values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags extra

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
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/mg_${stage}_cond${idx}.log"
}

# Report over whatever conditions are already present. Training is skipped, so
# this is safe to run after merging outputs from both accounts.
step_report() {
    local stage="s0" values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra; encoder_flags extra
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
        "${extra[@]}" 2>&1 | tee "$OUT/logs/mg_${stage}_report.log"
    python3 estimate_timing.py --plots-root "$plots" || true
    echo "OK -> $plots"
}

case "${1:-}" in
    setup)     step_setup ;;
    sanity)    step_sanity ;;
    smoke)     step_smoke ;;
    pilot)     step_pilot ;;
    baselines) step_baselines ;;
    cond)      step_cond "${2:-}" ;;
    report)    step_report ;;
    *)
        cat <<'USAGE'
Usage: bash run_kaggle.sh <stage> [args]

  setup                     install deps (pinned Meta-World) + CUDA check
  sanity                    structural checks, no GPU time
  smoke                     ~10 min full-pipeline test at a tiny budget
  pilot                     ~1 h: do the real tasks learn in 150k steps?
  baselines                 from-scratch SAC per task (FT denominator)
  cond <1..4>               ONE condition: 1 baseline  2 distil_only
                            3 weight_only  4 combined
  report                    metrics + plots over whatever is present

Env overrides: SEEDS, SCRATCH_SEEDS, TOTAL_TIMESTEPS, SUITE, SEQUENCE,
               POOL_SIZE, UNUSED_COMMIT
USAGE
        exit 2
        ;;
esac
