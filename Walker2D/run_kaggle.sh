#!/usr/bin/env bash
# Kaggle entrypoint for the Walker2D continual-dynamics benchmark.
#
# Default workflow mirrors the revised HalfCheetah directory: no encoder
# pretraining, shared encoder frozen after the root task, and Baseline + Combined run together.
set -euo pipefail
COMPOSITION_SPACES="${COMPOSITION_SPACES:-parameter policy}"
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
SEEDS="${SEEDS:-101 102}"
SCRATCH_SEEDS="${SCRATCH_SEEDS:-201}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-150000}"
SUITES="${SUITES:-walker2d_dynamics}"
mkdir -p "$OUT"/{agents,plots,logs,runs,analysis,scratch_models,analysis_scratch}

cuda_check() {
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("compiled architectures:", torch.cuda.get_arch_list())
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("real CUDA kernel test: OK", float(y.mean()))
PY
}

step_setup() {
    python3 -m pip install -q -r requirements.txt --break-system-packages 2>/dev/null || \
    python3 -m pip install -q -r requirements.txt
    python3 -m pip check
    cuda_check
}

step_sanity() {
    python3 sanity_check_pool.py
    python3 tasks.py --check
    python3 walker2d_smoke.py
}

step_baselines() {
    python3 scratch_baselines.py \
        --task-suites $SUITES \
        --seeds $SCRATCH_SEEDS \
        --variants plain \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$OUT/scratch_models" \
        --runs-root "$OUT/runs" \
        --analysis-root "$OUT/analysis_scratch" \
        --no-train-shared \
        2>&1 | tee -a "$OUT/logs/walker2d_baselines.log"
}

step_continual() {
    python3 run_continual_benchmark.py \
        --composition-spaces $COMPOSITION_SPACES \
        --task-suites $SUITES \
        --seeds $SEEDS \
        --scratch-seeds $SCRATCH_SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --condition-index 1 4 \
        --no-train-shared \
        --save-root "$OUT/agents" \
        --runs-root "$OUT/runs" \
        --analysis-root "$OUT/analysis" \
        --scratch-save-root "$OUT/scratch_models" \
        --plots-root "$OUT/plots" \
        2>&1 | tee -a "$OUT/logs/walker2d_continual.log"
}

step_pilot() {
    # Short end-to-end consolidation run: four tasks, two conditions, and an
    # actual pool merge. This validates training/merging without committing to
    # the full 12-task paper sequence or scratch-baseline suite.
    python3 run_continual_benchmark.py \
        --composition-spaces $COMPOSITION_SPACES \
        --task-suites walker2d_dynamics \
        --seeds 101 \
        --task-sequence 0 1 2 3 \
        --total-timesteps 20000 \
        --learning-starts 1000 \
        --random-actions-end 2000 \
        --pool-size 2 \
        --eval-every 5000 \
        --num-evals 1 \
        --retention-eval-episodes 1 \
        --test-adapt-steps 0 \
        --distill-extra-steps 1000 \
        --max-distill-buffer 4000 \
        --similarity-samples 256 \
        --distill-max-samples 1000 \
        --distill-epochs 2 \
        --analysis-log-every 2000 \
        --condition-index 1 4 \
        --no-train-shared \
        --skip-retention \
        --skip-survey-metrics \
        --save-root "$OUT/pilot_agents" \
        --runs-root "$OUT/pilot_runs" \
        --analysis-root "$OUT/pilot_analysis" \
        --plots-root "$OUT/pilot_plots" \
        2>&1 | tee -a "$OUT/logs/walker2d_pilot.log"
}

step_all() {
    step_sanity
    step_baselines
    step_continual
}

case "${1:-}" in
    setup) step_setup ;;
    sanity) step_sanity ;;
    pilot) step_pilot ;;
    baselines) step_baselines ;;
    continual) step_continual ;;
    all) step_all ;;
    *) echo "Usage: bash run_kaggle.sh {setup|sanity|pilot|baselines|continual|all}"; exit 2 ;;
esac
