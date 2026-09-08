#!/usr/bin/env bash
# Kaggle entrypoint for the CURRENT HalfCheetah workflow.
#
# The default experiment follows the task-blind comparison setup: no encoder pretraining,
# shared encoder frozen after the root task, and Baseline + Combined selected together. Optional
# TD-JEPA pretraining remains available through tdjepa_pretrain.py as an explicit
# ablation, but is not part of this default pipeline.
set -euo pipefail
COMPOSITION_SPACES="${COMPOSITION_SPACES:-parameter policy}"
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
SEEDS="${SEEDS:-101 102}"
SCRATCH_SEEDS="${SCRATCH_SEEDS:-201}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000}"
SUITES="${SUITES:-halfcheetah_vel halfcheetah_wind_vel}"
mkdir -p "$OUT"/{agents,plots,logs,runs,analysis,scratch_models,analysis_scratch}

cuda_check() {
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("compiled architectures:", torch.cuda.get_arch_list())
    # cuda.is_available() can be true even if the wheel has no kernel for the GPU.
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
        2>&1 | tee -a "$OUT/logs/hc_baselines.log"
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
        2>&1 | tee -a "$OUT/logs/hc_continual.log"
}

step_all() {
    step_sanity
    step_baselines
    step_continual
}

case "${1:-}" in
    setup) step_setup ;;
    sanity) step_sanity ;;
    baselines) step_baselines ;;
    continual) step_continual ;;
    all) step_all ;;
    *) echo "Usage: bash run_kaggle.sh {setup|sanity|baselines|continual|all}"; exit 2 ;;
esac
