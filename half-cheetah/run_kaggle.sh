#!/usr/bin/env bash
# Kaggle entrypoint for the FINAL HalfCheetah pipeline.
#
# S0 and S4 are encoder ablations, not the four CKA fusion conditions:
#   S0 = no TD-JEPA pretraining, root encoder learned on task 0 then frozen
#   S4 = TD-JEPA pretrained encoder, frozen from task 0
# Both use the SAME linear-output encoder/critic architecture so pretraining is
# the only S0-vs-S4 encoder treatment. The 4 CKA conditions are run inside each.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
SEEDS="${SEEDS:-1 2 3}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
PRETRAIN_STEPS_PER_TASK="${PRETRAIN_STEPS_PER_TASK:-100000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
SUITE="halfcheetah_vel"
ENC_DIR="$OUT/pretrained_encoders/halfcheetah_tdjepa"
mkdir -p "$OUT"/{pretrained_encoders,agents,plots,logs,runs,analysis,scratch_models}

cuda_check() {
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("compiled architectures:", torch.cuda.get_arch_list())
    # cuda.is_available() alone can be True on a P100 even when the wheel has
    # no sm_60 kernels. This actual kernel launch catches that configuration.
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

step_pretrain() {
    if [ -f "$ENC_DIR/fc.pt" ] && [ -f "$ENC_DIR/report.json" ] && [ "${FORCE_PRETRAIN:-0}" != "1" ]; then
        echo ">>> TD-JEPA encoder already exists: $ENC_DIR/fc.pt"
        return
    fi
    rm -rf "$ENC_DIR"
    python3 tdjepa_pretrain.py \
        --task-suite "$SUITE" \
        --pretrain-velocities 0.75 1.75 2.75 \
        --heldout-velocities 0.5 2.0 3.0 \
        --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
        --out "$ENC_DIR" 2>&1 | tee "$OUT/logs/hc_pretrain.log"
}

roots_for() {
    local stage="$1"
    echo "$OUT/agents/hc_${stage}|$OUT/runs/hc_${stage}|$OUT/analysis/hc_${stage}|$OUT/scratch_models/hc_${stage}|$OUT/analysis_scratch/hc_${stage}|$OUT/plots/hc_${stage}"
}

step_baselines() {
    local stage="${1:-s0}" values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra=(--encoder-linear-out --no-train-shared --no-freeze-root-encoder)
    if [ "$stage" = "s4" ]; then
        [ -f "$ENC_DIR/fc.pt" ] || { echo "Missing $ENC_DIR/fc.pt; run pretrain first"; exit 1; }
        extra+=(--pretrained-encoder "$ENC_DIR/fc.pt")
    elif [ "$stage" != "s0" ]; then
        echo "stage must be s0 or s4"; exit 2
    fi
    python3 scratch_baselines.py \
        --task-suites "$SUITE" --seeds 101 102 103 \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$scratch" --runs-root "$runs" --analysis-root "$scratch_analysis" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/hc_${stage}_baselines.log"
}

step_continual() {
    local stage="${1:-s0}" values agents runs analysis scratch scratch_analysis plots
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra=(--encoder-linear-out --no-train-shared --no-freeze-root-encoder)
    if [ "$stage" = "s4" ]; then
        [ -f "$ENC_DIR/fc.pt" ] || { echo "Missing $ENC_DIR/fc.pt; run pretrain first"; exit 1; }
        extra+=(--pretrained-encoder "$ENC_DIR/fc.pt")
    elif [ "$stage" != "s0" ]; then
        echo "stage must be s0 or s4"; exit 2
    fi
    python3 run_continual_benchmark.py \
        --task-suites "$SUITE" --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$agents" --runs-root "$runs" --analysis-root "$analysis" \
        --scratch-save-root "$scratch" --plots-root "$plots" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/hc_${stage}_continual.log"
}

step_all() {
    step_sanity
    step_pretrain
    step_baselines s0
    step_baselines s4
    step_continual s0
    step_continual s4
}

case "${1:-}" in
    setup) step_setup ;;
    sanity) step_sanity ;;
    pretrain) step_pretrain ;;
    baselines) step_baselines "${2:-s0}" ;;
    continual) step_continual "${2:-s0}" ;;
    all) step_all ;;
    *) echo "Usage: bash run_kaggle.sh {setup|sanity|pretrain|baselines [s0|s4]|continual [s0|s4]|all}"; exit 2 ;;
esac
