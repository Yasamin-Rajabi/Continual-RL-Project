#!/usr/bin/env bash
# Kaggle entrypoint for the FINAL Ant pipeline.
# Calibration is automatic and machine-readable; no manual source edit is used.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
SEEDS="${SEEDS:-1 2 3}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
PRETRAIN_STEPS_PER_TASK="${PRETRAIN_STEPS_PER_TASK:-100000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
CALIBRATION_STEPS="${CALIBRATION_STEPS:-150000}"
SUITE="ant_vel"
CALIBRATION_FILE="$PWD/ant_calibration.json"
ENC_DIR="$OUT/pretrained_encoders/ant_tdjepa"
mkdir -p "$OUT"/{pretrained_encoders,agents,plots,logs,runs,analysis,scratch_models}

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

step_calibrate() {
    if [ -f "$CALIBRATION_FILE" ] && [ "${FORCE_CALIBRATION:-0}" != "1" ]; then
        echo ">>> Ant calibration already exists: $CALIBRATION_FILE"
        python3 - <<'PY'
import json
print(json.load(open("ant_calibration.json")))
PY
        return
    fi
    python3 calibrate_ant.py \
        --total-timesteps "$CALIBRATION_STEPS" --seeds 1 2 3 \
        --save-root "$OUT/agents/ant_calibration" \
        --runs-root "$OUT/runs/ant_calibration" \
        --analysis-root "$OUT/analysis/ant_calibration" \
        --output "$CALIBRATION_FILE" 2>&1 | tee "$OUT/logs/ant_calibration.log"
}

step_sanity() {
    python3 sanity_check_pool.py
    python3 tasks.py --check
}

pretrain_matches_calibration() {
    [ -f "$ENC_DIR/fc.pt" ] && [ -f "$ENC_DIR/report.json" ] || return 1
    python3 - "$ENC_DIR/report.json" "$CALIBRATION_FILE" <<'PY'
import json, math, sys
report=json.load(open(sys.argv[1])); cal=json.load(open(sys.argv[2]))
saved=(report.get("ant_calibration") or {}).get("v_max")
raise SystemExit(0 if saved is not None and math.isclose(float(saved), float(cal["v_max"]), rel_tol=0, abs_tol=1e-12) else 1)
PY
}

step_pretrain() {
    [ -f "$CALIBRATION_FILE" ] || { echo "Missing calibration; run calibrate first"; exit 1; }
    if pretrain_matches_calibration && [ "${FORCE_PRETRAIN:-0}" != "1" ]; then
        echo ">>> TD-JEPA encoder matches current calibration: $ENC_DIR/fc.pt"
        return
    fi
    rm -rf "$ENC_DIR"
    # Velocities are derived automatically from ant_calibration.json by tasks.py.
    python3 tdjepa_pretrain.py \
        --task-suite "$SUITE" \
        --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
        --out "$ENC_DIR" 2>&1 | tee "$OUT/logs/ant_pretrain.log"
}

roots_for() {
    local stage="$1"
    echo "$OUT/agents/ant_${stage}|$OUT/runs/ant_${stage}|$OUT/analysis/ant_${stage}|$OUT/scratch_models/ant_${stage}|$OUT/analysis_scratch/ant_${stage}|$OUT/plots/ant_${stage}"
}

step_baselines() {
    local stage="${1:-s0}" values agents runs analysis scratch scratch_analysis plots
    [ -f "$CALIBRATION_FILE" ] || { echo "Missing calibration; run calibrate first"; exit 1; }
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    # Match the TD-JEPA linear-output architecture in BOTH S0 and S4. S0 thus
    # isolates pretraining rather than confounding it with a final-ReLU change.
    local extra=(--encoder-linear-out --no-train-shared --no-freeze-root-encoder)
    if [ "$stage" = "s4" ]; then
        pretrain_matches_calibration || { echo "Pretrained encoder is missing/stale; run pretrain"; exit 1; }
        extra+=(--pretrained-encoder "$ENC_DIR/fc.pt")
    elif [ "$stage" != "s0" ]; then
        echo "stage must be s0 or s4"; exit 2
    fi
    python3 scratch_baselines.py \
        --task-suites "$SUITE" --seeds 101 102 103 \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$scratch" --runs-root "$runs" --analysis-root "$scratch_analysis" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/ant_${stage}_baselines.log"
}

step_continual() {
    local stage="${1:-s0}" values agents runs analysis scratch scratch_analysis plots
    [ -f "$CALIBRATION_FILE" ] || { echo "Missing calibration; run calibrate first"; exit 1; }
    values="$(roots_for "$stage")"; IFS='|' read -r agents runs analysis scratch scratch_analysis plots <<< "$values"
    local extra=(--encoder-linear-out --no-train-shared --no-freeze-root-encoder)
    if [ "$stage" = "s4" ]; then
        pretrain_matches_calibration || { echo "Pretrained encoder is missing/stale; run pretrain"; exit 1; }
        extra+=(--pretrained-encoder "$ENC_DIR/fc.pt")
    elif [ "$stage" != "s0" ]; then
        echo "stage must be s0 or s4"; exit 2
    fi
    python3 run_continual_benchmark.py \
        --task-suites "$SUITE" --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$agents" --runs-root "$runs" --analysis-root "$analysis" \
        --scratch-save-root "$scratch" --plots-root "$plots" \
        "${extra[@]}" 2>&1 | tee -a "$OUT/logs/ant_${stage}_continual.log"
}

step_all() {
    step_calibrate
    step_sanity
    step_pretrain
    step_baselines s0
    step_baselines s4
    step_continual s0
    step_continual s4
}

case "${1:-}" in
    setup) step_setup ;;
    calibrate) step_calibrate ;;
    sanity) step_sanity ;;
    pretrain) step_pretrain ;;
    baselines) step_baselines "${2:-s0}" ;;
    continual) step_continual "${2:-s0}" ;;
    all) step_all ;;
    *) echo "Usage: bash run_kaggle.sh {setup|calibrate|sanity|pretrain|baselines [s0|s4]|continual [s0|s4]|all}"; exit 2 ;;
esac
