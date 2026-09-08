#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --output=logs/cka_hc_50k_%j.out
#SBATCH --error=logs/cka_hc_50k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --qos=normal

set -euo pipefail

# ============================================================
# EDIT YOUR EXPERIMENTS HERE
# ============================================================

# Each entry is submitted as its own SLURM job / H100.
# Default requested comparison:
VARIANTS=(
    baseline
    combined
    combined_policy
)

# All variant outputs live below this root in separate directories.
RUN_ROOT_BASE="artifacts/cka_halfcheetah_50k"

# Hyperparameters shared by every job.
# Add/remove any run_continual_benchmark.py option here.
COMMON_ARGS=(
    --task-suites halfcheetah_wind_vel
    --seeds 1 2 3
    --total-timesteps 80000
    --pool-size 5
    --batch-size 256
    --policy-lr 3e-4
    --q-lr 3e-4
    --learning-starts 5000
    --random-actions-end 5000
    --alpha-warmup-steps 5000
    --eval-every 5000
    --num-evals 5
    --distill-buffer-steps 5000
    --similarity-samples 2048
    --max-distill-buffer 50000
    --distill-max-samples 20000
    --distill-epochs 16
    --distill-lr 5e-4
    --distill-batch-size 256
    --distill-test-frac 0.2
)

# Optional per-variant additions or overrides.
# If a flag also appears in COMMON_ARGS, argparse uses the later value below.
BASELINE_ARGS=(
)

COMBINED_ARGS=(
)

COMBINED_POLICY_ARGS=(
    # Examples:
    --projection-epochs 32
    --projection-max-samples 50000
    --eval-action-mode stochastic
)

# Kept available for later ablations. They do not run unless added to VARIANTS.
DISTIL_ONLY_ARGS=(
)
WEIGHT_ONLY_ARGS=(
)
BASELINE_POLICY_ARGS=(
)
DISTIL_ONLY_POLICY_ARGS=(
)
WEIGHT_ONLY_POLICY_ARGS=(
)

# ============================================================
# END EXPERIMENT CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Variant -> benchmark mapping
# ------------------------------------------------------------
variant_mapping() {
    local variant="$1"
    case "$variant" in
        baseline)             echo "1 parameter" ;;
        distil_only)          echo "2 parameter" ;;
        weight_only)          echo "3 parameter" ;;
        combined)             echo "4 parameter" ;;
        baseline_policy)      echo "1 policy" ;;
        distil_only_policy)   echo "2 policy" ;;
        weight_only_policy)   echo "3 policy" ;;
        combined_policy)      echo "4 policy" ;;
        *) return 1 ;;
    esac
}

variant_extra_args() {
    local variant="$1"
    case "$variant" in
        baseline)             printf '%s\0' "${BASELINE_ARGS[@]}" ;;
        distil_only)          printf '%s\0' "${DISTIL_ONLY_ARGS[@]}" ;;
        weight_only)          printf '%s\0' "${WEIGHT_ONLY_ARGS[@]}" ;;
        combined)             printf '%s\0' "${COMBINED_ARGS[@]}" ;;
        baseline_policy)      printf '%s\0' "${BASELINE_POLICY_ARGS[@]}" ;;
        distil_only_policy)   printf '%s\0' "${DISTIL_ONLY_POLICY_ARGS[@]}" ;;
        weight_only_policy)   printf '%s\0' "${WEIGHT_ONLY_POLICY_ARGS[@]}" ;;
        combined_policy)      printf '%s\0' "${COMBINED_POLICY_ARGS[@]}" ;;
        *) return 1 ;;
    esac
}

# ------------------------------------------------------------
# Submission mode
# Run locally as:
#     bash submit_slurm_variants.sh
# It submits this same file once per VARIANT, then exits.
# ------------------------------------------------------------
if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p logs "$RUN_ROOT_BASE"

    SCRIPT_PATH="$(realpath "$0")"

    echo "============================================================"
    echo "Submitting variants: ${VARIANTS[*]}"
    echo "Worker script: $SCRIPT_PATH"
    echo "Run root: $RUN_ROOT_BASE"
    echo "============================================================"

    for variant in "${VARIANTS[@]}"; do
        if ! variant_mapping "$variant" >/dev/null; then
            echo "ERROR: unsupported variant: $variant" >&2
            exit 2
        fi

        safe_variant="${variant//_/-}"
        job_name="cka-${safe_variant}"
        out_file="logs/${variant}_%j.out"
        err_file="logs/${variant}_%j.err"

        job_id="$(
            sbatch --parsable \
                --job-name="$job_name" \
                --output="$out_file" \
                --error="$err_file" \
                "$SCRIPT_PATH" --worker "$variant"
        )"
        job_id="${job_id%%;*}"
        echo "[submitted] $variant -> job $job_id"
    done

    echo "============================================================"
    echo "All jobs submitted."
    echo "Queue: squeue -u \"$USER\""
    echo "============================================================"
    exit 0
fi

# ------------------------------------------------------------
# Worker mode: entered by SLURM after submission above.
# ------------------------------------------------------------
VARIANT="${2:?Missing worker variant}"
read -r CONDITION_INDEX COMPOSITION_SPACE <<<"$(variant_mapping "$VARIANT")" || {
    echo "ERROR: unsupported worker variant: $VARIANT" >&2
    exit 2
}

# Load optional variant-specific CLI flags into an array without eval.
VARIANT_ARGS=()
while IFS= read -r -d '' arg; do
    VARIANT_ARGS+=("$arg")
done < <(variant_extra_args "$VARIANT")

RUN_ROOT="$RUN_ROOT_BASE/$VARIANT"

# ------------------------------------------------------------
# Cluster setup
# ------------------------------------------------------------
module purge
module load gcc/13.2.0 python/3.9.18 py-virtualenv/20.24.5

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ------------------------------------------------------------
# Existing virtual environment
# ------------------------------------------------------------
VENV="${CKA_VENV:-$HOME/.venvs/cka_halfcheetah}"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: virtual environment does not exist:"
    echo "  $VENV"
    exit 1
fi

source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Headless MuJoCo
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0
export MPLBACKEND=Agg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------
mkdir -p \
    "$RUN_ROOT/agents" \
    "$RUN_ROOT/runs" \
    "$RUN_ROOT/plots" \
    "$RUN_ROOT/analysis"

echo "============================================================"
echo "[job] CKA-RL HalfCheetah benchmark"
echo "[job] ID: ${SLURM_JOB_ID:-unknown}"
echo "[job] variant: $VARIANT"
echo "[job] condition-index: $CONDITION_INDEX"
echo "[job] composition-space: $COMPOSITION_SPACE"
echo "[job] directory: $PWD"
echo "[job] venv: $VENV"
echo "[job] output: $RUN_ROOT"
echo "============================================================"

python --version
nvidia-smi

# ------------------------------------------------------------
# Verify H100
# ------------------------------------------------------------
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

if [[ "$GPU_NAME" != *H100* ]]; then
    echo "ERROR: expected H100, got: $GPU_NAME" >&2
    exit 1
fi

echo "[gpu] $GPU_NAME"

# ------------------------------------------------------------
# Ensure compatible PyTorch
#
# SB3 2.9.0 requires torch >= 2.8.
# Use official PyTorch 2.8 CUDA 12.6 wheel.
#
# We are NOT installing a system CUDA toolkit.
# The cluster provides the NVIDIA driver.
# PyTorch brings its CUDA runtime libraries.
# ------------------------------------------------------------
TORCH_VERSION="2.8.0"
TORCH_CUDA="12.6"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"

NEED_TORCH=1

if python - <<PY >/dev/null 2>&1
import torch

assert torch.__version__.split("+")[0] == "$TORCH_VERSION"
assert torch.version.cuda == "$TORCH_CUDA"
PY
then
    NEED_TORCH=0
fi

if (( NEED_TORCH )); then
    echo "[setup] installing torch $TORCH_VERSION + CUDA $TORCH_CUDA"

    python -m pip uninstall -y \
        torch torchvision torchaudio triton \
        >/dev/null 2>&1 || true

    python -m pip install \
        "torch==$TORCH_VERSION" \
        --index-url "$TORCH_INDEX"
else
    echo "[setup] correct PyTorch build already installed"
fi

# ------------------------------------------------------------
# Verify versions
# ------------------------------------------------------------
python - <<'PY'
import torch
import stable_baselines3
import gymnasium
import mujoco

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("Gymnasium:", gymnasium.__version__)
print("MuJoCo:", mujoco.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside H100 allocation")

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

# Actual CUDA computation
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
c = a @ b
torch.cuda.synchronize()

print("CUDA matrix multiplication: OK")
print("Result device:", c.device)
PY

# Check that pip sees no dependency conflicts
python -m pip check

# ------------------------------------------------------------
# CKA structural tests
# ------------------------------------------------------------
echo
echo "============================================================"
echo "[test] CKA pool sanity checks"
echo "============================================================"

python -u sanity_check_pool.py

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
echo
echo "============================================================"
echo "[train] variant:          $VARIANT"
echo "[train] condition-index:  $CONDITION_INDEX"
echo "[train] composition:      $COMPOSITION_SPACE"
printf '[train] command:           run_continual_benchmark.py'
printf ' %q' "${COMMON_ARGS[@]}" "${VARIANT_ARGS[@]}"
printf ' --condition-index %q --composition-spaces %q\n' "$CONDITION_INDEX" "$COMPOSITION_SPACE"
echo "============================================================"

srun python -u run_continual_benchmark.py \
    "${COMMON_ARGS[@]}" \
    "${VARIANT_ARGS[@]}" \
    --condition-index "$CONDITION_INDEX" \
    --composition-spaces "$COMPOSITION_SPACE" \
    --save-root "$RUN_ROOT/agents" \
    --runs-root "$RUN_ROOT/runs" \
    --plots-root "$RUN_ROOT/plots" \
    --analysis-root "$RUN_ROOT/analysis"

echo
echo "============================================================"
echo "[done] benchmark completed"
echo "[done] variant:     $VARIANT"
echo "[done] checkpoints: $RUN_ROOT/agents"
echo "[done] TensorBoard: $RUN_ROOT/runs"
echo "[done] plots:       $RUN_ROOT/plots"
echo "[done] analysis:    $RUN_ROOT/analysis"
echo "============================================================"
