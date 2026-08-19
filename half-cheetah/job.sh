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
RUN_ROOT="artifacts/cka_halfcheetah_50k"

mkdir -p \
    "$RUN_ROOT/agents" \
    "$RUN_ROOT/runs" \
    "$RUN_ROOT/plots" \
    "$RUN_ROOT/analysis"

echo "============================================================"
echo "[job] CKA-RL HalfCheetah benchmark"
echo "[job] ID: ${SLURM_JOB_ID:-unknown}"
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
echo "[train] suites:"
echo "        halfcheetah_vel"
echo "        halfcheetah_wind_vel"
echo "[train]"
echo "[train] variants:"
echo "        baseline"
echo "        distil_only"
echo "        weight_only"
echo "        combined"
echo "[train]"
echo "[train] seeds: 1 2 3"
echo "[train] timesteps/task: 50000"
echo "[train] pool-size: 5"
echo "============================================================"


srun python -u run_continual_benchmark.py \
    --task-suites halfcheetah_wind_vel \
    --seeds 1 2 \
    --total-timesteps 50000 \
    --pool-size 5 \
    --batch-size 256 \
    --policy-lr 3e-4 \
    --q-lr 3e-4 \
    --learning-starts 3000 \
    --random-actions-end 5000 \
    --eval-every 5000 \
    --num-evals 5 \
    --distill-extra-steps 5000 \
    --similarity-samples 2048 \
    --max-distill-buffer 50000 \
    --distill-max-samples 20000 \
    --distill-epochs 8 \
    --distill-lr 3e-4 \
    --distill-batch-size 256 \
    --distill-test-frac 0.2 \
    --save-root "$RUN_ROOT/agents" \
    --runs-root runs \
    --plots-root "$RUN_ROOT/plots" \
    --skip-training \
    --analysis-root "$RUN_ROOT/analysis"

echo
echo "============================================================"
echo "[done] benchmark completed"
echo "[done] checkpoints: $RUN_ROOT/agents"
echo "[done] TensorBoard: $RUN_ROOT/runs"
echo "[done] plots:      $RUN_ROOT/plots"
echo "[done] analysis:   $RUN_ROOT/analysis"
echo "============================================================"