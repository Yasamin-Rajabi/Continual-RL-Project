#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --output=logs/cka_mw_%j.out
#SBATCH --error=logs/cka_mw_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --qos=normal

set -euo pipefail

# Always run relative to the canonical metaworld/ directory containing this file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# EDIT YOUR EXPERIMENTS HERE
# ============================================================

VARIANTS=(
    baseline
    combined
    combined_policy
)

# Canonical suite from the fixed code:
#   mw_paper10 = the 10 CKA-RL Appendix C.1 tasks, repeated twice.
TASK_SUITE="${TASK_SUITE:-mw_paper10}"

# 150k is a practical first run.
# For the CKA-RL paper's 1M steps PER TASK:
#   TOTAL_TIMESTEPS=1000000 bash job_metaworld.sh
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"

# CKA-RL reports K_max=8 for Meta-World.
POOL_SIZE="${POOL_SIZE:-8}"

# Our preferred pool-search experiment:
# learning begins at 5k and the historical mixture controls collection from 5k.
# For the CKA-RL paper's original 10k random-action exploration:
#   RANDOM_ACTIONS_END=10000 bash job_metaworld.sh
RANDOM_ACTIONS_END="${RANDOM_ACTIONS_END:-5000}"

DISTILL_BUFFER_STEPS="${DISTILL_BUFFER_STEPS:-10000}"

RUN_ROOT_BASE="${RUN_ROOT_BASE:-artifacts/cka_metaworld_paper10_1000k}"

# Pinned Meta-World commit used by the fixed project.
--skip-forward-transfer="${MW_COMMIT:-c822f28f582ba1ad49eb5dcf61016566f28003ba}"

COMMON_ARGS=(
    --task-suites "$TASK_SUITE"
    --seeds 1 2 3
    --total-timesteps "$TOTAL_TIMESTEPS"

    --pool-size "$POOL_SIZE"

    # Meta-World / CKA-RL SAC settings.
    --batch-size 128
    --policy-lr 1e-3
    --q-lr 1e-3
    --gamma 0.99
    --tau 0.005
    --alpha 0.2
    --autotune

    # Pool-search / exploration schedule.
    --learning-starts 5000
    --random-actions-end "$RANDOM_ACTIONS_END"
    --alpha-warmup-steps 5000

    # Evaluation.
    --eval-every 10000
    --num-evals 5
    --retention-eval-episodes 5
    --test-adapt-steps 0
    --frozen-eval-policy pool
    --eval-action-mode deterministic

    # Keep every method on the same actor architecture.
    --no-distill-observation-skip

    # Frozen final B interactions are INSIDE total_timesteps.
    --distill-buffer-steps "$DISTILL_BUFFER_STEPS"

    # Merge / distillation diagnostics.
    --similarity-samples 2048
    --max-distill-buffer 50000
    --distill-max-samples 20000
    --distill-epochs 16
    --distill-lr 5e-4
    --distill-batch-size 256
    --distill-test-frac 0.2

    # Do A_N / FG / BWT now without requiring scratch FT baselines.
    # Remove this only after matched Meta-World scratch baselines are cached.
    # --skip-forward-transfer
)

BASELINE_ARGS=(
)

COMBINED_ARGS=(
)

COMBINED_POLICY_ARGS=(
    --projection-epochs 32
    --projection-max-samples 50000
)

# Kept available for later ablations.
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

# ------------------------------------------------------------
# Submission mode
# ------------------------------------------------------------
if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p logs "$RUN_ROOT_BASE"

    SCRIPT_PATH="$(realpath "$0")"

    echo "============================================================"
    echo "Submitting Meta-World variants: ${VARIANTS[*]}"
    echo "Folder: $SCRIPT_DIR"
    echo "Suite: $TASK_SUITE"
    echo "Steps/task: $TOTAL_TIMESTEPS"
    echo "Pool size: $POOL_SIZE"
    echo "Random actions end: $RANDOM_ACTIONS_END"
    echo "Run root: $RUN_ROOT_BASE"
    echo "============================================================"

    for variant in "${VARIANTS[@]}"; do
        if ! variant_mapping "$variant" >/dev/null; then
            echo "ERROR: unsupported variant: $variant" >&2
            exit 2
        fi

        safe_variant="${variant//_/-}"
        job_name="mw-${safe_variant}"
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
# Worker mode
# ------------------------------------------------------------
VARIANT="${2:?Missing worker variant}"
read -r CONDITION_INDEX COMPOSITION_SPACE <<<"$(variant_mapping "$VARIANT")" || {
    echo "ERROR: unsupported worker variant: $VARIANT" >&2
    exit 2
}

VARIANT_ARGS=()
case "$VARIANT" in
    baseline)
        VARIANT_ARGS=("${BASELINE_ARGS[@]}")
        ;;
    distil_only)
        VARIANT_ARGS=("${DISTIL_ONLY_ARGS[@]}")
        ;;
    weight_only)
        VARIANT_ARGS=("${WEIGHT_ONLY_ARGS[@]}")
        ;;
    combined)
        VARIANT_ARGS=("${COMBINED_ARGS[@]}")
        ;;
    baseline_policy)
        VARIANT_ARGS=("${BASELINE_POLICY_ARGS[@]}")
        ;;
    distil_only_policy)
        VARIANT_ARGS=("${DISTIL_ONLY_POLICY_ARGS[@]}")
        ;;
    weight_only_policy)
        VARIANT_ARGS=("${WEIGHT_ONLY_POLICY_ARGS[@]}")
        ;;
    combined_policy)
        VARIANT_ARGS=("${COMBINED_POLICY_ARGS[@]}")
        ;;
    *)
        echo "ERROR: unsupported worker variant: $VARIANT" >&2
        exit 2
        ;;
esac

RUN_ROOT="$RUN_ROOT_BASE/$VARIANT"

# ------------------------------------------------------------
# Cluster setup
# ------------------------------------------------------------
module purge
module load gcc/13.2.0 python/3.9.18 py-virtualenv/20.24.5

# Separate Meta-World venv by default.
VENV="${CKA_VENV:-$HOME/.venvs/cka_metaworld}"
VENV_LOCK="${VENV}.setup.lock"

mkdir -p "$(dirname "$VENV")"

# Multiple variants may start simultaneously. Serialize shared-venv setup.
exec 9>"$VENV_LOCK"
flock 9

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[setup] creating virtual environment: $VENV"
    python -m venv "$VENV"
fi

source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install -q --upgrade pip

# ------------------------------------------------------------
# PyTorch CUDA build
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
    python -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true
    python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi

# ------------------------------------------------------------
# Meta-World + remaining Python dependencies
# ------------------------------------------------------------
python -m pip install -q "mujoco>=3.0,<4"

MW_MARKER="$VENV/.metaworld_commit"
CURRENT_MW_COMMIT=""
if [[ -f "$MW_MARKER" ]]; then
    CURRENT_MW_COMMIT="$(cat "$MW_MARKER")"
fi

if [[ "$CURRENT_MW_COMMIT" != "$MW_COMMIT" ]] || ! python -c "import metaworld" >/dev/null 2>&1; then
    echo "[setup] installing pinned Meta-World commit: $MW_COMMIT"
    python -m pip uninstall -y metaworld >/dev/null 2>&1 || true
    python -m pip install --no-deps \
        "git+https://github.com/Farama-Foundation/Metaworld.git@$MW_COMMIT"
    printf '%s\n' "$MW_COMMIT" > "$MW_MARKER"
fi

python -m pip install -q -r requirements.txt

# Shared environment is ready; allow other jobs through.
flock -u 9

# ------------------------------------------------------------
# Headless MuJoCo
# ------------------------------------------------------------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0
export MPLBACKEND=Agg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p \
    "$RUN_ROOT/agents" \
    "$RUN_ROOT/runs" \
    "$RUN_ROOT/plots" \
    "$RUN_ROOT/analysis"

echo "============================================================"
echo "[job] CKA-RL / Ethos Meta-World benchmark"
echo "[job] ID: ${SLURM_JOB_ID:-unknown}"
echo "[job] variant: $VARIANT"
echo "[job] suite: $TASK_SUITE"
echo "[job] condition-index: $CONDITION_INDEX"
echo "[job] composition-space: $COMPOSITION_SPACE"
echo "[job] steps/task: $TOTAL_TIMESTEPS"
echo "[job] pool size: $POOL_SIZE"
echo "[job] directory: $PWD"
echo "[job] venv: $VENV"
echo "[job] output: $RUN_ROOT"
echo "============================================================"

python --version
nvidia-smi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$GPU_NAME" != *H100* ]]; then
    echo "ERROR: expected H100, got: $GPU_NAME" >&2
    exit 1
fi
echo "[gpu] $GPU_NAME"

# ------------------------------------------------------------
# Verify versions and CUDA
# ------------------------------------------------------------
python - <<'PY'
import torch
import stable_baselines3
import gymnasium
import mujoco
import metaworld

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("Gymnasium:", gymnasium.__version__)
print("MuJoCo:", mujoco.__version__)
print("Meta-World import:", metaworld.__file__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside H100 allocation")

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
c = a @ b
torch.cuda.synchronize()
print("CUDA matrix multiplication: OK", c.device)
PY

python -m pip check

# ------------------------------------------------------------
# Verify the actual mw_paper10 environments
# ------------------------------------------------------------
echo
echo "============================================================"
echo "[test] Meta-World task/API check"
echo "============================================================"

python - <<'PY'
import numpy as np
from tasks import TASK_SUITES, default_sequence, get_task

suite = "mw_paper10"
print("suite:", suite)
print("tasks:", len(TASK_SUITES[suite]))
print("sequence:", default_sequence(suite))

shapes = set()
for task_id, spec in enumerate(TASK_SUITES[suite]):
    env = get_task(task_id, task_suite=suite)
    obs, info = env.reset(seed=0)
    out = env.step(env.action_space.sample())
    _, _, _, _, step_info = out
    shape = (
        int(np.prod(env.observation_space.shape)),
        int(np.prod(env.action_space.shape)),
    )
    shapes.add(shape)
    print(
        f"  {task_id:2d}: {spec.name:28s} "
        f"obs={shape[0]} act={shape[1]} api={getattr(env.env, 'metaworld_api', '?')} "
        f"success_key={'success' in step_info}"
    )
    env.close()

if len(shapes) != 1:
    raise RuntimeError(f"Inconsistent Meta-World shapes: {shapes}")

print("[ok] common shape:", shapes.pop())
PY

# ------------------------------------------------------------
# Pool structural tests
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
