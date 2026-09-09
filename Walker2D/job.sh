#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --output=logs/w2d-main_%j.out
#SBATCH --error=logs/w2d-main_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --qos=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-artifacts/ethos_student_walker2d_150k}"
SCRATCH_ROOT_BASE="$EXPERIMENT_ROOT/scratch"
SCRATCH_JOB_ID_FILE="$SCRATCH_ROOT_BASE/job_ids.env"
SCRATCH_SEEDS=(101 102 103)

VARIANTS=(baseline combined combined_policy combined_policy_student)
EVAL_MODES=(deterministic stochastic)

COMMON_ARGS=(
    --task-suites walker2d_dynamics
    --seeds 1 2 3
    --total-timesteps 320000
    --pool-size 5
    --batch-size 256
    --policy-lr 3e-4
    --alpha-lr 5e-3
    --alpha-mass-reg 0.05
    --alpha-warmup-steps 5000
    --alpha-entropy-reg 0.01
    --drift-reg 1.0
    --distill-encoder-lr-mult 0.1
    --q-lr 3e-4
    --gamma 0.99
    --tau 0.005
    --alpha 0.2
    --autotune
    --autotune-init-from-alpha
    --learning-starts 5000
    --random-actions-end 5000
    --eval-every 5000
    --num-evals 5
    --retention-eval-episodes 5
    --test-adapt-steps 0
    --frozen-eval-policy pool
    --no-distill-observation-skip
    --distill-buffer-steps 5000
    --similarity-samples 2048
    --max-distill-buffer 50000
    --distill-max-samples 20000
    --distill-epochs 16
    --distill-lr 5e-4
    --distill-batch-size 256
    --distill-test-frac 0.2
    --distill-select-best-val
    --analysis-log-every 5000
    --no-train-shared
    --no-freeze-root-encoder
    --encoder-from-base
    --no-encoder-linear-out
    --no-condition-alpha-scale
    --no-use-alpha-scale
    --no-fix-alpha-scale
    --weight-use-alpha-mass
    --constrain-alpha-mass
)


COMBINED_POLICY_ARGS=(
    --projection-epochs 32
    --projection-max-samples 50000
)

COMBINED_POLICY_STUDENT_ARGS=(
    --policy-student-replay
)

variant_mapping() {
    local variant="$1"
    case "$variant" in
        baseline)        echo "1 parameter" ;;
        combined)        echo "4 parameter" ;;
        combined_policy)         echo "4 policy" ;;
        combined_policy_student) echo "4 policy" ;;
        *) return 1 ;;
    esac
}

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p logs "$EXPERIMENT_ROOT/main"
    if [[ ! -f "$SCRATCH_JOB_ID_FILE" ]]; then
        echo "ERROR: missing $SCRATCH_JOB_ID_FILE" >&2
        echo "Run 'bash job_scratch.sh' first. You do not need to wait for it to finish." >&2
        exit 2
    fi
    source "$SCRATCH_JOB_ID_FILE"
    : "${SCRATCH_DETERMINISTIC_JOB_IDS:?Missing deterministic scratch IDs}"
    : "${SCRATCH_STOCHASTIC_JOB_IDS:?Missing stochastic scratch IDs}"

    SCRIPT_PATH="$(realpath "$0")"
    echo "============================================================"
    echo "Submitting Walker2D main runs"
    echo "Methods: ${VARIANTS[*]}"
    echo "Modes: ${EVAL_MODES[*]}"
    echo "FT: enabled"
    echo "============================================================"

    for mode in "${EVAL_MODES[@]}"; do
        if [[ "$mode" == "deterministic" ]]; then
            dep_ids="$SCRATCH_DETERMINISTIC_JOB_IDS"
        else
            dep_ids="$SCRATCH_STOCHASTIC_JOB_IDS"
        fi
        for variant in "${VARIANTS[@]}"; do
            safe_variant="${variant//_/-}"
            job_id="$(
                sbatch --parsable                     --dependency="afterok:${dep_ids}"                     --job-name="causal"                     --output="logs/${variant}_${mode}_%j.out"                     --error="logs/${variant}_${mode}_%j.err"                     "$SCRIPT_PATH" --worker "$variant" "$mode"
            )"
            job_id="${job_id%%;*}"
            echo "[submitted] $variant / $mode -> $job_id (afterok:$dep_ids)"
        done
    done
    echo "All eight main jobs submitted. They wait for matching scratch jobs."
    exit 0
fi

VARIANT="${2:?Missing variant}"
EVAL_MODE="${3:?Missing evaluation mode}"
read -r CONDITION_INDEX COMPOSITION_SPACE <<<"$(variant_mapping "$VARIANT")" || {
    echo "ERROR: unsupported variant: $VARIANT" >&2
    exit 2
}
if [[ "$EVAL_MODE" != "deterministic" && "$EVAL_MODE" != "stochastic" ]]; then
    echo "ERROR: evaluation mode must be deterministic or stochastic" >&2
    exit 2
fi

RUN_ROOT="$EXPERIMENT_ROOT/main/${VARIANT}_${EVAL_MODE}"
SCRATCH_MODE_ROOT="$SCRATCH_ROOT_BASE/$EVAL_MODE"
mkdir -p "$RUN_ROOT/agents" "$RUN_ROOT/runs" "$RUN_ROOT/plots" "$RUN_ROOT/analysis"


module purge
module load gcc/13.2.0 python/3.9.18 py-virtualenv/20.24.5

VENV="${CKA_VENV:-$HOME/.venvs/cka_walker2d}"
VENV_LOCK="${VENV}.setup.lock"
mkdir -p "$(dirname "$VENV")"
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

TORCH_VERSION="2.8.0"
TORCH_CUDA="12.6"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
NEED_TORCH=1
if python - <<'TORCHCHECK' >/dev/null 2>&1
import torch
assert torch.__version__.split("+")[0] == "2.8.0"
assert torch.version.cuda == "12.6"
TORCHCHECK
then
    NEED_TORCH=0
fi
if (( NEED_TORCH )); then
    echo "[setup] installing torch $TORCH_VERSION + CUDA $TORCH_CUDA"
    python -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true
    python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi

flock -u 9

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "$GPU_NAME" != *H100* ]]; then
    echo "ERROR: expected H100, got: $GPU_NAME" >&2
    exit 1
fi

echo "[gpu] $GPU_NAME"
python - <<'VERIFY'
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
VERIFY
python -m pip check


if [[ ! -d "$SCRATCH_MODE_ROOT/runs/scratch" ]]; then
    echo "ERROR: scratch TensorBoard logs are missing: $SCRATCH_MODE_ROOT/runs/scratch" >&2
    exit 3
fi
if [[ -e "$RUN_ROOT/runs/scratch" && ! -L "$RUN_ROOT/runs/scratch" ]]; then
    echo "ERROR: $RUN_ROOT/runs/scratch exists and is not a symlink" >&2
    exit 3
fi
rm -f "$RUN_ROOT/runs/scratch"
ln -s "$(realpath "$SCRATCH_MODE_ROOT/runs/scratch")" "$RUN_ROOT/runs/scratch"

VARIANT_ARGS=()
case "$VARIANT" in
    combined_policy)
        VARIANT_ARGS=("${COMBINED_POLICY_ARGS[@]}")
        ;;
    combined_policy_student)
        VARIANT_ARGS=("${COMBINED_POLICY_STUDENT_ARGS[@]}")
        ;;
esac

echo "============================================================"
echo "[main] environment:  Walker2D"
echo "[main] variant:      $VARIANT"
echo "[main] evaluation:   $EVAL_MODE"
echo "[main] condition:    $CONDITION_INDEX"
echo "[main] composition:  $COMPOSITION_SPACE"
echo "[main] FT scratch:   $SCRATCH_MODE_ROOT"
echo "[main] output:       $RUN_ROOT"
echo "============================================================"

python -u sanity_check_pool.py

srun python -u run_continual_benchmark.py     "${COMMON_ARGS[@]}"     "${VARIANT_ARGS[@]}"     --eval-action-mode "$EVAL_MODE"     --condition-index "$CONDITION_INDEX"     --composition-spaces "$COMPOSITION_SPACE"     --scratch-seeds "${SCRATCH_SEEDS[@]}"     --scratch-save-root "$SCRATCH_MODE_ROOT/models"     --save-root "$RUN_ROOT/agents"     --runs-root "$RUN_ROOT/runs"     --plots-root "$RUN_ROOT/plots"     --analysis-root "$RUN_ROOT/analysis"

echo "============================================================"
echo "[done] $VARIANT / $EVAL_MODE"
echo "[done] survey metrics (including FT): $RUN_ROOT/plots/walker2d_dynamics/survey_metrics.csv"
echo "[done] plots: $RUN_ROOT/plots"
echo "============================================================"
