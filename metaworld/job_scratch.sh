#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --qos=normal

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" == "--worker" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO_DIR"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Cont/Continual-RL-Project}"
IMAGE="${ETHOS_IMAGE:-$HOME/containers/ethos_crl_torch280.sif}"
BASE_STORAGE="${BASE_STORAGE:-$PROJECT_ROOT/crl_experiments}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$BASE_STORAGE/ethos_student_metaworld_paper10_500k}"
LOG_ROOT="$EXPERIMENT_ROOT/logs"
SCRATCH_ROOT_BASE="$EXPERIMENT_ROOT/scratch"
SCRATCH_SEEDS=(101 102 103)
EVAL_MODES=(deterministic stochastic)

clean_host_python_env() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        local inherited_bin="${VIRTUAL_ENV%/}/bin"
        PATH=":$PATH:"
        PATH="${PATH//:$inherited_bin:/:}"
        PATH="${PATH#:}"
        PATH="${PATH%:}"
        export PATH
        unset VIRTUAL_ENV
    fi
    unset PYTHONHOME || true
    hash -r
}

prepare_container_runtime() {
    clean_host_python_env
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "ERROR: apptainer is not available on this node." >&2
        exit 3
    fi
    if [[ ! -r "$IMAGE" ]]; then
        echo "ERROR: container image not found/readable: $IMAGE" >&2
        exit 3
    fi

    export APPTAINERENV_MUJOCO_GL=egl
    export APPTAINERENV_PYOPENGL_PLATFORM=egl
    export APPTAINERENV_EGL_DEVICE_ID=0
    export APPTAINERENV_MUJOCO_EGL_DEVICE_ID=0
    export APPTAINERENV_MPLBACKEND=Agg
    export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export APPTAINERENV_MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export APPTAINERENV_PYTHONNOUSERSITE=1
}

verify_container_runtime() {
    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    if [[ "$gpu_name" != *H100* ]]; then
        echo "ERROR: expected H100, got: $gpu_name" >&2
        exit 1
    fi
    echo "[gpu] $gpu_name"
    echo "[container] $IMAGE"

    apptainer exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$IMAGE" python - <<'PYVERIFY'
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
print("MetaWorld:", metaworld.__file__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside H100 allocation")
if "H100" not in torch.cuda.get_device_name(0):
    raise RuntimeError(f"Expected H100 inside container, got {torch.cuda.get_device_name(0)}")
print("GPU:", torch.cuda.get_device_name(0))
PYVERIFY
}

run_in_container() {
    apptainer exec --nv \
        --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
        "$IMAGE" "$@"
}

if [[ ! -f scratch_baselines.py ]]; then
    echo "ERROR: scratch_baselines.py not found in repo directory: $PWD" >&2
    exit 2
fi

SCRATCH_ARGS=(
    --task-suites mw_paper10
    --seeds 1 2 3
    --total-timesteps 500000
    --pool-size 8
    --batch-size 128
    --policy-lr 1e-3
    --alpha-lr 5e-3
    --alpha-mass-reg 0.05
    --alpha-warmup-steps 5000
    --alpha-entropy-reg 0.01
    --drift-reg 1.0
    --distill-encoder-lr-mult 0.1
    --q-lr 1e-3
    --gamma 0.99
    --tau 0.005
    --alpha 0.2
    --autotune
    --autotune-init-from-alpha
    --learning-starts 5000
    --random-actions-end 5000
    --eval-every 10000
    --num-evals 5
    --no-distill-observation-skip
    --distill-buffer-steps 10000
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
    --constrain-alpha-mass
)

if [[ "$MODE" != "--worker" ]]; then
    mkdir -p "$LOG_ROOT" "$SCRATCH_ROOT_BASE"
    SCRIPT_PATH="$(realpath "$0")"
    echo "============================================================"
    echo "Submitting MetaWorld paper10 FT scratch baselines"
    echo "Modes: ${EVAL_MODES[*]}"
    echo "Scratch seeds: ${SCRATCH_SEEDS[*]}"
    echo "Container: $IMAGE"
    echo "Root: $SCRATCH_ROOT_BASE"
    echo "============================================================"

    for mode in "${EVAL_MODES[@]}"; do
        for seed in "${SCRATCH_SEEDS[@]}"; do
            job_id="$(
                sbatch --parsable \
                    --job-name="causal" \
                    --output="$LOG_ROOT/scratch_${mode}_seed${seed}_%j.out" \
                    --error="$LOG_ROOT/scratch_${mode}_seed${seed}_%j.err" \
                    "$SCRIPT_PATH" --worker "$mode" "$seed"
            )"
            job_id="${job_id%%;*}"
            echo "[submitted] $mode seed $seed -> $job_id"
        done

    done

    echo "All six scratch jobs submitted."
    echo "Wait for them to finish successfully before running: bash job.sh"
    exit 0
fi

EVAL_MODE="${2:?Missing evaluation mode}"
SCRATCH_SEED="${3:?Missing scratch seed}"
if [[ "$EVAL_MODE" != "deterministic" && "$EVAL_MODE" != "stochastic" ]]; then
    echo "ERROR: evaluation mode must be deterministic or stochastic" >&2
    exit 2
fi

SCRATCH_MODE_ROOT="$SCRATCH_ROOT_BASE/$EVAL_MODE"
mkdir -p "$SCRATCH_MODE_ROOT/models" "$SCRATCH_MODE_ROOT/runs" "$SCRATCH_MODE_ROOT/analysis"

prepare_container_runtime
verify_container_runtime

echo "============================================================"
echo "[scratch] environment: MetaWorld paper10"
echo "[scratch] mode:        $EVAL_MODE"
echo "[scratch] seed:        $SCRATCH_SEED"
echo "[scratch] suite:       mw_paper10"
echo "[scratch] root:        $SCRATCH_MODE_ROOT"
echo "[scratch] container:   $IMAGE"
echo "============================================================"

run_in_container python -u "$REPO_DIR/scratch_baselines.py" \
    "${SCRATCH_ARGS[@]}" \
    --seeds "$SCRATCH_SEED" \
    --eval-action-mode "$EVAL_MODE" \
    --save-root "$SCRATCH_MODE_ROOT/models" \
    --runs-root "$SCRATCH_MODE_ROOT/runs" \
    --analysis-root "$SCRATCH_MODE_ROOT/analysis"

echo "[done] scratch $EVAL_MODE seed $SCRATCH_SEED"
