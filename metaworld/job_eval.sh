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
COMMENT="${RUN_COMMENT:-}"
if [[ "$MODE" == "--worker" ]]; then
    COMMENT="${4:-${RUN_COMMENT:-}}"
    REPO_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --comment)
                [[ $# -ge 2 ]] || { echo "ERROR: --comment needs a value" >&2; exit 2; }
                COMMENT="$2"
                shift 2
                ;;
            --comment=*)
                COMMENT="${1#--comment=}"
                shift
                ;;
            -h|--help)
                echo "Usage: bash ${BASH_SOURCE[0]} [--comment NAME]"
                echo "Example: bash ${BASH_SOURCE[0]} --comment amass"
                exit 0
                ;;
            *)
                echo "ERROR: unknown argument: $1" >&2
                exit 2
                ;;
        esac
    done
fi
if [[ -n "$COMMENT" && ! "$COMMENT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: --comment may contain only letters, digits, '.', '_' and '-' and must start with a letter/digit" >&2
    exit 2
fi
COMMENT_SUFFIX=""
if [[ -n "$COMMENT" ]]; then
    COMMENT_SUFFIX="_${COMMENT}"
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

if [[ ! -f run_continual_benchmark.py ]]; then
    echo "ERROR: run_continual_benchmark.py not found in repo directory: $PWD" >&2
    exit 2
fi
if [[ ! -f sanity_check_pool.py ]]; then
    echo "ERROR: sanity_check_pool.py not found in repo directory: $PWD" >&2
    exit 2
fi

VARIANTS=(baseline combined combined_policy combined_policy_student)

COMMON_ARGS=(
    --skip-training
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
    --retention-eval-episodes 5
    --test-adapt-steps 5000
    --frozen-eval-policy pool
    --no-distill-observation-skip
    --distill-buffer-steps 10000
    --similarity-samples 2048
    --no-balance-source-lineages
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
        baseline)                echo "1 parameter" ;;
        combined)                echo "4 parameter" ;;
        combined_policy)         echo "4 policy" ;;
        combined_policy_student) echo "4 policy" ;;
        *) return 1 ;;
    esac
}

if [[ "$MODE" != "--worker" ]]; then
    mkdir -p "$LOG_ROOT" "$EXPERIMENT_ROOT/main"
    SCRIPT_PATH="$(realpath "$0")"
    echo "============================================================"
    echo "Submitting MetaWorld paper10 main runs"
    echo "Methods: ${VARIANTS[*]}"
    echo "Modes: ${EVAL_MODES[*]}"
    echo "Comment: ${COMMENT:-<none>}"
    echo "FT: enabled"
    echo "Training: DISABLED (--skip-training); existing checkpoints only"
    echo "Container: $IMAGE"
    echo "============================================================"

    for mode in "${EVAL_MODES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            job_id="$(
                sbatch --parsable \
                    --job-name="causal" \
                    --output="$LOG_ROOT/eval_${variant}_${mode}${COMMENT_SUFFIX}_%j.out" \
                    --error="$LOG_ROOT/eval_${variant}_${mode}${COMMENT_SUFFIX}_%j.err" \
                    "$SCRIPT_PATH" --worker "$variant" "$mode" "$COMMENT"
            )"
            job_id="${job_id%%;*}"
            echo "[submitted] $variant / $mode -> $job_id"
        done
    done

    echo "All eight main jobs submitted. job.sh does not create SLURM dependencies on scratch jobs."
    echo "Make sure the matching scratch baselines are complete before submitting job.sh."
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

RUN_ROOT="$EXPERIMENT_ROOT/main/${VARIANT}_${EVAL_MODE}${COMMENT_SUFFIX}"
SCRATCH_MODE_ROOT="$SCRATCH_ROOT_BASE/$EVAL_MODE"
mkdir -p "$RUN_ROOT/agents" "$RUN_ROOT/runs" "$RUN_ROOT/plots" "$RUN_ROOT/analysis"

prepare_container_runtime
verify_container_runtime

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
echo "[main] environment:  MetaWorld paper10"
echo "[main] variant:      $VARIANT"
echo "[main] evaluation:   $EVAL_MODE"
echo "[main] comment:      ${COMMENT:-<none>}"
echo "[main] condition:    $CONDITION_INDEX"
echo "[main] composition:  $COMPOSITION_SPACE"
echo "[main] FT scratch:   $SCRATCH_MODE_ROOT"
echo "[main] output:       $RUN_ROOT"
echo "[main] container:    $IMAGE"
echo "============================================================"

run_in_container python -u "$REPO_DIR/sanity_check_pool.py"

run_in_container python -u "$REPO_DIR/run_continual_benchmark.py" \
    "${COMMON_ARGS[@]}" \
    "${VARIANT_ARGS[@]}" \
    --eval-action-mode "$EVAL_MODE" \
    --condition-index "$CONDITION_INDEX" \
    --composition-spaces "$COMPOSITION_SPACE" \
    --scratch-seeds "${SCRATCH_SEEDS[@]}" \
    --scratch-save-root "$SCRATCH_MODE_ROOT/models" \
    --save-root "$RUN_ROOT/agents" \
    --runs-root "$RUN_ROOT/runs" \
    --plots-root "$RUN_ROOT/plots" \
    --analysis-root "$RUN_ROOT/analysis"

echo "============================================================"
echo "[done] $VARIANT / $EVAL_MODE"
echo "[done] survey metrics: $RUN_ROOT/plots/mw_paper10/survey_metrics.csv"
echo "[done] plots: $RUN_ROOT/plots"
echo "============================================================"
