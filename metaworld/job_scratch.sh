#!/bin/bash
#SBATCH --job-name=causal
#SBATCH --output=logs/mw-scratch_%j.out
#SBATCH --error=logs/mw-scratch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --qos=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-artifacts/ethos_student_metaworld_paper10_150k}"
SCRATCH_ROOT_BASE="$EXPERIMENT_ROOT/scratch"
JOB_ID_FILE="$SCRATCH_ROOT_BASE/job_ids.env"
SCRATCH_SEEDS=(101 102 103)
EVAL_MODES=(deterministic stochastic)

SCRATCH_ARGS=(
    --task-suites mw_paper10
    --variants plain
    --total-timesteps 1000000
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


if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p logs "$SCRATCH_ROOT_BASE"
    SCRIPT_PATH="$(realpath "$0")"
    : > "$JOB_ID_FILE"

    echo "============================================================"
    echo "Submitting MetaWorld paper10 FT scratch baselines"
    echo "Modes: ${EVAL_MODES[*]}"
    echo "Scratch seeds: ${SCRATCH_SEEDS[*]}"
    echo "Root: $SCRATCH_ROOT_BASE"
    echo "============================================================"

    for mode in "${EVAL_MODES[@]}"; do
        ids=()
        for seed in "${SCRATCH_SEEDS[@]}"; do
            job_id="$(
                sbatch --parsable                     --job-name="causal"                     --output="logs/scratch_${mode}_seed${seed}_%j.out"                     --error="logs/scratch_${mode}_seed${seed}_%j.err"                     "$SCRIPT_PATH" --worker "$mode" "$seed"
            )"
            job_id="${job_id%%;*}"
            ids+=("$job_id")
            echo "[submitted] $mode seed $seed -> $job_id"
        done
        joined="$(IFS=:; echo "${ids[*]}")"
        if [[ "$mode" == "deterministic" ]]; then
            printf 'SCRATCH_DETERMINISTIC_JOB_IDS="%s"
' "$joined" >> "$JOB_ID_FILE"
        else
            printf 'SCRATCH_STOCHASTIC_JOB_IDS="%s"
' "$joined" >> "$JOB_ID_FILE"
        fi
    done

    echo "[saved] dependency IDs -> $JOB_ID_FILE"
    echo "Now run: bash job.sh"
    echo "No manual wait is needed; the main jobs use SLURM dependencies."
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


module purge
module load gcc/13.2.0 python/3.9.18 py-virtualenv/20.24.5

VENV="${CKA_VENV:-$HOME/.venvs/cka_metaworld}"
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

MW_COMMIT="${MW_COMMIT:-c822f28f582ba1ad49eb5dcf61016566f28003ba}"
python -m pip install -q "mujoco>=3.0,<4"
MW_MARKER="$VENV/.metaworld_commit"
CURRENT_MW_COMMIT=""
if [[ -f "$MW_MARKER" ]]; then
    CURRENT_MW_COMMIT="$(cat "$MW_MARKER")"
fi
if [[ "$CURRENT_MW_COMMIT" != "$MW_COMMIT" ]] || ! python -c "import metaworld" >/dev/null 2>&1; then
    echo "[setup] installing pinned MetaWorld commit: $MW_COMMIT"
    python -m pip uninstall -y metaworld >/dev/null 2>&1 || true
    python -m pip install --no-deps "git+https://github.com/Farama-Foundation/Metaworld.git@$MW_COMMIT"
    printf '%s\n' "$MW_COMMIT" > "$MW_MARKER"
fi
python -m pip install -q -r requirements.txt

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
import metaworld
print("MetaWorld:", metaworld.__file__)
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


echo "============================================================"
echo "[scratch] environment: MetaWorld paper10"
echo "[scratch] mode:        $EVAL_MODE"
echo "[scratch] seed:        $SCRATCH_SEED"
echo "[scratch] suite:       mw_paper10"
echo "[scratch] root:        $SCRATCH_MODE_ROOT"
echo "============================================================"

python -u scratch_baselines.py     "${SCRATCH_ARGS[@]}"     --seeds "$SCRATCH_SEED"     --eval-action-mode "$EVAL_MODE"     --save-root "$SCRATCH_MODE_ROOT/models"     --runs-root "$SCRATCH_MODE_ROOT/runs"     --analysis-root "$SCRATCH_MODE_ROOT/analysis"

echo "[done] scratch $EVAL_MODE seed $SCRATCH_SEED"
