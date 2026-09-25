#!/bin/bash
#SBATCH --job-name=ethos-mg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --exclude=kh023,kh032

# This file is the declarative preset read by ../job_paper.sh.
# Use the paper launcher rather than executing this file directly.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Cont/Continual-RL-Project}"
IMAGE="${ETHOS_IMAGE:-$HOME/containers/ethos_crl_torch280_mj237.sif}"
BASE_STORAGE="${BASE_STORAGE:-$PROJECT_ROOT/crl_experiments}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$BASE_STORAGE/ethos_student_minigrid_doorkey4_60k}"

SCRATCH_SEEDS=(101 102)
MAIN_SEEDS=(1 2)
EVAL_MODES=(deterministic)

# MiniGrid counterpart of the HalfCheetah paper protocol. Delta=60k total
# interactions per occurrence; the final B=5k frozen rollout is INSIDE Delta.
COMMON_ARGS=(
    --task-suites doorkey4
    --total-timesteps 60000
    --pool-size 4
    --batch-size 256
    --policy-lr 3e-4
    --alpha-lr 5e-3
    --alpha-mass-reg 0.0
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
    --learning-starts 1000
    --random-actions-end 2000
    --eval-every 2500
    --num-evals 5
    --retention-eval-episodes 5
    --test-adapt-steps 5000
    --frozen-eval-policy pool
    --no-distill-observation-skip
    --distill-buffer-steps 5000
    --similarity-samples 2048
    --balance-source-lineages
    --max-distill-buffer 50000
    --distill-max-samples 20000
    --distill-epochs 16
    --distill-lr 5e-4
    --distill-batch-size 256
    --distill-test-frac 0.2
    --distill-select-best-val
    --analysis-log-every 2500
    --compact-storage
    --no-save-analysis-snapshots
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
    --projection-max-samples 5000
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat <<'EOF'
MiniGrid is integrated into the paper launcher.

Preview:
  bash ../job_paper.sh --environments minigrid --groups main kl lineage pool warmup --comment paper_v1

Submit:
  bash ../job_paper.sh --environments minigrid --groups main kl lineage pool warmup --comment paper_v1 --submit
EOF
fi
