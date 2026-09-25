#!/bin/bash
#SBATCH --job-name=causal
set -euo pipefail
SPEC="${1:?Pass the frozen job-spec JSON}"
# job specs are written by the launcher before sbatch, not regenerated from
# a possibly edited job.sh while the task waits in the queue.
mapfile -t SETTINGS < <(python3 - "$SPEC" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
print(s['project_root']);print(s['environment']);print(s['IMAGE'])
PY
)
PROJECT_ROOT="${SETTINGS[0]}"; ENV_DIR="${SETTINGS[1]}"; IMAGE="${SETTINGS[2]}"
[[ -r "$IMAGE" ]] || { echo "Missing image: $IMAGE" >&2; exit 2; }
export APPTAINERENV_PYTHONNOUSERSITE=1
# Optional project-local dependencies. MiniGrid is not part of every MuJoCo
# container, so minigrid/setup_cluster_env.sh can install only that package
# under this bind-mounted directory without modifying the immutable image.
if [[ "$ENV_DIR" == "minigrid" && -d "$PROJECT_ROOT/.paper_deps/minigrid" ]]; then
    export APPTAINERENV_PYTHONPATH="$PROJECT_ROOT/.paper_deps/minigrid${PYTHONPATH:+:$PYTHONPATH}"
fi
export APPTAINERENV_MPLBACKEND=Agg
export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export APPTAINERENV_MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
unset PYTHONHOME PYTHONPATH
printf '[worker] host=%s image=%s spec=%s\n' "$(hostname)" "$IMAGE" "$SPEC"
exec apptainer exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --pwd "$PROJECT_ROOT/$ENV_DIR" "$IMAGE" \
    /opt/conda/bin/python "$PROJECT_ROOT/paper_runs/worker.py" --spec "$SPEC"
