#!/bin/bash
# One-time bootstrap for the paper MiniGrid environment when the immutable
# MuJoCo container does not already ship the third-party `minigrid` package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${ETHOS_IMAGE:-$HOME/containers/ethos_crl_torch280_mj237.sif}"
DEPS="$PROJECT_ROOT/.paper_deps/minigrid"

command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found" >&2; exit 2; }
[[ -r "$IMAGE" ]] || { echo "ERROR: container image not readable: $IMAGE" >&2; exit 2; }
mkdir -p "$DEPS"

check_cmd=(apptainer exec --bind "$PROJECT_ROOT:$PROJECT_ROOT" --pwd "$HERE" "$IMAGE" /opt/conda/bin/python)
if APPTAINERENV_PYTHONPATH="$DEPS" "${check_cmd[@]}" - <<'PY' >/dev/null 2>&1
import gymnasium, minigrid
from tasks import get_task
e=get_task(0, task_suite='doorkey4'); e.reset(seed=0); e.close()
PY
then
    echo "MiniGrid runtime already available."
else
    echo "Installing MiniGrid into $DEPS (the container itself is not modified)..."
    # Keep the image's PyTorch/Numpy/Gymnasium/SB3. Only add the packages that
    # the MuJoCo image may lack, avoiding dependency upgrades that could change
    # the existing benchmark runtime.
    apptainer exec --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$IMAGE" \
        /opt/conda/bin/python -m pip install --no-deps --upgrade \
        --target "$DEPS" 'minigrid>=2.3,<3' 'pygame>=2.4,<3'
fi

export APPTAINERENV_PYTHONPATH="$DEPS"
"${check_cmd[@]}" tasks.py --check doorkey4
"${check_cmd[@]}" sanity_check_pool.py

echo
printf 'MiniGrid cluster runtime is ready. Dependencies: %s\n' "$DEPS"
printf 'Next: bash job_paper.sh --environments minigrid --groups main --methods combined_policy --seeds 1 --comment mg_pilot --submit\n'
