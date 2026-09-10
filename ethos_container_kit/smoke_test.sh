#!/bin/bash
set -euo pipefail

IMAGE="${1:-$HOME/containers/ethos_crl_torch280.sif}"
REPO="${2:-$HOME/Cont/Continual-RL-Project}"

if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: image not found: $IMAGE" >&2
    exit 2
fi
if [[ ! -d "$REPO" ]]; then
    echo "ERROR: repo not found: $REPO" >&2
    exit 2
fi

export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MPLBACKEND=Agg

apptainer exec --nv --bind "$REPO:$REPO" "$IMAGE" python - <<'PY'
import torch
import gymnasium
import stable_baselines3
import mujoco
import imageio
import metaworld

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("gymnasium:", gymnasium.__version__)
print("SB3:", stable_baselines3.__version__)
print("mujoco:", mujoco.__version__)
print("imageio:", imageio.__version__)
print("metaworld:", metaworld.__file__)
PY

for d in half-cheetah Walker2D; do
    if [[ -f "$REPO/$d/sanity_check_pool.py" ]]; then
        echo "[test] $d sanity_check_pool.py"
        apptainer exec --nv --bind "$REPO:$REPO" "$IMAGE" \
            python "$REPO/$d/sanity_check_pool.py"
    fi
done

if [[ -f "$REPO/metaworld/sanity_check_pool.py" ]]; then
    echo "[test] metaworld sanity_check_pool.py"
    apptainer exec --nv --bind "$REPO:$REPO" "$IMAGE" \
        python "$REPO/metaworld/sanity_check_pool.py"
fi
