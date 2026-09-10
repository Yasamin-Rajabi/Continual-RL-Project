FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    libgl1 \
    libegl1 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_ethos.txt /tmp/requirements_ethos.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements_ethos.txt \
    && python -m pip install "mujoco>=3.0,<4"

ARG METAWORLD_COMMIT=c822f28f582ba1ad49eb5dcf61016566f28003ba
RUN python -m pip install --no-deps \
    "git+https://github.com/Farama-Foundation/Metaworld.git@${METAWORLD_COMMIT}"

RUN python - <<'PY'
import torch
import gymnasium
import stable_baselines3
import mujoco
import imageio
import metaworld

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("gymnasium:", gymnasium.__version__)
print("stable_baselines3:", stable_baselines3.__version__)
print("mujoco:", mujoco.__version__)
print("imageio:", imageio.__version__)
print("metaworld:", metaworld.__file__)

assert torch.__version__.split("+")[0] == "2.8.0"
assert torch.version.cuda == "12.6"
assert int(mujoco.__version__.split(".")[0]) >= 3
PY

RUN python -m pip freeze > /opt/ethos-pip-freeze.txt
WORKDIR /workspace
