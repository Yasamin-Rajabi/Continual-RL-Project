# Ethos CRL container kit

One dependency image for HalfCheetah, Walker2D, and MetaWorld.

The image contains PyTorch 2.8.0 / CUDA 12.6, Gymnasium, Stable-Baselines3,
MuJoCo 3.x, TensorBoard, and MetaWorld at commit
`c822f28f582ba1ad49eb5dcf61016566f28003ba`.

The research repository is intentionally NOT copied into the image. Bind the
live repository at runtime so code edits do not require rebuilding the image.

## Files

- `Dockerfile`: immutable dependency image.
- `requirements_ethos.txt`: merged runtime requirements for the three benchmarks.
- `build_and_push.sh`: build and push the Docker image from a machine with Docker.
- `kuma_pull.sh`: pull the registry image into an Apptainer `.sif` on Kuma.
- `smoke_test.sh`: test CUDA, MuJoCo, MetaWorld, and available pool sanity scripts.

## Build machine

```bash
docker login
./build_and_push.sh docker.io/YOUR_USERNAME/ethos-crl:torch280
```

On Apple Silicon the script already defaults to `linux/amd64`, which is the
architecture required by the PyTorch 2.8.0 CUDA 12.6 base image used here.

## Kuma

Copy this kit to Kuma (or just use the commands directly), then:

```bash
./kuma_pull.sh docker.io/YOUR_USERNAME/ethos-crl:torch280 \
  "$HOME/containers/ethos_crl_torch280.sif"
```

Obtain an H100 allocation using your normal SCITAS allocation command, then:

```bash
./smoke_test.sh \
  "$HOME/containers/ethos_crl_torch280.sif" \
  "$HOME/Cont/Continual-RL-Project"
```

## Runtime pattern for SLURM

Use the same `.sif` for all three environments:

```bash
IMAGE="$HOME/containers/ethos_crl_torch280.sif"
REPO="$HOME/Cont/Continual-RL-Project"

export APPTAINERENV_MUJOCO_GL=egl
export APPTAINERENV_PYOPENGL_PLATFORM=egl
export APPTAINERENV_MPLBACKEND=Agg

srun apptainer exec --nv \
  --bind "$REPO:$REPO" \
  "$IMAGE" \
  python -u "$REPO/half-cheetah/run_continual_benchmark.py" ...
```

Change only the project subdirectory for Walker2D or MetaWorld.

Keep experiment results on persistent storage, e.g.:

```bash
BASE_STORAGE="$HOME/Cont/Continual-RL-Project/crl_experiments"
```

Do not run pip from the training jobs once the SIF is in use.
