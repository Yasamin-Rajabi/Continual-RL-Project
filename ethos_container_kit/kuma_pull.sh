#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <registry-image> [output-sif]" >&2
    echo "Example: $0 docker.io/myuser/ethos-crl:torch280 \"$HOME/containers/ethos_crl_torch280.sif\"" >&2
    exit 2
fi

IMAGE="$1"
SIF="${2:-$HOME/containers/ethos_crl_torch280.sif}"
mkdir -p "$(dirname "$SIF")"

# If your cluster exposes Apptainer through a module, uncomment/adjust this:
# module load apptainer

apptainer pull --force "$SIF" "docker://$IMAGE"
sha256sum "$SIF" | tee "${SIF}.sha256"

echo "[done] SIF: $SIF"
echo "[done] SHA256: ${SIF}.sha256"
