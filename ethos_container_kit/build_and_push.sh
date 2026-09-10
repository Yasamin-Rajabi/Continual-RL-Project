#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <registry-image> [platform]" >&2
    echo "Example: $0 docker.io/myuser/ethos-crl:torch280 linux/amd64" >&2
    exit 2
fi

IMAGE="$1"
PLATFORM="${2:-linux/amd64}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[build] image:    $IMAGE"
echo "[build] platform: $PLATFORM"

docker build --platform "$PLATFORM" -t "$IMAGE" .
docker push "$IMAGE"

echo "[done] pushed $IMAGE"
