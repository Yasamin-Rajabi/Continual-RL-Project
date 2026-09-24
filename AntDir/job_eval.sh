#!/bin/bash
# Delegate to the same AntDir settings used for training.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/job.sh" --aggregate "$@"
