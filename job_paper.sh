#!/bin/bash
# Additive batch launcher; existing environment job.sh files are untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/paper_experiments.py" "$@"
