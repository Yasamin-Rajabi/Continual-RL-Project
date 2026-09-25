#!/bin/bash
# Compatibility helper for the MiniGrid directory. The canonical paper pipeline
# is repository-level job_paper.sh; this script only exposes local checks.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

case "${1:-}" in
  sanity)
    python sanity_check_pool.py
    ;;
  check-env)
    python tasks.py --check doorkey4
    ;;
  pilot)
    shift
    python pilot_check.py "$@"
    ;;
  smoke)
    python run_continual_benchmark.py --quick-test
    ;;
  *)
    cat <<'USAGE'
Usage: bash run_kaggle.sh <command>

  sanity       CPU-only pool/composition/distillation checks
  check-env    instantiate every task in doorkey4
  pilot [...]  run the 60k single-task budget screen
  smoke        tiny end-to-end continual benchmark

For paper runs use, from the repository root:
  bash job_paper.sh --environments minigrid --groups main kl lineage pool warmup --comment paper_v1 --submit
USAGE
    ;;
esac
