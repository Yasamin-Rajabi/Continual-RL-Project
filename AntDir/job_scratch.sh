#!/bin/bash
# New AntDir reference jobs take their core settings from AntDir/job.sh.
# This avoids keeping a second, potentially inconsistent set of defaults.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/job_paper.sh" --environments AntDir --phase scratch --submit "$@"
