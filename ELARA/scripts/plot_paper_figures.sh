#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_executable="${ELARA_PYTHON:-python3}"
"$python_executable" -m ELARA.plot_paper_figures \
  --temporal-bin-slots 5 \
  --temporal-smoothing-window 7 \
  "$@"
