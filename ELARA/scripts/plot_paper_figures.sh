#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_executable="${ELARA_PYTHON:-python3}"
"$python_executable" -m ELARA.plot_paper_figures \
  --baseline-root ELARA/outputs/baseline-tests2 \
  --bandit-root ELARA/outputs/baseline-tests2 \
  --sensitivity-root ELARA/outputs/sensitivity \
  --output-dir ELARA/paper_fig4 \
  --temporal-bin-slots 5 \
  --temporal-smoothing-window 7 \
  --minimum-seeds 4 \
  "$@"
