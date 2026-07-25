#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${ELARA_PYTHON:-python3}"
"$python_bin" -m ELARA.fair_baseline_runner "$@"
