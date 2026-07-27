#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_executable="${ELARA_PYTHON:-python3}"
font_scale="${ELARA_FIG_FONT_SCALE:-1.0}"
"$python_executable" -m ELARA.plot_paper_figures \
  --baseline-root ELARA/outputs/20260726-152326 \
  --bandit-root ELARA/outputs/20260726-152326 \
  --sensitivity-root ELARA/outputs/sensitivity \
  --output-dir ELARA/paper_fig4-2 \
  --temporal-bin-slots 5 \
  --temporal-smoothing-window 7 \
  --minimum-seeds 4 \
  --font-scale "$font_scale" \
  "$@"

paper_exp_dir="${ELARA_PAPER_EXP_DIR:-MyPaper/exp}"
mkdir -p "$paper_exp_dir"
cp ELARA/paper_fig4-2/fig4_bandit_online_adaptation.pdf "$paper_exp_dir/"
cp ELARA/paper_fig4-2/fig5_latency_energy_weight_sensitivity.pdf "$paper_exp_dir/"
