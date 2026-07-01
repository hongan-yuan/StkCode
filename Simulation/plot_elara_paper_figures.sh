#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ROOT="${TRAIN_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/pics/elara_paper}"
SEEDS="${SEEDS:-}"
WINDOW="${WINDOW:-100}"
FORMAT="${FORMAT:-png}"

echo "Plotting ELARA paper figures"
echo "  train_root: ${TRAIN_ROOT}"
echo "  experiment_dir: ${EXPERIMENT_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  seeds: ${SEEDS:-all discovered}"
echo "  window: ${WINDOW}"
echo "  format: ${FORMAT}"

ARGS=(
  --train-root "${TRAIN_ROOT}"
  --experiment-dir "${EXPERIMENT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --window "${WINDOW}"
  --format "${FORMAT}"
)

if [[ -n "${SEEDS}" ]]; then
  ARGS+=(--seeds "${SEEDS}")
fi

(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" -m Simulation.pics.plot_elara_paper_figures "${ARGS[@]}" "$@"
)

echo "Done. Figures are under ${OUTPUT_DIR}."
