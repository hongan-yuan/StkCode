#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/plots}"
ABLATIONS="${ABLATIONS:-full no_bandit shortest_hop_routing nearest_replica}"
WINDOW="${WINDOW:-1}"
FORMAT="${FORMAT:-auto}"

echo "Plotting ablation experiments"
echo "  input_dir: ${INPUT_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  ablations: ${ABLATIONS}"
echo "  window: ${WINDOW}"
echo "  format: ${FORMAT}"

(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" -m Simulation.pics.plot_ablation_experiments \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --ablations "${ABLATIONS}" \
    --window "${WINDOW}" \
    --format "${FORMAT}" \
    "$@"
)

echo "Done. Plots are under ${OUTPUT_DIR}."
