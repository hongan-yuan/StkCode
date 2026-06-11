#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/selected_seed_plots}"
TARGET_ABLATION="${TARGET_ABLATION:-full}"
ABLATIONS="${ABLATIONS:-full no_bandit shortest_hop_routing nearest_replica service_pressure sc_nfv fairness_nfv_greedy}"
FORMAT="${FORMAT:-auto}"

echo "Plotting selected seed extremes"
echo "  input_dir: ${INPUT_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  target_ablation: ${TARGET_ABLATION}"
echo "  ablations: ${ABLATIONS}"
echo "  format: ${FORMAT}"

(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" -m Simulation.pics.plot_selected_seed_extremes \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-ablation "${TARGET_ABLATION}" \
    --ablations "${ABLATIONS}" \
    --format "${FORMAT}" \
    "$@"
)

echo "Done. Selected-seed plots are under ${OUTPUT_DIR}."
