#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/test_outputs/bandit_redeployment_replay_experiments}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/plots}"
ABLATIONS="${ABLATIONS:-ELARA ELARA-NB ELARA-NR ELARA-SH Fair-NFV SECO SP-Routing SC-NFV}"
FORMAT="${FORMAT:-png}"

echo "Plotting bandit redeployment replay experiments"
echo "  input_dir: ${INPUT_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  ablations: ${ABLATIONS}"
echo "  format: ${FORMAT}"

(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" -m Simulation.pics.plot_redeployment_replay_experiments \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --ablations "${ABLATIONS}" \
    --format "${FORMAT}" \
    "$@"
)

echo "Done. Redeployment replay plots are under ${OUTPUT_DIR}."
