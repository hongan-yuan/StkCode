#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/plots}"
ABLATION_GROUP_ABLATIONS="${ABLATION_GROUP_ABLATIONS:-ELARA ELARA-NB ELARA-NR ELARA-SH}"
COMPARISON_GROUP_ABLATIONS="${COMPARISON_GROUP_ABLATIONS:-ELARA Fair-NFV SP-Routing SC-NFV}"
FORMAT="${FORMAT:-png}"
EXTRA_ARGS=("$@")

echo "Plotting grouped ablation/comparison metric panels"
echo "  input_dir: ${INPUT_DIR}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  ablation_group: ${ABLATION_GROUP_ABLATIONS}"
echo "  comparison_group: ${COMPARISON_GROUP_ABLATIONS}"
echo "  format: ${FORMAT}"

manifest="${OUTPUT_DIR}/plot_manifest.txt"
mkdir -p "${OUTPUT_DIR}"
: > "${manifest}"

plot_group() {
  local group_name="$1"
  local group_ablations="$2"
  local group_output_dir="${OUTPUT_DIR}/${group_name}"
  echo "Plotting group=${group_name}: ${group_ablations}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.pics.plot_group_metric_panels \
      --input-dir "${INPUT_DIR}" \
      --output-dir "${group_output_dir}" \
      --ablations "${group_ablations}" \
      --format "${FORMAT}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  )
  if [[ -f "${group_output_dir}/plot_manifest.txt" ]]; then
    cat "${group_output_dir}/plot_manifest.txt" >> "${manifest}"
  fi
}

plot_group "ablation_group" "${ABLATION_GROUP_ABLATIONS}"
plot_group "comparison_group" "${COMPARISON_GROUP_ABLATIONS}"

echo "Done. New plots are under ${OUTPUT_DIR}."
