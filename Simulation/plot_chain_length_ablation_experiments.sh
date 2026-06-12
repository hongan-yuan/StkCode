#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/test_outputs/chain_length_ablation_experiments}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INPUT_ROOT}/plots}"
CHAIN_LENGTHS="${CHAIN_LENGTHS:-5 10 15}"
ABLATION_GROUP_ABLATIONS="${ABLATION_GROUP_ABLATIONS:-ELARA ELARA-NB ELARA-NR ELARA-SH}"
COMPARISON_GROUP_ABLATIONS="${COMPARISON_GROUP_ABLATIONS:-ELARA Fair-NFV SP-Routing SC-NFV}"
FORMAT="${FORMAT:-png}"
EXTRA_ARGS=("$@")

read -r -a CHAIN_LENGTH_ARRAY <<< "${CHAIN_LENGTHS}"
if [[ "${#CHAIN_LENGTH_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one chain length, got: ${CHAIN_LENGTHS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Plotting chain-length grouped metric panels"
echo "  input_root: ${INPUT_ROOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  chain_lengths: ${CHAIN_LENGTHS}"
echo "  ablation_group: ${ABLATION_GROUP_ABLATIONS}"
echo "  comparison_group: ${COMPARISON_GROUP_ABLATIONS}"

manifest="${OUTPUT_ROOT}/plot_manifest.txt"
: > "${manifest}"

plot_group() {
  local group_name="$1"
  local group_ablations="$2"
  for chain_length in "${CHAIN_LENGTH_ARRAY[@]}"; do
    local input_dir="${INPUT_ROOT}/chain_length_${chain_length}"
    local output_dir="${OUTPUT_ROOT}/${group_name}/chain_length_${chain_length}"
    if [[ ! -d "${input_dir}" ]]; then
      echo "Skipping missing chain-length directory: ${input_dir}" >&2
      continue
    fi
    echo "Plotting group=${group_name} chain_length=${chain_length}: ${group_ablations}"
    (
      cd "${PROJECT_ROOT}"
      "${PYTHON_BIN}" -m Simulation.pics.plot_group_metric_panels \
        --input-dir "${input_dir}" \
        --output-dir "${output_dir}" \
        --ablations "${group_ablations}" \
        --format "${FORMAT}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    )
    if [[ -f "${output_dir}/plot_manifest.txt" ]]; then
      cat "${output_dir}/plot_manifest.txt" >> "${manifest}"
    fi
  done
}

plot_group "ablation_group" "${ABLATION_GROUP_ABLATIONS}"
plot_group "comparison_group" "${COMPARISON_GROUP_ABLATIONS}"

echo "Done. Chain-length plots are under ${OUTPUT_ROOT}."
