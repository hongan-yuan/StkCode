#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
CHAIN_INPUT_ROOT="${CHAIN_INPUT_ROOT:-${SCRIPT_DIR}/test_outputs/chain_length_ablation_experiments}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/test_outputs/selected_seed_extremes}"
CHAIN_LENGTHS="${CHAIN_LENGTHS:-5 10 15}"
TARGET_ABLATION="${TARGET_ABLATION:-ELARA}"
ABLATION_GROUP_ABLATIONS="${ABLATION_GROUP_ABLATIONS:-ELARA ELARA-NB ELARA-NR ELARA-SH}"
COMPARISON_GROUP_ABLATIONS="${COMPARISON_GROUP_ABLATIONS:-ELARA Fair-NFV SP-Routing SC-NFV}"
ALL_ABLATIONS="${ALL_ABLATIONS:-${ABLATION_GROUP_ABLATIONS} ${COMPARISON_GROUP_ABLATIONS}}"
SLOT_WINDOW="${SLOT_WINDOW:-10}"
MAX_SLOT="${MAX_SLOT:-600}"
FORMAT="${FORMAT:-png}"
EXTRA_ARGS=("$@")

FILTERED_ABLATION_INPUT="${OUTPUT_ROOT}/ablation_experiments/filtered_input"
ABLATION_PLOT_DIR="${OUTPUT_ROOT}/ablation_experiments/plots"
CROSS_SLOT_PLOT_DIR="${OUTPUT_ROOT}/ablation_experiments/cross_slot_plots"
FILTERED_CHAIN_INPUT_ROOT="${OUTPUT_ROOT}/chain_length_ablation_experiments/filtered_input"
CHAIN_PLOT_ROOT="${OUTPUT_ROOT}/chain_length_ablation_experiments/plots"
MANIFEST="${OUTPUT_ROOT}/plot_manifest.txt"

read -r -a CHAIN_LENGTH_ARRAY <<< "${CHAIN_LENGTHS}"
if [[ "${#CHAIN_LENGTH_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one chain length, got: ${CHAIN_LENGTHS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

echo "Plotting selected-seed extreme experiment figures"
echo "  target_ablation_best: ${TARGET_ABLATION}"
echo "  non_target_ablations: worst seed"
echo "  input_dir: ${INPUT_DIR}"
echo "  chain_input_root: ${CHAIN_INPUT_ROOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  all_ablations: ${ALL_ABLATIONS}"
echo "  ablation_group: ${ABLATION_GROUP_ABLATIONS}"
echo "  comparison_group: ${COMPARISON_GROUP_ABLATIONS}"
echo "  chain_lengths: ${CHAIN_LENGTHS}"
echo "  slot_window: ${SLOT_WINDOW}"
echo "  max_slot: ${MAX_SLOT}"

append_manifest() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cat "${path}" >> "${MANIFEST}"
  fi
}

prepare_filtered_input() {
  local input_dir="$1"
  local output_dir="$2"
  local ablations="$3"
  echo "Selecting seeds from ${input_dir}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.pics.plot_selected_seed_extremes \
      --input-dir "${input_dir}" \
      --output-dir "${output_dir}" \
      --ablations "${ablations}" \
      --target-ablation "${TARGET_ABLATION}" \
      --force
  )
}

plot_metric_group() {
  local input_dir="$1"
  local output_dir="$2"
  local group_name="$3"
  local group_ablations="$4"
  echo "Plotting metric panels group=${group_name}: ${group_ablations}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.pics.plot_group_metric_panels \
      --input-dir "${input_dir}" \
      --output-dir "${output_dir}/${group_name}" \
      --ablations "${group_ablations}" \
      --format "${FORMAT}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  )
  append_manifest "${output_dir}/${group_name}/plot_manifest.txt"
}

plot_cross_slot_group() {
  local input_dir="$1"
  local output_dir="$2"
  local group_name="$3"
  local group_ablations="$4"
  echo "Plotting cross-slot curves group=${group_name}: ${group_ablations}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.pics.plot_cross_slot_windows \
      --input-dir "${input_dir}" \
      --output-dir "${output_dir}/${group_name}" \
      --ablations "${group_ablations}" \
      --slot-window "${SLOT_WINDOW}" \
      --max-slot "${MAX_SLOT}" \
      --format "${FORMAT}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  )
  append_manifest "${output_dir}/${group_name}/plot_manifest.txt"
}

prepare_filtered_input "${INPUT_DIR}" "${FILTERED_ABLATION_INPUT}" "${ALL_ABLATIONS}"

plot_metric_group "${FILTERED_ABLATION_INPUT}" "${ABLATION_PLOT_DIR}" "ablation_group" "${ABLATION_GROUP_ABLATIONS}"
plot_metric_group "${FILTERED_ABLATION_INPUT}" "${ABLATION_PLOT_DIR}" "comparison_group" "${COMPARISON_GROUP_ABLATIONS}"

plot_cross_slot_group "${FILTERED_ABLATION_INPUT}" "${CROSS_SLOT_PLOT_DIR}" "ablation_group" "${ABLATION_GROUP_ABLATIONS}"
plot_cross_slot_group "${FILTERED_ABLATION_INPUT}" "${CROSS_SLOT_PLOT_DIR}" "comparison_group" "${COMPARISON_GROUP_ABLATIONS}"

for chain_length in "${CHAIN_LENGTH_ARRAY[@]}"; do
  input_dir="${CHAIN_INPUT_ROOT}/chain_length_${chain_length}"
  filtered_dir="${FILTERED_CHAIN_INPUT_ROOT}/chain_length_${chain_length}"
  output_dir="${CHAIN_PLOT_ROOT}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "Skipping missing chain-length directory: ${input_dir}" >&2
    continue
  fi
  prepare_filtered_input "${input_dir}" "${filtered_dir}" "${ALL_ABLATIONS}"
  plot_metric_group "${filtered_dir}" "${output_dir}/ablation_group" "chain_length_${chain_length}" "${ABLATION_GROUP_ABLATIONS}"
  plot_metric_group "${filtered_dir}" "${output_dir}/comparison_group" "chain_length_${chain_length}" "${COMPARISON_GROUP_ABLATIONS}"
done

echo "Done. Selected-seed extreme plots are under ${OUTPUT_ROOT}."
echo "Selection manifests are under each filtered_input directory."
