#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-${SCRIPT_DIR}/test_outputs/chain_length_ablation_experiments}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INPUT_ROOT}/plots}"
CHAIN_LENGTHS="${CHAIN_LENGTHS:-5 10 15}"
ABLATIONS="${ABLATIONS:-full no_bandit shortest_hop_routing nearest_replica service_pressure sc_nfv}"
WINDOW="${WINDOW:-1}"
FORMAT="${FORMAT:-auto}"

read -r -a CHAIN_LENGTH_ARRAY <<< "${CHAIN_LENGTHS}"

if [[ "${#CHAIN_LENGTH_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one chain length, got: ${CHAIN_LENGTHS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Plotting chain-length ablation experiments"
echo "  input_root: ${INPUT_ROOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  chain_lengths: ${CHAIN_LENGTHS}"
echo "  ablations: ${ABLATIONS}"

manifest="${OUTPUT_ROOT}/plot_manifest.txt"
: > "${manifest}"

for chain_length in "${CHAIN_LENGTH_ARRAY[@]}"; do
  input_dir="${INPUT_ROOT}/chain_length_${chain_length}"
  output_dir="${OUTPUT_ROOT}/chain_length_${chain_length}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "Skipping missing chain-length directory: ${input_dir}" >&2
    continue
  fi
  echo "Plotting chain_length=${chain_length}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.pics.plot_ablation_experiments \
      --input-dir "${input_dir}" \
      --output-dir "${output_dir}" \
      --ablations "${ABLATIONS}" \
      --window "${WINDOW}" \
      --format "${FORMAT}"
  )
  if [[ -f "${output_dir}/plot_manifest.txt" ]]; then
    cat "${output_dir}/plot_manifest.txt" >> "${manifest}"
  fi
done

echo "Done. Plot manifest: ${manifest}"
