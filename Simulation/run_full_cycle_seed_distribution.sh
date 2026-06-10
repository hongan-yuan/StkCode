#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45}"
GPUS="${GPUS:-0 1 2 3}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/test_outputs/full_cycle_seed_distribution}"
DEVICE="${DEVICE:-cuda}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ppo_gnn_latest.pth}"
BANDIT_STATS_NAME="${BANDIT_STATS_NAME:-bandit_arm_stats.csv}"

common_args=(
  --model-root "${MODEL_ROOT}"
  --bandit-period-slots "${BANDIT_PERIOD_SLOTS}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --bandit-stats-name "${BANDIT_STATS_NAME}"
)

if [[ -n "${ARRIVAL_LAMBDA:-}" ]]; then
  common_args+=(--arrival-lambda "${ARRIVAL_LAMBDA}")
fi

if [[ -n "${MAX_SLOTS:-}" ]]; then
  common_args+=(--max-slots "${MAX_SLOTS}")
fi

if [[ -n "${ISL_CSV:-}" ]]; then
  common_args+=(--isl-csv "${ISL_CSV}")
fi

if [[ -n "${REQUEST_TEMPLATE_CSV:-}" ]]; then
  common_args+=(--request-template-csv "${REQUEST_TEMPLATE_CSV}")
fi

if [[ "${NO_LOAD_CHECKPOINT:-0}" == "1" ]]; then
  common_args+=(--no-load-checkpoint)
fi

if [[ "${NO_LOAD_BANDIT:-0}" == "1" ]]; then
  common_args+=(--no-load-bandit)
fi

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one seed, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one GPU id, got: ${GPUS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running full-cycle seed distribution test"
echo "  seeds: ${SEEDS}"
echo "  gpus: ${GPUS}"
echo "  model_root: ${MODEL_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  device: ${DEVICE}"

gpu_count="${#GPU_ARRAY[@]}"
pids=()

for index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$index]}"
  gpu="${GPU_ARRAY[$((index % gpu_count))]}"
  log_file="${OUTPUT_DIR}/seed_${seed}.log"
  echo "Launching seed=${seed} on GPU=${gpu}; log=${log_file}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
      --seeds "${seed}" \
      --output-dir "${OUTPUT_DIR}" \
      --device "${DEVICE}" \
      --skip-aggregate \
      "${common_args[@]}" \
      "$@"
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
done

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one seed test failed. Check logs under ${OUTPUT_DIR}." >&2
  exit 1
fi

echo "All seed tests finished. Merging CSV files and plotting distributions."
(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
    --seeds "${SEEDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --plot-only
)

echo "Done. Outputs are under ${OUTPUT_DIR}."
