#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/test_outputs/full_cycle_seed_distribution}"
DEVICE="${DEVICE:-auto}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ppo_gnn_latest.pth}"
BANDIT_STATS_NAME="${BANDIT_STATS_NAME:-bandit_arm_stats.csv}"

args=(
  --seeds "${SEEDS}"
  --model-root "${MODEL_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --bandit-period-slots "${BANDIT_PERIOD_SLOTS}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --bandit-stats-name "${BANDIT_STATS_NAME}"
)

if [[ -n "${ARRIVAL_LAMBDA:-}" ]]; then
  args+=(--arrival-lambda "${ARRIVAL_LAMBDA}")
fi

if [[ -n "${MAX_SLOTS:-}" ]]; then
  args+=(--max-slots "${MAX_SLOTS}")
fi

if [[ -n "${ISL_CSV:-}" ]]; then
  args+=(--isl-csv "${ISL_CSV}")
fi

if [[ -n "${REQUEST_TEMPLATE_CSV:-}" ]]; then
  args+=(--request-template-csv "${REQUEST_TEMPLATE_CSV}")
fi

if [[ "${NO_LOAD_CHECKPOINT:-0}" == "1" ]]; then
  args+=(--no-load-checkpoint)
fi

if [[ "${NO_LOAD_BANDIT:-0}" == "1" ]]; then
  args+=(--no-load-bandit)
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running full-cycle seed distribution test"
echo "  seeds: ${SEEDS}"
echo "  model_root: ${MODEL_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  device: ${DEVICE}"

(
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution "${args[@]}" "$@"
)

echo "Done. Outputs are under ${OUTPUT_DIR}."
