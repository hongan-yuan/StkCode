#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45}"
GPUS="${GPUS:-0 1 2 3}"
EPOCHS="${EPOCHS:-6060}"
MAX_SLOTS="${MAX_SLOTS:-606}"
ARRIVAL_LAMBDA="${ARRIVAL_LAMBDA:-0.35}"
PPO_UPDATE_SLOTS="${PPO_UPDATE_SLOTS:-5}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
ROUTE_HORIZON_SLOTS="${ROUTE_HORIZON_SLOTS:-3}"
PPO_ROLLOUT_BUFFER_SIZE="${PPO_ROLLOUT_BUFFER_SIZE:-256}"
REWARD_CHAIN_LENGTH_ALPHA="${REWARD_CHAIN_LENGTH_ALPHA:-0.5}"
LOG_EVERY="${LOG_EVERY:-500}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#SEED_ARRAY[@]}" -ne 4 ]]; then
  echo "Expected exactly 4 seeds, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "Expected exactly 4 GPU ids, got: ${GPUS}" >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"

pids=()
for index in "${!SEED_ARRAY[@]}"; do
  seed="${SEED_ARRAY[$index]}"
  gpu="${GPU_ARRAY[$index]}"
  model_dir="${MODEL_ROOT}/seed_${seed}"
  log_file="${MODEL_ROOT}/seed_${seed}.log"
  mkdir -p "${model_dir}"

  echo "Launching seed=${seed} on GPU=${gpu}; output=${model_dir}"
  (
    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m Simulation.train \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --max-slots "${MAX_SLOTS}" \
      --arrival-lambda "${ARRIVAL_LAMBDA}" \
      --ppo-update-slots "${PPO_UPDATE_SLOTS}" \
      --bandit-period-slots "${BANDIT_PERIOD_SLOTS}" \
      --route-horizon-slots "${ROUTE_HORIZON_SLOTS}" \
      --ppo-rollout-buffer-size "${PPO_ROLLOUT_BUFFER_SIZE}" \
      --reward-chain-length-alpha "${REWARD_CHAIN_LENGTH_ALPHA}" \
      --log-every "${LOG_EVERY}" \
      --model-dir "${model_dir}" \
      --device cuda \
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
  echo "At least one training job failed. Check logs under ${MODEL_ROOT}." >&2
  exit 1
fi

echo "All four training jobs completed. Outputs are under ${MODEL_ROOT}."
