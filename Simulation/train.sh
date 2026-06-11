#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45}"
GPUS="${GPUS:-0 1 2 3}"
DEVICE="${DEVICE:-cuda}"
CPU_WORKERS="${CPU_WORKERS:-4}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-9}"
EPOCHS="${EPOCHS:-6060}"
MAX_SLOTS="${MAX_SLOTS:-606}"
ARRIVAL_LAMBDA="${ARRIVAL_LAMBDA:-0.35}"
PPO_UPDATE_SLOTS="${PPO_UPDATE_SLOTS:-5}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
ROUTE_HORIZON_SLOTS="${ROUTE_HORIZON_SLOTS:-3}"
MAX_CANDIDATE_REPLICAS="${MAX_CANDIDATE_REPLICAS:-4}"
BANDIT_PRESSURE_TOP_K_SERVICES="${BANDIT_PRESSURE_TOP_K_SERVICES:-8}"
BANDIT_TARGET_TOP_N_PLANES="${BANDIT_TARGET_TOP_N_PLANES:-3}"
ROUTE_ESTIMATE_TIME_BUCKET_S="${ROUTE_ESTIMATE_TIME_BUCKET_S:-1.0}"
ROUTE_ESTIMATE_DATA_BUCKET_GB="${ROUTE_ESTIMATE_DATA_BUCKET_GB:-0.25}"
PPO_ROLLOUT_BUFFER_SIZE="${PPO_ROLLOUT_BUFFER_SIZE:-256}"
REWARD_CHAIN_LENGTH_ALPHA="${REWARD_CHAIN_LENGTH_ALPHA:-0.5}"
FAILURE_PENALTY="${FAILURE_PENALTY:-100.0}"
LOG_EVERY="${LOG_EVERY:-500}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one seed, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "cpu" && "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one GPU id, got: ${GPUS}" >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"

seed_count="${#SEED_ARRAY[@]}"
if [[ "${DEVICE}" == "cpu" ]]; then
  max_parallel="${MAX_PARALLEL:-${CPU_WORKERS}}"
  echo "Launching ${seed_count} training job(s) on CPU with max_parallel=${max_parallel}"
else
  gpu_count="${#GPU_ARRAY[@]}"
  max_parallel="${MAX_PARALLEL:-$((gpu_count * GPU_WORKERS_PER_GPU))}"
  echo "Launching ${seed_count} training job(s) over ${gpu_count} GPU(s): ${GPUS}"
  echo "  gpu_workers_per_gpu: ${GPU_WORKERS_PER_GPU}"
  echo "  max_parallel_tasks: ${max_parallel}"
fi

if [[ "${max_parallel}" -lt 1 ]]; then
  echo "MAX_PARALLEL/CPU_WORKERS/GPU_WORKERS_PER_GPU must be at least 1, got: ${max_parallel}" >&2
  exit 1
fi

pids=()
labels=()
failed=0

wait_active_tasks() {
  local i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "Task failed: ${labels[$i]}" >&2
      failed=1
    fi
  done
  pids=()
  labels=()
}

wait_for_slot() {
  if [[ "${#pids[@]}" -ge "${max_parallel}" ]]; then
    wait_active_tasks
  fi
}

for index in "${!SEED_ARRAY[@]}"; do
  wait_for_slot
  seed="${SEED_ARRAY[$index]}"
  model_dir="${MODEL_ROOT}/seed_${seed}"
  log_file="${MODEL_ROOT}/seed_${seed}.log"
  mkdir -p "${model_dir}"

  if [[ "${DEVICE}" == "cpu" ]]; then
    echo "Launching seed=${seed} on CPU; output=${model_dir}; log=${log_file}"
    (
      cd "${PROJECT_ROOT}"
      CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m Simulation.train \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --max-slots "${MAX_SLOTS}" \
        --arrival-lambda "${ARRIVAL_LAMBDA}" \
        --ppo-update-slots "${PPO_UPDATE_SLOTS}" \
        --bandit-period-slots "${BANDIT_PERIOD_SLOTS}" \
        --route-horizon-slots "${ROUTE_HORIZON_SLOTS}" \
        --max-candidate-replicas "${MAX_CANDIDATE_REPLICAS}" \
        --bandit-pressure-top-k-services "${BANDIT_PRESSURE_TOP_K_SERVICES}" \
        --bandit-target-top-n-planes "${BANDIT_TARGET_TOP_N_PLANES}" \
        --route-estimate-time-bucket-s "${ROUTE_ESTIMATE_TIME_BUCKET_S}" \
        --route-estimate-data-bucket-gb "${ROUTE_ESTIMATE_DATA_BUCKET_GB}" \
        --ppo-rollout-buffer-size "${PPO_ROLLOUT_BUFFER_SIZE}" \
        --reward-chain-length-alpha "${REWARD_CHAIN_LENGTH_ALPHA}" \
        --failure-penalty "${FAILURE_PENALTY}" \
        --log-every "${LOG_EVERY}" \
        --model-dir "${model_dir}" \
        --device "${DEVICE}" \
        "$@"
    ) > "${log_file}" 2>&1 &
    labels+=("seed=${seed} device=cpu")
  else
    gpu="${GPU_ARRAY[$((index % gpu_count))]}"
    echo "Launching seed=${seed} on GPU=${gpu}; output=${model_dir}; log=${log_file}"
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
        --max-candidate-replicas "${MAX_CANDIDATE_REPLICAS}" \
        --bandit-pressure-top-k-services "${BANDIT_PRESSURE_TOP_K_SERVICES}" \
        --bandit-target-top-n-planes "${BANDIT_TARGET_TOP_N_PLANES}" \
        --route-estimate-time-bucket-s "${ROUTE_ESTIMATE_TIME_BUCKET_S}" \
        --route-estimate-data-bucket-gb "${ROUTE_ESTIMATE_DATA_BUCKET_GB}" \
        --ppo-rollout-buffer-size "${PPO_ROLLOUT_BUFFER_SIZE}" \
        --reward-chain-length-alpha "${REWARD_CHAIN_LENGTH_ALPHA}" \
        --failure-penalty "${FAILURE_PENALTY}" \
        --log-every "${LOG_EVERY}" \
        --model-dir "${model_dir}" \
        --device "${DEVICE}" \
        "$@"
    ) > "${log_file}" 2>&1 &
    labels+=("seed=${seed} gpu=${gpu}")
  fi
  pids+=("$!")
done

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

wait_active_tasks

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one training job failed. Check logs under ${MODEL_ROOT}." >&2
  exit 1
fi

echo "All ${seed_count} training job(s) completed. Outputs are under ${MODEL_ROOT}."
