#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45}"
GPUS="${GPUS:-}"
GPU_COUNT="${GPU_COUNT:-}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-6060}"
MAX_SLOTS="${MAX_SLOTS:-606}"
ARRIVAL_LAMBDA="${ARRIVAL_LAMBDA:-0.35}"
PPO_UPDATE_SLOTS="${PPO_UPDATE_SLOTS:-5}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
ROUTE_HORIZON_SLOTS="${ROUTE_HORIZON_SLOTS:-3}"
MAX_CANDIDATE_REPLICAS="${MAX_CANDIDATE_REPLICAS:-4}"
BANDIT_PRESSURE_TOP_K_SERVICES="${BANDIT_PRESSURE_TOP_K_SERVICES:-10}"
REQUEST_TEMPLATE_CSV="${REQUEST_TEMPLATE_CSV:-}"
ROUTE_ESTIMATE_TIME_BUCKET_S="${ROUTE_ESTIMATE_TIME_BUCKET_S:-1.0}"
ROUTE_ESTIMATE_DATA_BUCKET_GB="${ROUTE_ESTIMATE_DATA_BUCKET_GB:-0.25}"
PPO_ROLLOUT_BUFFER_SIZE="${PPO_ROLLOUT_BUFFER_SIZE:-256}"
REWARD_CHAIN_LENGTH_ALPHA="${REWARD_CHAIN_LENGTH_ALPHA:-0.5}"
FAILURE_PENALTY="${FAILURE_PENALTY:-100.0}"
LOG_EVERY="${LOG_EVERY:-500}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
TRAIN_PLAN_ONLY="${TRAIN_PLAN_ONLY:-0}"

EXTRA_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --gpu-count) GPU_COUNT="$2"; shift 2 ;;
    --gpu-count=*) GPU_COUNT="${1#*=}"; shift ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --max-parallel=*) MAX_PARALLEL="${1#*=}"; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpus=*) GPUS="${1#*=}"; shift ;;
    --plan-only) TRAIN_PLAN_ONLY=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

read -r -a SEED_ARRAY <<< "${SEEDS}"
GPU_ARRAY=()
template_args=()
if [[ -n "${REQUEST_TEMPLATE_CSV}" ]]; then
  template_args+=(--request-template-csv "${REQUEST_TEMPLATE_CSV}")
fi

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one seed, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "cpu" ]]; then
  detected_gpus=""
  if [[ -n "${GPUS//[[:space:],]/}" ]]; then
    detected_gpus="${GPUS//,/ }"
    gpu_source="GPUS"
  elif [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
      detected_gpus="${CUDA_VISIBLE_DEVICES//,/ }"
    fi
    gpu_source="CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    detected_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')"
    gpu_source="nvidia-smi"
  fi

  if [[ -z "${detected_gpus//[[:space:]]/}" && "${gpu_source:-}" != "CUDA_VISIBLE_DEVICES" ]]; then
    gpu_count_from_torch="$(
      "${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true
    )"
    if [[ "${gpu_count_from_torch}" =~ ^[0-9]+$ && "${gpu_count_from_torch}" -gt 0 ]]; then
      for ((gpu_index = 0; gpu_index < gpu_count_from_torch; gpu_index++)); do
        detected_gpus+="${gpu_index} "
      done
      gpu_source="torch.cuda.device_count"
    elif [[ -z "${gpu_source:-}" ]]; then
      gpu_source="none"
    fi
  fi

  detected_gpu_array=()
  if [[ -n "${detected_gpus//[[:space:]]/}" ]]; then
    read -r -a detected_gpu_array <<< "${detected_gpus}"
    for gpu in "${detected_gpu_array[@]}"; do
      duplicate=0
      if [[ "${#GPU_ARRAY[@]}" -gt 0 ]]; then
        for existing_gpu in "${GPU_ARRAY[@]}"; do
          if [[ "${gpu}" == "${existing_gpu}" ]]; then
            duplicate=1
            break
          fi
        done
      fi
      if [[ "${duplicate}" -eq 0 ]]; then
        GPU_ARRAY+=("${gpu}")
      fi
    done
  fi

  if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
    echo "No visible GPU found for DEVICE=${DEVICE}. Set GPUS explicitly or use DEVICE=cpu." >&2
    exit 1
  fi

  if [[ -n "${GPU_COUNT}" ]]; then
    if [[ ! "${GPU_COUNT}" =~ ^[0-9]+$ || "${GPU_COUNT}" -lt 1 ]]; then
      echo "GPU_COUNT must be a positive integer when provided, got: ${GPU_COUNT}" >&2
      exit 1
    fi
    if [[ "${GPU_COUNT}" -ne "${#GPU_ARRAY[@]}" ]]; then
      echo "WARNING: GPU_COUNT=${GPU_COUNT} is ignored; all ${#GPU_ARRAY[@]} visible/configured GPUs will be used."
    fi
  fi
  GPU_COUNT="${#GPU_ARRAY[@]}"
fi

seed_count="${#SEED_ARRAY[@]}"
if [[ "${DEVICE}" == "cpu" ]]; then
  max_parallel="${MAX_PARALLEL:-4}"
  echo "Launching ${seed_count} training job(s) on CPU with max_parallel=${max_parallel}"
else
  gpu_count="${#GPU_ARRAY[@]}"
  max_parallel="${MAX_PARALLEL:-4}"
  echo "Launching ${seed_count} training job(s) over ${gpu_count} GPU(s): ${GPU_ARRAY[*]}"
  echo "  gpu_source: ${gpu_source}"
  echo "  max_parallel_tasks: ${max_parallel}"
  echo "  assignment_policy: round-robin (task counts differ by at most one)"
  for gpu_index in "${!GPU_ARRAY[@]}"; do
    planned_jobs=$((seed_count / gpu_count))
    if [[ "${gpu_index}" -lt $((seed_count % gpu_count)) ]]; then
      planned_jobs=$((planned_jobs + 1))
    fi
    echo "  GPU ${GPU_ARRAY[$gpu_index]}: ${planned_jobs} planned training job(s)"
  done
fi

if [[ "${max_parallel}" -lt 1 ]]; then
  echo "MAX_PARALLEL must be at least 1, got: ${max_parallel}" >&2
  exit 1
fi

if [[ "${TRAIN_PLAN_ONLY}" != "0" && "${TRAIN_PLAN_ONLY}" != "1" ]]; then
  echo "TRAIN_PLAN_ONLY must be 0 or 1, got: ${TRAIN_PLAN_ONLY}" >&2
  exit 1
fi
if [[ "${TRAIN_PLAN_ONLY}" == "1" ]]; then
  echo "Training assignment plan (no jobs launched):"
  for index in "${!SEED_ARRAY[@]}"; do
    seed="${SEED_ARRAY[$index]}"
    if [[ "${DEVICE}" == "cpu" ]]; then
      echo "  seed=${seed} -> CPU"
    else
      gpu="${GPU_ARRAY[$((index % gpu_count))]}"
      echo "  seed=${seed} -> GPU=${gpu}"
    fi
  done
  exit 0
fi

mkdir -p "${MODEL_ROOT}"

failed=0
source "${SCRIPT_DIR}/parallel_runner.sh"
parallel_runner_init "${seed_count}" "${max_parallel}"

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
        --route-estimate-time-bucket-s "${ROUTE_ESTIMATE_TIME_BUCKET_S}" \
        --route-estimate-data-bucket-gb "${ROUTE_ESTIMATE_DATA_BUCKET_GB}" \
        --ppo-rollout-buffer-size "${PPO_ROLLOUT_BUFFER_SIZE}" \
        --reward-chain-length-alpha "${REWARD_CHAIN_LENGTH_ALPHA}" \
        --failure-penalty "${FAILURE_PENALTY}" \
        --log-every "${LOG_EVERY}" \
        --model-dir "${model_dir}" \
        --device "${DEVICE}" \
        ${template_args[@]+"${template_args[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ) > "${log_file}" 2>&1 &
    label="seed=${seed} device=cpu"
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
        --route-estimate-time-bucket-s "${ROUTE_ESTIMATE_TIME_BUCKET_S}" \
        --route-estimate-data-bucket-gb "${ROUTE_ESTIMATE_DATA_BUCKET_GB}" \
        --ppo-rollout-buffer-size "${PPO_ROLLOUT_BUFFER_SIZE}" \
        --reward-chain-length-alpha "${REWARD_CHAIN_LENGTH_ALPHA}" \
        --failure-penalty "${FAILURE_PENALTY}" \
        --log-every "${LOG_EVERY}" \
        --model-dir "${model_dir}" \
        --device "${DEVICE}" \
        ${template_args[@]+"${template_args[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ) > "${log_file}" 2>&1 &
    label="seed=${seed} gpu=${gpu}"
  fi
  parallel_register_task "$!" "${label}"
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
