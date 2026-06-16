#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-41 42 43 44}"
RUN_ABLATIONS="${RUN_ABLATIONS:-${ABLATIONS:-ELARA ELARA-NB}}"
MERGE_ABLATIONS="${MERGE_ABLATIONS:-${RUN_ABLATIONS}}"
REDEPLOY_ABLATIONS="${REDEPLOY_ABLATIONS:-ELARA}"
TRACE_SOURCE_ABLATION="${TRACE_SOURCE_ABLATION:-ELARA}"
GPUS="${GPUS:-0 1 2 3}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/test_outputs/bandit_redeployment_replay_experiments}"
DEFAULT_ISL_CSV="${PROJECT_ROOT}/WalkerDeltaConstellationSimu/Walker_Delta_ISL_Simu.csv"
ISL_CSV="${ISL_CSV:-${DEFAULT_ISL_CSV}}"
DEVICE="${DEVICE:-cpu}"
CPU_WORKERS="${CPU_WORKERS:-4}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-9}"
WINDOW_SLOTS="${WINDOW_SLOTS:-600}"
BASE_WINDOW_SLOTS="${BASE_WINDOW_SLOTS:-10}"
START_SLOT="${START_SLOT:-0}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ppo_gnn_latest.pth}"
BANDIT_STATS_NAME="${BANDIT_STATS_NAME:-bandit_arm_stats.csv}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
EXTRA_ARGS=("$@")

common_args=(
  --model-root "${MODEL_ROOT}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --bandit-stats-name "${BANDIT_STATS_NAME}"
  --isl-csv "${ISL_CSV}"
  --window-slots "${WINDOW_SLOTS}"
  --base-window-slots "${BASE_WINDOW_SLOTS}"
  --start-slot "${START_SLOT}"
  --redeploy-ablations "${REDEPLOY_ABLATIONS}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ -n "${ARRIVAL_LAMBDA:-}" ]]; then
  common_args+=(--arrival-lambda "${ARRIVAL_LAMBDA}")
fi

if [[ -n "${ARRIVAL_MODE:-}" ]]; then
  common_args+=(--arrival-mode "${ARRIVAL_MODE}")
fi

if [[ -n "${TOTAL_ARRIVAL_LAMBDA:-}" ]]; then
  common_args+=(--total-arrival-lambda "${TOTAL_ARRIVAL_LAMBDA}")
fi

if [[ -n "${CHAIN_LENGTH_FILTER:-}" ]]; then
  common_args+=(--chain-length-filter "${CHAIN_LENGTH_FILTER}")
fi

if [[ -n "${MAX_SLOTS:-}" ]]; then
  common_args+=(--max-slots "${MAX_SLOTS}")
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
read -r -a RUN_ABLATION_ARRAY <<< "${RUN_ABLATIONS}"
read -r -a MERGE_ABLATION_ARRAY <<< "${MERGE_ABLATIONS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one seed, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${#RUN_ABLATION_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one ablation variant, got: ${RUN_ABLATIONS}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "cpu" && "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one GPU id, got: ${GPUS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Running bandit redeployment replay experiments"
echo "  run_ablations: ${RUN_ABLATIONS}"
echo "  merge_ablations: ${MERGE_ABLATIONS}"
echo "  redeploy_ablations: ${REDEPLOY_ABLATIONS}"
echo "  trace_source_ablation: ${TRACE_SOURCE_ABLATION}"
echo "  seeds: ${SEEDS}"
echo "  window_slots: ${WINDOW_SLOTS}"
echo "  base_window_slots: ${BASE_WINDOW_SLOTS}"
echo "  start_slot: ${START_SLOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  device: ${DEVICE}"

if [[ "${DEVICE}" == "cpu" ]]; then
  max_parallel="${MAX_PARALLEL:-${CPU_WORKERS}}"
else
  gpu_count="${#GPU_ARRAY[@]}"
  max_parallel="${MAX_PARALLEL:-$((gpu_count * GPU_WORKERS_PER_GPU))}"
fi
echo "  max_parallel_tasks: ${max_parallel}"

if [[ "${max_parallel}" -lt 1 ]]; then
  echo "MAX_PARALLEL/CPU_WORKERS/GPU_WORKERS_PER_GPU must be at least 1, got: ${max_parallel}" >&2
  exit 1
fi

task_index=0
failed=0
pids=()
labels=()

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

existing_seed_list() {
  local variant_dir="$1"
  local seed_dirs=()
  local seed_dir seed
  for seed_dir in "${variant_dir}"/seed_*; do
    [[ -d "${seed_dir}" ]] || continue
    [[ -f "${seed_dir}/redeployment_summary_metrics.csv" ]] || continue
    [[ -f "${seed_dir}/window_slot_metrics.csv" ]] || continue
    [[ -f "${seed_dir}/window_request_metrics.csv" ]] || continue
    [[ -f "${seed_dir}/redeployment_window_metrics.csv" ]] || continue
    [[ -f "${seed_dir}/request_trace.csv" ]] || continue
    seed="${seed_dir##*/seed_}"
    seed_dirs+=("${seed}")
  done
  if [[ "${#seed_dirs[@]}" -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${seed_dirs[@]}" | sort -n | tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

require_seed_outputs() {
  local variant_dir="$1"
  local label="$2"
  local seed="$3"
  local seed_dir="${variant_dir}/seed_${seed}"
  local missing=0
  local file
  for file in request_trace.csv window_slot_metrics.csv window_request_metrics.csv redeployment_window_metrics.csv redeployment_summary_metrics.csv summary.json; do
    if [[ ! -f "${seed_dir}/${file}" ]]; then
      echo "Missing ${file} for ${label} seed=${seed}: ${seed_dir}/${file}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

contains_ablation() {
  local needle="$1"
  local item
  shift
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

launch_task() {
  local ablation="$1"
  local seed="$2"
  local trace_csv="${3:-}"
  local variant_dir="${OUTPUT_ROOT}/${ablation}"
  local log_file="${variant_dir}/seed_${seed}.log"
  local label="ablation=${ablation} seed=${seed}"
  local trace_args=()
  mkdir -p "${variant_dir}"
  if [[ -n "${trace_csv}" ]]; then
    trace_args+=(--request-trace-csv "${trace_csv}")
    label="${label} trace=${trace_csv}"
  fi
  wait_for_slot
  if [[ "${DEVICE}" == "cpu" ]]; then
    echo "Launching ${label} on CPU; log=${log_file}"
    (
      cd "${PROJECT_ROOT}"
      CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m Simulation.tests.redeployment_replay_experiment \
        --ablation "${ablation}" \
        --seeds "${seed}" \
        --output-dir "${variant_dir}" \
        --device "${DEVICE}" \
        --skip-aggregate \
        ${trace_args[@]+"${trace_args[@]}"} \
        "${common_args[@]}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ) > "${log_file}" 2>&1 &
  else
    gpu="${GPU_ARRAY[$((task_index % gpu_count))]}"
    echo "Launching ${label} on GPU=${gpu}; log=${log_file}"
    (
      cd "${PROJECT_ROOT}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m Simulation.tests.redeployment_replay_experiment \
        --ablation "${ablation}" \
        --seeds "${seed}" \
        --output-dir "${variant_dir}" \
        --device "${DEVICE}" \
        --skip-aggregate \
        ${trace_args[@]+"${trace_args[@]}"} \
        "${common_args[@]}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    ) > "${log_file}" 2>&1 &
  fi
  pids+=("$!")
  labels+=("${label}")
  task_index=$((task_index + 1))
}

if ! contains_ablation "${TRACE_SOURCE_ABLATION}" "${RUN_ABLATION_ARRAY[@]}"; then
  echo "RUN_ABLATIONS must include TRACE_SOURCE_ABLATION=${TRACE_SOURCE_ABLATION}." >&2
  exit 1
fi

echo "Stage 1: running ${TRACE_SOURCE_ABLATION} to generate shared request traces."
for seed in "${SEED_ARRAY[@]}"; do
  launch_task "${TRACE_SOURCE_ABLATION}" "${seed}"
done

wait_active_tasks
if [[ "${failed}" -ne 0 ]]; then
  echo "Trace source ablation failed. Check logs under ${OUTPUT_ROOT}/${TRACE_SOURCE_ABLATION}." >&2
  exit 1
fi

echo "Validating shared request traces from ${TRACE_SOURCE_ABLATION}."
for seed in "${SEED_ARRAY[@]}"; do
  trace_csv="${OUTPUT_ROOT}/${TRACE_SOURCE_ABLATION}/seed_${seed}/request_trace.csv"
  if [[ ! -f "${trace_csv}" ]]; then
    echo "Missing shared request trace for seed=${seed}: ${trace_csv}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "Stage 2: running comparison ablations with the shared ${TRACE_SOURCE_ABLATION} traces."
for ablation in "${RUN_ABLATION_ARRAY[@]}"; do
  [[ "${ablation}" == "${TRACE_SOURCE_ABLATION}" ]] && continue
  for seed in "${SEED_ARRAY[@]}"; do
    trace_csv="${OUTPUT_ROOT}/${TRACE_SOURCE_ABLATION}/seed_${seed}/request_trace.csv"
    launch_task "${ablation}" "${seed}" "${trace_csv}"
  done
done

wait_active_tasks

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one redeployment replay task failed. Check logs under ${OUTPUT_ROOT}." >&2
  exit 1
fi

echo "Validating per-seed metric outputs."
for ablation in "${RUN_ABLATION_ARRAY[@]}"; do
  variant_dir="${OUTPUT_ROOT}/${ablation}"
  for seed in "${SEED_ARRAY[@]}"; do
    if ! require_seed_outputs "${variant_dir}" "ablation=${ablation}" "${seed}"; then
      failed=1
    fi
  done
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one seed is missing required replay metric CSV files." >&2
  exit 1
fi

echo "Merging per-variant outputs."
for ablation in "${MERGE_ABLATION_ARRAY[@]}"; do
  variant_dir="${OUTPUT_ROOT}/${ablation}"
  if [[ ! -d "${variant_dir}" ]]; then
    echo "Skipping missing variant directory during merge: ${variant_dir}" >&2
    continue
  fi
  variant_seeds="$(existing_seed_list "${variant_dir}" || true)"
  if [[ -z "${variant_seeds}" ]]; then
    echo "Skipping ${ablation}; no seed replay metrics found." >&2
    continue
  fi
  echo "Merging ablation=${ablation} seeds=${variant_seeds}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.tests.redeployment_replay_experiment \
      --ablation "${ablation}" \
      --seeds "${variant_seeds}" \
      --output-dir "${variant_dir}" \
      --plot-only \
      "${common_args[@]}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  )
done

echo "Building cross-variant redeployment replay summary table."
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${MERGE_ABLATIONS}" <<'PY'
import csv
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
ablations = sys.argv[2].split()

def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

summary_rows = []
slot_rows = []
request_rows = []
window_rows = []
for ablation in ablations:
    variant_dir = output_root / ablation
    for row in read_rows(variant_dir / "redeployment_summary_by_seed.csv"):
        row["ablation"] = row.get("ablation") or ablation
        summary_rows.append(row)
    for row in read_rows(variant_dir / "window_slot_metrics_by_seed.csv"):
        row["ablation"] = row.get("ablation") or ablation
        slot_rows.append(row)
    for row in read_rows(variant_dir / "window_request_metrics_by_seed.csv"):
        row["ablation"] = row.get("ablation") or ablation
        request_rows.append(row)
    for row in read_rows(variant_dir / "redeployment_window_metrics_by_seed.csv"):
        row["ablation"] = row.get("ablation") or ablation
        window_rows.append(row)

write_rows(output_root / "all_redeployment_summary_metrics.csv", summary_rows)
write_rows(output_root / "all_redeployment_window_metrics.csv", window_rows)
write_rows(output_root / "all_redeployment_slot_metrics.csv", slot_rows)
write_rows(output_root / "all_redeployment_request_metrics.csv", request_rows)
PY

echo "Done. Outputs are under ${OUTPUT_ROOT}."
