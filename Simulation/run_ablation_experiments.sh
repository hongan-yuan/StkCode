#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-41 42 43 44}"
RUN_ABLATIONS="${RUN_ABLATIONS:-${ABLATIONS:-service_pressure sc_nfv fairness_nfv_greedy}}"
MERGE_ABLATIONS="${MERGE_ABLATIONS:-full no_bandit shortest_hop_routing nearest_replica service_pressure sc_nfv fairness_nfv_greedy}"
GPUS="${GPUS:-0 1 2 3}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
DEFAULT_ISL_CSV="${PROJECT_ROOT}/WalkerDeltaConstellationSimu/Walker_Delta_ISL_Simu.csv"
ISL_CSV="${ISL_CSV:-${DEFAULT_ISL_CSV}}"
DEVICE="${DEVICE:-cuda}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ppo_gnn_latest.pth}"
BANDIT_STATS_NAME="${BANDIT_STATS_NAME:-bandit_arm_stats.csv}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
CPU_WORKERS="${CPU_WORKERS:-4}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-9}"

common_args=(
  --model-root "${MODEL_ROOT}"
  --bandit-period-slots "${BANDIT_PERIOD_SLOTS}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --bandit-stats-name "${BANDIT_STATS_NAME}"
  --progress-every "${PROGRESS_EVERY}"
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
read -r -a RUN_ABLATION_ARRAY <<< "${RUN_ABLATIONS}"
read -r -a MERGE_ABLATION_ARRAY <<< "${MERGE_ABLATIONS}"

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one seed, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "cpu" && "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one GPU id, got: ${GPUS}" >&2
  exit 1
fi

if [[ "${#RUN_ABLATION_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one ablation variant to run, got: ${RUN_ABLATIONS}" >&2
  exit 1
fi

if [[ "${#MERGE_ABLATION_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one ablation variant to merge, got: ${MERGE_ABLATIONS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Running ablation experiments"
echo "  run_ablations: ${RUN_ABLATIONS}"
echo "  merge_ablations: ${MERGE_ABLATIONS}"
echo "  seeds: ${SEEDS}"
if [[ "${DEVICE}" == "cpu" ]]; then
  echo "  cpu_workers: ${CPU_WORKERS}"
else
  echo "  gpus: ${GPUS}"
  echo "  gpu_workers_per_gpu: ${GPU_WORKERS_PER_GPU}"
fi
echo "  model_root: ${MODEL_ROOT}"
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
    [[ -f "${seed_dir}/slot_metrics.csv" ]] || continue
    seed="${seed_dir##*/seed_}"
    seed_dirs+=("${seed}")
  done
  if [[ "${#seed_dirs[@]}" -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${seed_dirs[@]}" | sort -n | tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

for ablation in "${RUN_ABLATION_ARRAY[@]}"; do
  variant_dir="${OUTPUT_ROOT}/${ablation}"
  mkdir -p "${variant_dir}"
  for seed in "${SEED_ARRAY[@]}"; do
    wait_for_slot
    log_file="${variant_dir}/seed_${seed}.log"
    label="ablation=${ablation} seed=${seed}"
    if [[ "${DEVICE}" == "cpu" ]]; then
      echo "Launching ${label} on CPU; log=${log_file}"
      (
        cd "${PROJECT_ROOT}"
        CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
          --ablation "${ablation}" \
          --seeds "${seed}" \
          --output-dir "${variant_dir}" \
          --device "${DEVICE}" \
          --skip-aggregate \
          "${common_args[@]}" \
          "$@"
      ) > "${log_file}" 2>&1 &
    else
      gpu="${GPU_ARRAY[$((task_index % gpu_count))]}"
      echo "Launching ${label} on GPU=${gpu}; log=${log_file}"
      (
        cd "${PROJECT_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
          --ablation "${ablation}" \
          --seeds "${seed}" \
          --output-dir "${variant_dir}" \
          --device "${DEVICE}" \
          --skip-aggregate \
          "${common_args[@]}" \
          "$@"
      ) > "${log_file}" 2>&1 &
    fi
    pids+=("$!")
    labels+=("${label}")
    task_index=$((task_index + 1))
  done
done

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

wait_active_tasks

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one ablation task failed. Check logs under ${OUTPUT_ROOT}." >&2
  exit 1
fi

echo "All ablation tasks finished. Merging per-variant outputs."
for ablation in "${MERGE_ABLATION_ARRAY[@]}"; do
  variant_dir="${OUTPUT_ROOT}/${ablation}"
  if [[ ! -d "${variant_dir}" ]]; then
    echo "Skipping missing variant directory during merge: ${variant_dir}" >&2
    continue
  fi
  variant_seeds="$(existing_seed_list "${variant_dir}" || true)"
  if [[ -z "${variant_seeds}" ]]; then
    echo "Skipping ${ablation}; no seed_<seed>/slot_metrics.csv files found." >&2
    continue
  fi
  echo "Merging ablation=${ablation} seeds=${variant_seeds}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
      --ablation "${ablation}" \
      --seeds "${variant_seeds}" \
      --output-dir "${variant_dir}" \
      --plot-only
  )
done

echo "Building cross-variant merged metric tables."
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

slot_rows = []
request_rows = []
cycle_rows = []
for ablation in ablations:
    variant_dir = output_root / ablation
    for row in read_rows(variant_dir / "slot_metrics_by_seed.csv"):
        row.setdefault("ablation", ablation)
        slot_rows.append(row)
    for row in read_rows(variant_dir / "request_metrics_by_seed.csv"):
        row.setdefault("ablation", ablation)
        request_rows.append(row)
    for row in read_rows(variant_dir / "cycle_metrics_by_seed.csv"):
        row.setdefault("ablation", ablation)
        cycle_rows.append(row)

write_rows(output_root / "all_ablation_slot_metrics.csv", slot_rows)
write_rows(output_root / "all_ablation_request_metrics.csv", request_rows)
write_rows(output_root / "all_ablation_cycle_metrics.csv", cycle_rows)

metric_columns = [
    "task_completion_rate",
    "success_rate",
    "average_end_to_end_delay_s",
    "average_energy_j",
    "p95_end_to_end_delay_s",
    "average_communication_delay_s",
    "average_slot_crossings",
    "average_reward_per_request",
    "failure_count",
    "bandit_action_count",
    "deadline_acceptance_rate",
    "delay_margin_mean",
    "delay_margin_max",
    "delay_margin_jain_fairness",
]

summary_rows = []
for ablation in ablations:
    rows = [row for row in cycle_rows if row.get("ablation") == ablation]
    if not rows:
        rows = [row for row in slot_rows if row.get("ablation") == ablation]
    summary = {"ablation": ablation, "row_count": len(rows)}
    slot_count = len([row for row in slot_rows if row.get("ablation") == ablation])
    summary["slot_count"] = slot_count
    seeds = sorted({row.get("seed", "") for row in rows if row.get("seed", "")})
    summary["seed_count"] = len(seeds)
    summary["seeds"] = " ".join(seeds)
    for column in metric_columns:
        values = []
        for row in rows:
            value = row.get(column, "")
            if value in ("", "None", "null"):
                continue
            try:
                values.append(float(value))
            except ValueError:
                continue
        summary[f"mean_{column}"] = sum(values) / len(values) if values else ""
    summary_rows.append(summary)

write_rows(output_root / "ablation_metric_summary.csv", summary_rows)
PY

echo "Done. Outputs are under ${OUTPUT_ROOT}."
echo "Per-task progress bars and ETA are in each seed_<seed>.log file."
