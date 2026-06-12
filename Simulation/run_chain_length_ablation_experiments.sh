#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHAIN_LENGTHS="${CHAIN_LENGTHS:-5 10 15}"
SEEDS="${SEEDS:-42 43 44 45}"
RUN_ABLATIONS="${RUN_ABLATIONS:-${ABLATIONS:-ELARA ELARA-NB ELARA-NR ELARA-SH Fair-NFV SP-Routing SC-NFV}}"
GPUS="${GPUS:-0 1 2 3}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/test_outputs/chain_length_ablation_experiments}"
DEFAULT_ISL_CSV="${PROJECT_ROOT}/WalkerDeltaConstellationSimu/Walker_Delta_ISL_Simu.csv"
ISL_CSV="${ISL_CSV:-${DEFAULT_ISL_CSV}}"
DEVICE="${DEVICE:-cuda}"
CPU_WORKERS="${CPU_WORKERS:-4}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-9}"
TOTAL_ARRIVAL_LAMBDA="${TOTAL_ARRIVAL_LAMBDA:-4.9}"
BANDIT_PERIOD_SLOTS="${BANDIT_PERIOD_SLOTS:-10}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-ppo_gnn_latest.pth}"
BANDIT_STATS_NAME="${BANDIT_STATS_NAME:-bandit_arm_stats.csv}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

common_args=(
  --model-root "${MODEL_ROOT}"
  --bandit-period-slots "${BANDIT_PERIOD_SLOTS}"
  --checkpoint-name "${CHECKPOINT_NAME}"
  --bandit-stats-name "${BANDIT_STATS_NAME}"
  --progress-every "${PROGRESS_EVERY}"
  --isl-csv "${ISL_CSV}"
  --arrival-mode total_per_slot
  --total-arrival-lambda "${TOTAL_ARRIVAL_LAMBDA}"
)

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

read -r -a CHAIN_LENGTH_ARRAY <<< "${CHAIN_LENGTHS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a RUN_ABLATION_ARRAY <<< "${RUN_ABLATIONS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#CHAIN_LENGTH_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one chain length, got: ${CHAIN_LENGTHS}" >&2
  exit 1
fi

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

echo "Running chain-length ablation experiments"
echo "  chain_lengths: ${CHAIN_LENGTHS}"
echo "  ablations: ${RUN_ABLATIONS}"
echo "  seeds: ${SEEDS}"
echo "  total_arrival_lambda_per_slot: ${TOTAL_ARRIVAL_LAMBDA}"
echo "  model_root: ${MODEL_ROOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  device: ${DEVICE}"
if [[ "${DEVICE}" == "cpu" ]]; then
  echo "  cpu_workers: ${CPU_WORKERS}"
else
  echo "  gpus: ${GPUS}"
  echo "  gpu_workers_per_gpu: ${GPU_WORKERS_PER_GPU}"
fi

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

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for chain_length in "${CHAIN_LENGTH_ARRAY[@]}"; do
  length_dir="${OUTPUT_ROOT}/chain_length_${chain_length}"
  mkdir -p "${length_dir}"
  for ablation in "${RUN_ABLATION_ARRAY[@]}"; do
    variant_dir="${length_dir}/${ablation}"
    mkdir -p "${variant_dir}"
    for seed in "${SEED_ARRAY[@]}"; do
      wait_for_slot
      log_file="${variant_dir}/seed_${seed}.log"
      label="chain_length=${chain_length} ablation=${ablation} seed=${seed}"
      if [[ "${DEVICE}" == "cpu" ]]; then
        echo "Launching ${label} on CPU; log=${log_file}"
        (
          cd "${PROJECT_ROOT}"
          CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
            --chain-length-filter "${chain_length}" \
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
            --chain-length-filter "${chain_length}" \
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
done

wait_active_tasks

if [[ "${failed}" -ne 0 ]]; then
  echo "At least one chain-length ablation task failed. Check logs under ${OUTPUT_ROOT}." >&2
  exit 1
fi

echo "All chain-length ablation tasks finished. Merging outputs."
for chain_length in "${CHAIN_LENGTH_ARRAY[@]}"; do
  length_dir="${OUTPUT_ROOT}/chain_length_${chain_length}"
  for ablation in "${RUN_ABLATION_ARRAY[@]}"; do
    variant_dir="${length_dir}/${ablation}"
    variant_seeds="$(existing_seed_list "${variant_dir}" || true)"
    if [[ -z "${variant_seeds}" ]]; then
      echo "Skipping chain_length=${chain_length} ablation=${ablation}; no seed metrics found." >&2
      continue
    fi
    echo "Merging chain_length=${chain_length} ablation=${ablation} seeds=${variant_seeds}"
    (
      cd "${PROJECT_ROOT}"
      "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
        --ablation "${ablation}" \
        --seeds "${variant_seeds}" \
        --output-dir "${variant_dir}" \
        --plot-only
    )
  done
done

echo "Building merged chain-length metric tables."
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${CHAIN_LENGTHS}" "${RUN_ABLATIONS}" <<'PY'
import csv
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
chain_lengths = sys.argv[2].split()
name_map = {
    "full": "ELARA",
    "no_bandit": "ELARA-NB",
    "shortest_hop_routing": "ELARA-SH",
    "nearest_replica": "ELARA-NR",
    "service_pressure": "SP-Routing",
    "sc_nfv": "SC-NFV",
    "fairness_nfv_greedy": "Fair-NFV",
}

def canonical_ablation_name(name):
    return name_map.get(name, name)

ablations = [canonical_ablation_name(name) for name in sys.argv[3].split()]

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

metric_columns = [
    "task_completion_rate",
    "success_rate",
    "average_end_to_end_delay_s",
    "average_energy_j",
    "p95_end_to_end_delay_s",
    "average_communication_delay_s",
    "average_slot_crossings",
    "failure_count",
    "deadline_acceptance_rate",
    "delay_margin_mean",
    "delay_margin_max",
    "delay_margin_jain_fairness",
]

all_slot_rows = []
all_request_rows = []
all_hop_rows = []
all_cycle_rows = []
all_summary_rows = []

for chain_length in chain_lengths:
    length_dir = output_root / f"chain_length_{chain_length}"
    length_slot_rows = []
    length_request_rows = []
    length_hop_rows = []
    length_cycle_rows = []
    for ablation in ablations:
        variant_dir = length_dir / ablation
        for row in read_rows(variant_dir / "slot_metrics_by_seed.csv"):
            row["ablation"] = canonical_ablation_name(row.get("ablation", ablation))
            row["chain_length_filter"] = chain_length
            length_slot_rows.append(row)
            all_slot_rows.append(row)
        for row in read_rows(variant_dir / "request_metrics_by_seed.csv"):
            row["ablation"] = canonical_ablation_name(row.get("ablation", ablation))
            row["chain_length_filter"] = chain_length
            length_request_rows.append(row)
            all_request_rows.append(row)
        for row in read_rows(variant_dir / "request_hop_metrics_by_seed.csv"):
            row["ablation"] = canonical_ablation_name(row.get("ablation", ablation))
            row["chain_length_filter"] = chain_length
            length_hop_rows.append(row)
            all_hop_rows.append(row)
        for row in read_rows(variant_dir / "cycle_metrics_by_seed.csv"):
            row["ablation"] = canonical_ablation_name(row.get("ablation", ablation))
            row.setdefault("chain_length_filter", chain_length)
            length_cycle_rows.append(row)
            all_cycle_rows.append(row)

    write_rows(length_dir / "all_ablation_slot_metrics.csv", length_slot_rows)
    write_rows(length_dir / "all_ablation_request_metrics.csv", length_request_rows)
    write_rows(length_dir / "all_ablation_request_hop_metrics.csv", length_hop_rows)
    write_rows(length_dir / "all_ablation_cycle_metrics.csv", length_cycle_rows)

    for ablation in ablations:
        rows = [row for row in length_cycle_rows if row.get("ablation") == ablation]
        summary = {
            "chain_length_filter": chain_length,
            "ablation": ablation,
            "row_count": len(rows),
            "seed_count": len({row.get("seed", "") for row in rows if row.get("seed", "")}),
            "seeds": " ".join(sorted({row.get("seed", "") for row in rows if row.get("seed", "")})),
        }
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
        all_summary_rows.append(summary)

write_rows(output_root / "all_chain_length_slot_metrics.csv", all_slot_rows)
write_rows(output_root / "all_chain_length_request_metrics.csv", all_request_rows)
write_rows(output_root / "all_chain_length_request_hop_metrics.csv", all_hop_rows)
write_rows(output_root / "all_chain_length_cycle_metrics.csv", all_cycle_rows)
write_rows(output_root / "chain_length_ablation_metric_summary.csv", all_summary_rows)
PY

echo "Done. Outputs are under ${OUTPUT_ROOT}."
echo "Per-task progress bars and ETA are in each seed_<seed>.log file."
