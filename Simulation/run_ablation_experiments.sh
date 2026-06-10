#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-42 43 44 45}"
GPUS="${GPUS:-0 1 2 3}"
ABLATIONS="${ABLATIONS:-full no_bandit shortest_hop_routing nearest_replica}"
MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/multi_seed_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/test_outputs/ablation_experiments}"
DEVICE="${DEVICE:-cuda}"
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
read -r -a ABLATION_ARRAY <<< "${ABLATIONS}"

if [[ "${#SEED_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one seed, got: ${SEEDS}" >&2
  exit 1
fi

if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one GPU id, got: ${GPUS}" >&2
  exit 1
fi

if [[ "${#ABLATION_ARRAY[@]}" -eq 0 ]]; then
  echo "Expected at least one ablation variant, got: ${ABLATIONS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "Running ablation experiments"
echo "  ablations: ${ABLATIONS}"
echo "  seeds: ${SEEDS}"
echo "  gpus: ${GPUS}"
echo "  model_root: ${MODEL_ROOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  device: ${DEVICE}"

gpu_count="${#GPU_ARRAY[@]}"
task_index=0
pids=()

for ablation in "${ABLATION_ARRAY[@]}"; do
  variant_dir="${OUTPUT_ROOT}/${ablation}"
  mkdir -p "${variant_dir}"
  for seed in "${SEED_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$((task_index % gpu_count))]}"
    log_file="${variant_dir}/seed_${seed}.log"
    echo "Launching ablation=${ablation} seed=${seed} on GPU=${gpu}; log=${log_file}"
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
    pids+=("$!")
    task_index=$((task_index + 1))
  done
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
  echo "At least one ablation task failed. Check logs under ${OUTPUT_ROOT}." >&2
  exit 1
fi

echo "All ablation tasks finished. Merging per-variant outputs."
for ablation in "${ABLATION_ARRAY[@]}"; do
  variant_dir="${OUTPUT_ROOT}/${ablation}"
  (
    cd "${PROJECT_ROOT}"
    "${PYTHON_BIN}" -m Simulation.tests.full_cycle_seed_distribution \
      --ablation "${ablation}" \
      --seeds "${SEEDS}" \
      --output-dir "${variant_dir}" \
      --plot-only
  )
done

echo "Building cross-variant merged metric tables."
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${ABLATIONS}" <<'PY'
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
for ablation in ablations:
    variant_dir = output_root / ablation
    for row in read_rows(variant_dir / "slot_metrics_by_seed.csv"):
        row.setdefault("ablation", ablation)
        slot_rows.append(row)
    for row in read_rows(variant_dir / "request_metrics_by_seed.csv"):
        row.setdefault("ablation", ablation)
        request_rows.append(row)

write_rows(output_root / "all_ablation_slot_metrics.csv", slot_rows)
write_rows(output_root / "all_ablation_request_metrics.csv", request_rows)

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
]

summary_rows = []
for ablation in ablations:
    rows = [row for row in slot_rows if row.get("ablation") == ablation]
    summary = {"ablation": ablation, "slot_count": len(rows)}
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
