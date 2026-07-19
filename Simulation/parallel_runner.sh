#!/usr/bin/env bash

# Shared bounded-parallel scheduler for the Simulation shell entry points.
# The caller sets `failed=0`, calls parallel_runner_init, launches a background
# process, and registers it with parallel_register_task "$!" "label".

format_seconds() {
  local total="${1:-0}"
  local hours minutes seconds
  (( total < 0 )) && total=0
  hours=$((total / 3600))
  minutes=$(((total % 3600) / 60))
  seconds=$((total % 60))
  if (( hours > 0 )); then
    printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
  else
    printf '%02d:%02d' "${minutes}" "${seconds}"
  fi
}

parallel_runner_init() {
  total_tasks="$1"
  max_parallel="$2"
  runner_started_at="$(date +%s)"
  completed_tasks=0
  completed_duration_sum=0
  pids=()
  labels=()
  task_started_at=()
}

parallel_register_task() {
  local pid="$1"
  local label="$2"
  pids+=("${pid}")
  labels+=("${label}")
  task_started_at+=("$(date +%s)")
}

parallel_print_progress() {
  local now elapsed average remaining eta active
  now="$(date +%s)"
  elapsed=$((now - runner_started_at))
  active="${#pids[@]}"
  remaining=$((total_tasks - completed_tasks))
  if (( completed_tasks > 0 )); then
    average=$((completed_duration_sum / completed_tasks))
    eta=$(((average * remaining + max_parallel - 1) / max_parallel))
    echo "Progress: ${completed_tasks}/${total_tasks} complete, ${active} running, elapsed=$(format_seconds "${elapsed}"), ETA≈$(format_seconds "${eta}")"
  else
    echo "Progress: 0/${total_tasks} complete, ${active} running, ETA=estimating"
  fi
}

parallel_reap_one() {
  local i pid label started now duration task_failed
  while true; do
    for i in "${!pids[@]}"; do
      pid="${pids[$i]}"
      if kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
      label="${labels[$i]}"
      started="${task_started_at[$i]}"
      task_failed=0
      if ! wait "${pid}"; then
        task_failed=1
        failed=1
      fi
      now="$(date +%s)"
      duration=$((now - started))
      completed_tasks=$((completed_tasks + 1))
      completed_duration_sum=$((completed_duration_sum + duration))
      unset 'pids[i]' 'labels[i]' 'task_started_at[i]'
      if (( task_failed )); then
        echo "Task failed after $(format_seconds "${duration}"): ${label}" >&2
      else
        echo "Task completed in $(format_seconds "${duration}"): ${label}"
      fi
      parallel_print_progress
      return
    done
    sleep 1
  done
}

wait_for_slot() {
  while (( ${#pids[@]} >= max_parallel )); do
    parallel_reap_one
  done
}

wait_active_tasks() {
  while (( ${#pids[@]} > 0 )); do
    parallel_reap_one
  done
}
