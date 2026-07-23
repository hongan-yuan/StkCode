from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .parallel_runner import (
    detect_gpu_ids,
    detect_mps,
    format_duration,
    gpu_for_task,
    select_accelerator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELARA_ROOT = PROJECT_ROOT / "ELARA"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "Simulation" / "multi_seed_runs"
BASELINES = (
    "ELARA",
    "ELARA-NB",
    "ELARA-NR",
    "ELARA-SH",
    "SECO",
    "SP-Routing",
    "SC-NFV",
)
PPO_BASELINES = frozenset(("ELARA", "ELARA-NB", "ELARA-SH"))
SEED_OUTPUTS = (
    "slot_metrics.csv",
    "cycle_request_metrics.csv",
    "request_metrics.csv",
    "request_hop_metrics.csv",
    "summary.json",
)
MERGED_METRICS = (
    "slot_metrics_by_seed.csv",
    "cycle_request_metrics_by_seed.csv",
    "request_metrics_by_seed.csv",
    "request_hop_metrics_by_seed.csv",
    "cycle_metrics_by_seed.csv",
)
CROSS_BASELINE_OUTPUTS = {
    "slot_metrics_by_seed.csv": "all_ablation_slot_metrics.csv",
    "cycle_request_metrics_by_seed.csv": "all_ablation_cycle_request_metrics.csv",
    "request_metrics_by_seed.csv": "all_ablation_request_metrics.csv",
    "request_hop_metrics_by_seed.csv": "all_ablation_request_hop_metrics.csv",
    "cycle_metrics_by_seed.csv": "all_ablation_cycle_metrics.csv",
}
PROGRESS_PATTERN = re.compile(r"\b(\d+)/(\d+)\s+elapsed=")


def _timestamped_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ELARA_ROOT / "outputs" / "baseline-tests" / stamp


def _split_values(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _parse_baselines(value: str) -> list[str]:
    requested = _split_values(value)
    unknown = [name for name in requested if name not in BASELINES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown baseline(s): {', '.join(unknown)}; choices: {', '.join(BASELINES)}"
        )
    return list(dict.fromkeys(requested))


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in _split_values(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return list(dict.fromkeys(seeds))


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run ELARA and all Simulation baselines with bounded parallelism. "
            "Every result and log is stored below the ELARA directory."
        )
    )
    parser.add_argument(
        "--baselines",
        type=_parse_baselines,
        default=list(BASELINES),
        help="comma or space separated baseline names",
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=[42, 43, 44, 45]
    )
    parser.add_argument("--tasks", type=int, default=4, help="maximum concurrent tasks")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default="ppo_gnn_latest.pth")
    parser.add_argument("--bandit-stats-name", default="bandit_arm_stats.csv")
    parser.add_argument("--bandit-period-slots", type=int, default=10)
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--arrival-lambda", type=float)
    parser.add_argument(
        "--arrival-mode",
        choices=("per_template", "total_per_slot"),
        default="per_template",
    )
    parser.add_argument("--total-arrival-lambda", type=float)
    parser.add_argument("--chain-length-filter", type=int, choices=(5, 10, 15))
    parser.add_argument("--request-template-csv", type=Path)
    parser.add_argument("--isl-csv", type=Path)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-load-checkpoint", action="store_true")
    parser.add_argument("--no-load-bandit", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)

    if args.tasks < 1:
        parser.error("--tasks must be at least 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    managed = {
        "--ablation", "--seeds", "--output-dir", "--model-root", "--device",
        "--skip-aggregate", "--plot-only",
    }
    for argument in forwarded:
        option = argument.split("=", 1)[0]
        if option in managed:
            parser.error(f"{option} is managed by the baseline test runner")

    output_root = args.output_root or _timestamped_output_root()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    args.output_root = output_root.resolve()
    elara_root = ELARA_ROOT.resolve()
    if args.output_root != elara_root and elara_root not in args.output_root.parents:
        parser.error("--output-root must be located inside the ELARA directory")

    if not args.model_root.is_absolute():
        args.model_root = (PROJECT_ROOT / args.model_root).resolve()
    else:
        args.model_root = args.model_root.resolve()
    return args, forwarded


def _append_option(command: list[str], option: str, value) -> None:
    if value is not None:
        command.extend((option, str(value)))


def _job_specs(args, forwarded: list[str], accelerator: str, gpu_ids: list[str]):
    jobs = []
    expected_slots = args.max_slots or 606
    for baseline in args.baselines:
        baseline_dir = args.output_root / baseline
        for seed in args.seeds:
            index = len(jobs)
            gpu = gpu_for_task(index, gpu_ids) if accelerator == "cuda" else None
            command = [
                sys.executable,
                "-m",
                "Simulation.tests.full_cycle_seed_distribution",
                "--ablation",
                baseline,
                "--seeds",
                str(seed),
                "--output-dir",
                str(baseline_dir),
                "--model-root",
                str(args.model_root),
                "--device",
                accelerator,
                "--checkpoint-name",
                args.checkpoint_name,
                "--bandit-stats-name",
                args.bandit_stats_name,
                "--bandit-period-slots",
                str(args.bandit_period_slots),
                "--arrival-mode",
                args.arrival_mode,
                "--progress-every",
                str(args.progress_every),
                "--skip-aggregate",
            ]
            _append_option(command, "--max-slots", args.max_slots)
            _append_option(command, "--arrival-lambda", args.arrival_lambda)
            _append_option(command, "--total-arrival-lambda", args.total_arrival_lambda)
            _append_option(command, "--chain-length-filter", args.chain_length_filter)
            _append_option(command, "--request-template-csv", args.request_template_csv)
            _append_option(command, "--isl-csv", args.isl_csv)
            if args.no_load_checkpoint:
                command.append("--no-load-checkpoint")
            if args.no_load_bandit:
                command.append("--no-load-bandit")
            command.extend(forwarded)
            jobs.append(
                {
                    "index": index,
                    "baseline": baseline,
                    "seed": seed,
                    "gpu": gpu,
                    "accelerator": accelerator,
                    "output_dir": str(baseline_dir / f"seed_{seed}"),
                    "log_file": str(
                        args.output_root / "logs" / f"{baseline}_seed_{seed}.log"
                    ),
                    "expected_slots": expected_slots,
                    "completed_slots": 0,
                    "command": command,
                    "status": "pending",
                }
            )
    return jobs


def _validate_checkpoints(args) -> None:
    if args.no_load_checkpoint or not PPO_BASELINES.intersection(args.baselines):
        return
    missing = [
        args.model_root / f"seed_{seed}" / args.checkpoint_name
        for seed in args.seeds
        if not (args.model_root / f"seed_{seed}" / args.checkpoint_name).is_file()
    ]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            "Missing PPO checkpoints required by ELARA, ELARA-NB, or ELARA-SH:\n"
            f"  {paths}\nUse --model-root/--checkpoint-name or explicitly pass "
            "--no-load-checkpoint for an untrained smoke test."
        )


def _read_completed_slots(job: dict) -> int:
    try:
        content = Path(job["log_file"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return int(job.get("completed_slots", 0))
    matches = PROGRESS_PATTERN.findall(content)
    if not matches:
        return int(job.get("completed_slots", 0))
    completed, total = map(int, matches[-1])
    job["expected_slots"] = total
    job["completed_slots"] = min(completed, total)
    return job["completed_slots"]


def _progress_line(jobs: list[dict], started_at: float, width: int = 28) -> str:
    completed = sum(_read_completed_slots(job) for job in jobs)
    total = sum(int(job["expected_slots"]) for job in jobs)
    succeeded = sum(job["status"] == "succeeded" for job in jobs)
    running = sum(job["status"] == "running" for job in jobs)
    failed = sum(job["status"] == "failed" for job in jobs)
    fraction = completed / max(1, total)
    filled = min(width, int(fraction * width))
    bar = ("=" * filled + (">" if filled < width else "")).ljust(width, "-")
    elapsed = max(1.0e-9, time.monotonic() - started_at)
    eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
    return (
        f"[{bar}] {fraction * 100:6.2f}% slots {completed}/{total} | "
        f"tasks done {succeeded}/{len(jobs)} running {running} failed {failed} | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
    )


def _start_job(job: dict):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = job["gpu"] or ""
    if job["accelerator"] == "mps":
        environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    log_path = Path(job["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        job["command"],
        cwd=PROJECT_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat(timespec="seconds")
    return process, handle


def _run_jobs(args, jobs: list[dict]) -> bool:
    pending = list(jobs)
    active: list[tuple[dict, subprocess.Popen, object]] = []
    started_at = time.monotonic()
    interactive = sys.stdout.isatty()
    last_render = 0.0
    try:
        while pending or active:
            while pending and len(active) < args.tasks:
                job = pending.pop(0)
                process, handle = _start_job(job)
                active.append((job, process, handle))
            for job, process, handle in list(active):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                job["return_code"] = return_code
                job["status"] = "succeeded" if return_code == 0 else "failed"
                job["finished_at"] = datetime.now().isoformat(timespec="seconds")
                if return_code == 0:
                    job["completed_slots"] = job["expected_slots"]
                active.remove((job, process, handle))
            now = time.monotonic()
            if now - last_render >= (0.5 if interactive else 5.0) or not (pending or active):
                line = _progress_line(jobs, started_at)
                print(f"\r{line}" if interactive else line, end="" if interactive else "\n", flush=True)
                last_render = now
            if pending or active:
                time.sleep(0.2)
        if interactive:
            print()
    except KeyboardInterrupt:
        print("\nInterrupted; terminating active baseline tasks.", file=sys.stderr)
        for job, process, handle in active:
            if process.poll() is None:
                process.terminate()
            process.wait()
            handle.close()
            job["return_code"] = process.returncode
            job["status"] = "interrupted"
        for job in pending:
            job["status"] = "cancelled"
        return False
    return all(job.get("return_code") == 0 for job in jobs)


def _require_seed_outputs(args) -> None:
    missing = []
    for baseline in args.baselines:
        for seed in args.seeds:
            seed_dir = args.output_root / baseline / f"seed_{seed}"
            missing.extend(path for name in SEED_OUTPUTS if not (path := seed_dir / name).is_file())
    if missing:
        raise RuntimeError("Missing baseline outputs:\n  " + "\n  ".join(map(str, missing)))


def _aggregate_variants(args) -> None:
    seed_list = " ".join(map(str, args.seeds))
    for baseline in args.baselines:
        log_path = args.output_root / "logs" / f"aggregate_{baseline}.log"
        command = [
            sys.executable,
            "-m",
            "Simulation.tests.full_cycle_seed_distribution",
            "--ablation",
            baseline,
            "--seeds",
            seed_list,
            "--output-dir",
            str(args.output_root / baseline),
            "--plot-only",
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Aggregation failed for {baseline}; see {log_path}")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _finite_mean(rows: list[dict], field: str):
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else ""


def _aggregate_across_baselines(args) -> None:
    combined: dict[str, list[dict]] = {name: [] for name in MERGED_METRICS}
    for baseline in args.baselines:
        for filename in MERGED_METRICS:
            rows = _read_csv(args.output_root / baseline / filename)
            for row in rows:
                row["ablation"] = baseline
            combined[filename].extend(rows)
    for filename, rows in combined.items():
        _write_csv(args.output_root / CROSS_BASELINE_OUTPUTS[filename], rows)

    metric_fields = (
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
    )
    cycle_rows = combined["cycle_metrics_by_seed.csv"]
    slot_rows = combined["slot_metrics_by_seed.csv"]
    summary_rows = []
    for baseline in args.baselines:
        rows = [row for row in cycle_rows if row.get("ablation") == baseline]
        if not rows:
            rows = [row for row in slot_rows if row.get("ablation") == baseline]
        summary = {
            "ablation": baseline,
            "row_count": len(rows),
            "seed_count": len({row.get("seed") for row in rows if row.get("seed")}),
        }
        summary.update({f"mean_{field}": _finite_mean(rows, field) for field in metric_fields})
        summary_rows.append(summary)
    _write_csv(args.output_root / "ablation_metric_summary.csv", summary_rows)


def main(argv: list[str] | None = None) -> int:
    args, forwarded = parse_args(argv)
    _validate_checkpoints(args)
    gpu_ids = detect_gpu_ids()
    mps_available = detect_mps()
    try:
        accelerator = select_accelerator(args.device, gpu_ids, mps_available)
    except RuntimeError as exc:
        print(f"Device error: {exc}", file=sys.stderr)
        return 2
    jobs = _job_specs(args, forwarded, accelerator, gpu_ids)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "baseline_test_manifest.json"
    manifest = {
        "baselines": args.baselines,
        "seeds": args.seeds,
        "task_count": len(jobs),
        "max_parallel_tasks": args.tasks,
        "accelerator": accelerator,
        "cuda_gpu_ids": gpu_ids,
        "mps_available": mps_available,
        "model_root": str(args.model_root),
        "output_root": str(args.output_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": jobs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Seeds: {', '.join(map(str, args.seeds))}")
    print(f"Accelerator: {accelerator}; CUDA GPUs: {', '.join(gpu_ids) or 'none'}")
    print(f"Parallel tasks: {args.tasks}; total tasks: {len(jobs)}")
    print(f"Output root: {args.output_root}")
    if args.dry_run:
        for job in jobs:
            assignment = f"cuda:{job['gpu']}" if job["gpu"] is not None else accelerator
            print(f"[{assignment}] {shlex.join(job['command'])}")
        return 0

    succeeded = _run_jobs(args, jobs)
    if not succeeded:
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        failed = [job for job in jobs if job.get("return_code") != 0]
        print("Failed tasks:", file=sys.stderr)
        for job in failed:
            print(
                f"  {job['baseline']} seed={job['seed']}: {job['log_file']}",
                file=sys.stderr,
            )
        return 1

    try:
        _require_seed_outputs(args)
        if not args.skip_aggregate:
            print("Per-seed tests complete; aggregating metrics.", flush=True)
            _aggregate_variants(args)
            _aggregate_across_baselines(args)
    except RuntimeError as exc:
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["status"] = "failed"
        manifest["aggregation_error"] = str(exc)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Baseline aggregation failed: {exc}", file=sys.stderr)
        return 1
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "succeeded"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Baseline testing complete: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
