from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .comparison_policies import BASELINES, BASELINE_DESCRIPTIONS
from .config import PROJECT_ROOT
from .parallel_runner import (
    detect_gpu_ids,
    detect_mps,
    gpu_for_task,
    progress_line,
    select_accelerator,
)
from .request_templates import DEFAULT_TEMPLATE_PATH, load_templates


ELARA_ROOT = PROJECT_ROOT / "ELARA"
DEFAULT_OUTPUT_PARENT = ELARA_ROOT / "outputs" / "baseline-tests"


def _split_strings(value: str) -> list[str]:
    return [
        item
        for item in value.replace(",", " ").split()
        if item
    ]


def _split_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in _split_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return list(dict.fromkeys(values))


def _split_baselines(value: str) -> list[str]:
    values = _split_strings(value)
    unknown = [item for item in values if item not in BASELINES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown baseline(s): {', '.join(unknown)}"
        )
    return list(dict.fromkeys(values))


def _timestamped_output_root() -> Path:
    return DEFAULT_OUTPUT_PARENT / datetime.now().strftime("%Y%m%d-%H%M%S")


def discover_model_root(model_seeds: list[int]) -> Path:
    sensitivity_root = ELARA_ROOT / "outputs" / "sensitivity"
    candidates = sorted(
        sensitivity_root.glob("*/latency_energy_weights/d50_e50"),
        reverse=True,
    )
    for candidate in candidates:
        valid = True
        for seed in model_seeds:
            train_dir = candidate / f"seed_{seed}" / "train"
            config_path = train_dir / "config.json"
            checkpoint = train_dir / "ppo_final.pt"
            if not config_path.is_file() or not checkpoint.is_file():
                valid = False
                break
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                valid = (
                    abs(float(config["delay_weight"]) - 0.5) <= 1.0e-9
                    and abs(float(config["energy_weight"]) - 0.5) <= 1.0e-9
                    and int(config["route_max_paths_per_slot"]) == 3
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                break
        if valid:
            return candidate.resolve()
    raise FileNotFoundError(
        "no complete latency:energy=0.5:0.5 model set was found below "
        f"{sensitivity_root}"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare ELARA with all requested baselines using the trained "
            "0.5:0.5, paths=3 models and shared out-of-sample scenarios."
        )
    )
    parser.add_argument(
        "--baselines", type=_split_baselines, default=list(BASELINES)
    )
    parser.add_argument(
        "--model-seeds", type=_split_ints, default=[42, 43, 44, 45]
    )
    parser.add_argument(
        "--test-seeds", type=_split_ints, default=[142, 143, 144, 145]
    )
    parser.add_argument(
        "--background-seeds",
        type=_split_ints,
        help=(
            "one seed per model/test pair; defaults to each test seed plus "
            "100000"
        ),
    )
    parser.add_argument(
        "--chain-lengths", type=_split_ints, default=[5, 10, 15]
    )
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--request-template-file",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
    )
    parser.add_argument(
        "--total-arrival-lambda",
        type=float,
        help=(
            "total Poisson arrivals per slot for every chain length; defaults "
            "to the total rate used for training"
        ),
    )
    parser.add_argument("--max-slots", type=int, default=606)
    parser.add_argument("--checkpoint-name", default="ppo_final.pt")
    parser.add_argument(
        "--control-state-checkpoint-name",
        help=(
            "checkpoint used only to initialize placement and Bandit state; "
            "defaults to --checkpoint-name"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.model_seeds) != len(args.test_seeds):
        parser.error("--model-seeds and --test-seeds must have equal lengths")
    if args.background_seeds is None:
        args.background_seeds = [
            seed + 100_000 for seed in args.test_seeds
        ]
    if len(args.background_seeds) != len(args.model_seeds):
        parser.error("--background-seeds must match the model-seed count")
    if set(args.model_seeds) & set(args.test_seeds):
        parser.error("test seeds must be different from training model seeds")
    if set(args.model_seeds) & set(args.background_seeds):
        parser.error("background seeds must differ from training model seeds")
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    if args.max_slots < 1:
        parser.error("--max-slots must be positive")
    if any(length not in {5, 10, 15} for length in args.chain_lengths):
        parser.error("--chain-lengths may only contain 5, 10, and 15")
    if args.total_arrival_lambda is not None and args.total_arrival_lambda <= 0:
        parser.error("--total-arrival-lambda must be positive")
    args.request_template_file = args.request_template_file.expanduser().resolve()
    if not args.request_template_file.is_file():
        parser.error(
            f"request template file does not exist: {args.request_template_file}"
        )
    args.model_root = (
        args.model_root.expanduser().resolve()
        if args.model_root is not None
        else discover_model_root(args.model_seeds)
    )
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _timestamped_output_root().resolve()
    )
    if ELARA_ROOT.resolve() not in args.output_root.parents:
        parser.error("--output-root must be below the ELARA directory")
    return args


def _validate_models(args) -> None:
    for seed in args.model_seeds:
        train_dir = args.model_root / f"seed_{seed}" / "train"
        config_path = train_dir / "config.json"
        checkpoint = train_dir / args.checkpoint_name
        control_checkpoint = train_dir / (
            args.control_state_checkpoint_name or args.checkpoint_name
        )
        if (
            not config_path.is_file()
            or not checkpoint.is_file()
            or not control_checkpoint.is_file()
        ):
            raise FileNotFoundError(
                "missing config, policy checkpoint, or control-state "
                f"checkpoint for model seed {seed}: {train_dir}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            abs(float(config.get("delay_weight", -1.0)) - 0.5) > 1.0e-9
            or abs(float(config.get("energy_weight", -1.0)) - 0.5) > 1.0e-9
            or int(config.get("route_max_paths_per_slot", -1)) != 3
        ):
            raise ValueError(
                f"model seed {seed} is not a 0.5:0.5, paths=3 model"
            )


def _training_total_arrival_lambda(args) -> float:
    if args.total_arrival_lambda is not None:
        return float(args.total_arrival_lambda)
    first_config = json.loads(
        (
            args.model_root
            / f"seed_{args.model_seeds[0]}"
            / "train"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    per_template = float(
        first_config["request_arrival_lambda_per_template_per_slot"]
    )
    template_count = len(load_templates(args.request_template_file))
    return per_template * template_count


def _job_specs(args, accelerator: str, gpu_ids: list[str]) -> list[dict]:
    jobs = []
    for chain_length in args.chain_lengths:
        for model_seed, test_seed, background_seed in zip(
            args.model_seeds, args.test_seeds, args.background_seeds
        ):
            model_dir = args.model_root / f"seed_{model_seed}" / "train"
            scenario = (
                f"model_{model_seed}_test_{test_seed}_background_{background_seed}"
            )
            for baseline in args.baselines:
                index = len(jobs)
                gpu = (
                    gpu_for_task(index, gpu_ids)
                    if accelerator == "cuda"
                    else None
                )
                output_dir = (
                    args.output_root
                    / f"chain_{chain_length}"
                    / baseline
                    / scenario
                )
                stem = (
                    f"chain_{chain_length}_{baseline}_model_{model_seed}"
                    f"_test_{test_seed}"
                )
                progress_file = (
                    args.output_root / "logs" / f"{stem}.progress.json"
                )
                command = [
                    sys.executable,
                    "-m",
                    "ELARA.comparison_evaluate",
                    "--baseline",
                    baseline,
                    "--model-seed",
                    str(model_seed),
                    "--test-seed",
                    str(test_seed),
                    "--background-seed",
                    str(background_seed),
                    "--chain-length",
                    str(chain_length),
                    "--model-dir",
                    str(model_dir),
                    "--checkpoint-name",
                    args.checkpoint_name,
                    "--request-template-file",
                    str(args.request_template_file),
                    "--total-arrival-lambda",
                    str(args.total_arrival_lambda),
                    "--max-slots",
                    str(args.max_slots),
                    "--device",
                    accelerator,
                    "--output-dir",
                    str(output_dir),
                    "--progress-file",
                    str(progress_file),
                ]
                if args.control_state_checkpoint_name:
                    command.extend(
                        [
                            "--control-state-checkpoint-name",
                            args.control_state_checkpoint_name,
                        ]
                    )
                jobs.append(
                    {
                        "index": index,
                        "baseline": baseline,
                        "model_seed": model_seed,
                        "test_seed": test_seed,
                        "background_seed": background_seed,
                        "chain_length": chain_length,
                        "accelerator": accelerator,
                        "gpu": gpu,
                        "total": args.max_slots,
                        "unit": "slots",
                        "output_dir": str(output_dir),
                        "progress_file": str(progress_file),
                        "log_file": str(
                            args.output_root / "logs" / f"{stem}.log"
                        ),
                        "command": command,
                        "status": "pending",
                    }
                )
    return jobs


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
    active = []
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
                job["status"] = (
                    "succeeded" if return_code == 0 else "failed"
                )
                job["finished_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )
                active.remove((job, process, handle))
            now = time.monotonic()
            if (
                now - last_render >= (0.5 if interactive else 5.0)
                or not (pending or active)
            ):
                line = progress_line(jobs)
                print(
                    f"\r{line}" if interactive else line,
                    end="" if interactive else "\n",
                    flush=True,
                )
                last_render = now
            if pending or active:
                time.sleep(0.2)
        if interactive:
            print()
    except KeyboardInterrupt:
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


def _read_summary(job: dict) -> dict:
    path = Path(job["output_dir"]) / "summary.json"
    if not path.is_file():
        raise RuntimeError(f"missing comparison summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_shared_scenarios(jobs: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, int, int], list[dict]] = {}
    summaries = []
    for job in jobs:
        summary = _read_summary(job)
        summaries.append(summary)
        key = (
            job["model_seed"],
            job["test_seed"],
            job["background_seed"],
            job["chain_length"],
        )
        groups.setdefault(key, []).append(summary)
    for key, rows in groups.items():
        hashes = {row["request_stream_hash"] for row in rows}
        counts = {row["request_count"] for row in rows}
        if len(hashes) != 1 or len(counts) != 1:
            raise RuntimeError(
                "baselines did not receive the same request stream for "
                f"scenario {key}"
            )
        control_hashes = {
            row.get("initial_control_state_hash")
            for row in rows
            if row.get("initial_control_state_hash")
        }
        placement_hashes = {
            row.get("initial_placement_hash")
            for row in rows
            if row.get("initial_placement_hash")
        }
        if len(control_hashes) > 1 or len(placement_hashes) > 1:
            raise RuntimeError(
                "baselines did not start from the same control state for "
                f"scenario {key}"
            )
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary(summary: dict) -> dict:
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"bandit", "migration_actions"}
    }
    row.update(
        {
            f"migration_{key}_count": value
            for key, value in summary["migration_actions"].items()
        }
    )
    return row


def _aggregate_summary(rows: list[dict]) -> list[dict]:
    metrics = (
        "success_rate",
        "mean_return",
        "mean_latency_s",
        "p95_latency_s",
        "mean_energy_j",
        "mean_route_slot_crossings",
        "mean_route_phase_count",
        "mean_route_used_path_count",
    )
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["ablation"]), int(row["chain_length"])), []
        ).append(row)
    result = []
    for (baseline, chain_length), group in sorted(groups.items()):
        output = {
            "ablation": baseline,
            "chain_length": chain_length,
            "seed_count": len(group),
            "request_count": sum(int(row["request_count"]) for row in group),
            "failure_count": sum(int(row["failure_count"]) for row in group),
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            ]
            mean = statistics.fmean(values) if values else math.nan
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95"] = (
                1.96 * std / math.sqrt(len(values)) if values else math.nan
            )
        result.append(output)
    return result


def _merge_job_csvs(jobs: list[dict], filename: str, output: Path) -> None:
    writer = None
    handle = None
    try:
        for job in jobs:
            source = Path(job["output_dir"]) / filename
            with source.open(
                "r", encoding="utf-8-sig", newline=""
            ) as input_handle:
                reader = csv.DictReader(input_handle)
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    handle = output.open(
                        "w", encoding="utf-8-sig", newline=""
                    )
                    writer = csv.DictWriter(
                        handle, fieldnames=reader.fieldnames
                    )
                    writer.writeheader()
                writer.writerows(reader)
    finally:
        if handle is not None:
            handle.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_models(args)
    args.total_arrival_lambda = _training_total_arrival_lambda(args)
    gpu_ids = detect_gpu_ids()
    mps_available = detect_mps()
    try:
        accelerator = select_accelerator(
            args.device, gpu_ids, mps_available
        )
    except RuntimeError as exc:
        print(f"Device error: {exc}", file=sys.stderr)
        return 2
    jobs = _job_specs(args, accelerator, gpu_ids)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "comparison_manifest.json"
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "baselines": args.baselines,
        "baseline_definitions": {
            baseline: BASELINE_DESCRIPTIONS[baseline]
            for baseline in args.baselines
        },
        "model_seeds": args.model_seeds,
        "test_seeds": args.test_seeds,
        "background_seeds": args.background_seeds,
        "chain_lengths": args.chain_lengths,
        "request_template_file": str(args.request_template_file),
        "total_arrival_lambda_per_slot": args.total_arrival_lambda,
        "model_root": str(args.model_root),
        "policy_checkpoint_name": args.checkpoint_name,
        "control_state_checkpoint_name": (
            args.control_state_checkpoint_name or args.checkpoint_name
        ),
        "output_root": str(args.output_root),
        "accelerator": accelerator,
        "gpu_ids": gpu_ids,
        "parallel_tasks": args.tasks,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": jobs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"{len(jobs)} jobs; concurrency={args.tasks}; "
        f"device={accelerator}; model_root={args.model_root}"
    )
    if args.dry_run:
        for job in jobs:
            assignment = (
                f"cuda:{job['gpu']}"
                if job["gpu"] is not None
                else accelerator
            )
            print(f"[{assignment}] {shlex.join(job['command'])}")
        return 0

    success = _run_jobs(args, jobs)
    if not success:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return 1
    try:
        summaries = _verify_shared_scenarios(jobs)
        flat = [_flatten_summary(summary) for summary in summaries]
        _write_csv(args.output_root / "comparison_summary.csv", flat)
        _write_csv(
            args.output_root / "comparison_summary_by_chain.csv",
            _aggregate_summary(summaries),
        )
        _merge_job_csvs(
            jobs,
            "request_metrics.csv",
            args.output_root / "all_request_metrics.csv",
        )
        _merge_job_csvs(
            jobs,
            "slot_metrics.csv",
            args.output_root / "all_slot_metrics.csv",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        manifest["status"] = "failed"
        manifest["aggregation_error"] = str(exc)
        manifest["finished_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 1

    manifest["status"] = "succeeded"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Comparison testing complete: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
