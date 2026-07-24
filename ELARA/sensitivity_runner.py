from __future__ import annotations

import argparse
import csv
import json
import os
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
from .request_templates import (
    DEFAULT_CHAIN_PLAN,
    DEFAULT_TEMPLATE_PATH,
    generate_templates,
    save_templates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELARA_ROOT = PROJECT_ROOT / "ELARA"
DEFAULT_TEMPLATE_FILE = DEFAULT_TEMPLATE_PATH


def _split_ints(value: str) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _split_weights(value: str) -> tuple[tuple[float, float], ...]:
    result = []
    try:
        for item in value.split(","):
            delay, energy = (float(part.strip()) for part in item.split(":", 1))
            if delay < 0.0 or energy < 0.0 or abs(delay + energy - 1.0) > 1.0e-9:
                raise ValueError
            result.append((delay, energy))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "weights must use delay:energy pairs that are nonnegative and sum to one"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one weight pair is required")
    return tuple(dict.fromkeys(result))


def _timestamped_root() -> Path:
    return ELARA_ROOT / "outputs" / "sensitivity" / datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate latency/energy-weight and routing-path "
            "sensitivity experiments with bounded parallelism."
        )
    )
    parser.add_argument("--phase", choices=("train", "test", "all"), default="all")
    parser.add_argument("--seeds", type=_split_ints, default=(42, 43, 44, 45))
    parser.add_argument("--weights", type=_split_weights, default=((0.5, 0.5), (0.35, 0.65), (0.65, 0.35)))
    parser.add_argument("--route-max-paths", type=_split_ints, default=(3, 5, 7))
    parser.add_argument(
        "--tasks",
        type=int,
        help="compatibility override that sets both training and testing concurrency",
    )
    parser.add_argument("--train-tasks", type=int, default=10)
    parser.add_argument("--test-tasks", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--request-template-file", type=Path, default=DEFAULT_TEMPLATE_FILE)
    parser.add_argument("--template-seed", type=int, default=2026)
    parser.add_argument("--max-trace-slots", type=int, default=606)
    parser.add_argument("--arrival-lambda", type=float, default=0.35)
    parser.add_argument("--compute-capacity-scale", type=float, default=1.0)
    parser.add_argument("--link-capacity-scale", type=float, default=1.0)
    parser.add_argument("--background-load-scale", type=float, default=0.5)
    parser.add_argument("--request-data-scale", type=float, default=1.0)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-update-interval-slots", type=int, default=5)
    parser.add_argument("--pretrain-cycles", type=int, default=1)
    parser.add_argument("--joint-training-cycles", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.tasks is not None:
        args.train_tasks = args.tasks
        args.test_tasks = args.tasks
    if args.train_tasks < 1 or args.test_tasks < 1:
        parser.error("training and testing task limits must be at least one")
    if args.max_trace_slots < 1:
        parser.error("--max-trace-slots must be at least one")
    if args.ppo_update_interval_slots < 1:
        parser.error("--ppo-update-interval-slots must be at least one")
    if args.pretrain_cycles < 0 or args.joint_training_cycles < 0:
        parser.error("training cycle counts must be nonnegative")
    if args.pretrain_cycles + args.joint_training_cycles < 1:
        parser.error("at least one training cycle is required")
    if args.output_root is None:
        args.output_root = _timestamped_root()
    if not args.output_root.is_absolute():
        args.output_root = (PROJECT_ROOT / args.output_root).resolve()
    if ELARA_ROOT.resolve() not in args.output_root.parents:
        parser.error("--output-root must be located below ELARA")
    if not args.request_template_file.is_absolute():
        args.request_template_file = (PROJECT_ROOT / args.request_template_file).resolve()
    return args


def ensure_template_catalog(args) -> None:
    if args.request_template_file.exists():
        return
    templates = generate_templates(
        seed=args.template_seed,
        num_services=30,
        chain_plan=DEFAULT_CHAIN_PLAN,
    )
    save_templates(
        args.request_template_file,
        templates,
        metadata={
            "seed": args.template_seed,
            "num_services": 30,
            "chain_plan": DEFAULT_CHAIN_PLAN,
            "purpose": "common catalog for all ELARA sensitivity experiments",
        },
    )


def experiment_specs(args) -> list[dict]:
    specs = []
    for delay_weight, energy_weight in args.weights:
        label = f"d{int(round(delay_weight * 100)):02d}_e{int(round(energy_weight * 100)):02d}"
        specs.append(
            {
                "category": "latency_energy_weights",
                "condition": label,
                "delay_weight": delay_weight,
                "energy_weight": energy_weight,
                "route_max_paths": 3,
            }
        )
    for route_max_paths in args.route_max_paths:
        specs.append(
            {
                "category": "routing_max_paths",
                "condition": f"paths_{route_max_paths}",
                "delay_weight": 0.5,
                "energy_weight": 0.5,
                "route_max_paths": route_max_paths,
            }
        )
    return specs


def _common_arguments(args, spec) -> list[str]:
    return [
        "--max-trace-slots", str(args.max_trace_slots),
        "--request-template-file", str(args.request_template_file),
        "--arrival-lambda", str(args.arrival_lambda),
        "--delay-weight", str(spec["delay_weight"]),
        "--energy-weight", str(spec["energy_weight"]),
        "--route-max-paths", str(spec["route_max_paths"]),
        "--compute-capacity-scale", str(args.compute_capacity_scale),
        "--link-capacity-scale", str(args.link_capacity_scale),
        "--background-load-scale", str(args.background_load_scale),
        "--request-data-scale", str(args.request_data_scale),
    ]


def build_jobs(args, phase: str, accelerator: str, gpu_ids: list[str]) -> list[dict]:
    jobs = []
    for spec in experiment_specs(args):
        for seed in args.seeds:
            base = args.output_root / spec["category"] / spec["condition"] / f"seed_{seed}"
            train_dir = base / "train"
            output_dir = train_dir if phase == "train" else base / "test"
            module = "ELARA.train" if phase == "train" else "ELARA.evaluate"
            command = [
                sys.executable, "-m", module,
                *_common_arguments(args, spec),
                "--seed", str(seed),
                "--output-dir", str(output_dir),
                "--device", accelerator,
            ]
            if phase == "train":
                command.extend(
                    (
                        "--ppo-learning-rate", str(args.ppo_learning_rate),
                        "--ppo-entropy-coef", str(args.ppo_entropy_coef),
                        "--ppo-epochs", str(args.ppo_epochs),
                        "--ppo-minibatch-size", str(args.ppo_minibatch_size),
                        "--rollout-steps", str(args.rollout_steps),
                        "--ppo-update-interval-slots",
                        str(args.ppo_update_interval_slots),
                        "--pretrain-cycles", str(args.pretrain_cycles),
                        "--joint-training-cycles",
                        str(args.joint_training_cycles),
                    )
                )
            else:
                command.extend(
                    (
                        "--policy", "ppo",
                        "--checkpoint", str(train_dir / "ppo_final.pt"),
                        "--full-cycle",
                    )
                )
            index = len(jobs)
            progress_file = (
                args.output_root / "logs" / f"{phase}_{spec['category']}_{spec['condition']}_seed{seed}.progress.json"
            )
            log_file = progress_file.with_suffix(".log")
            command.extend(("--progress-file", str(progress_file)))
            jobs.append(
                {
                    "index": index,
                    "phase": phase,
                    "category": spec["category"],
                    "condition": spec["condition"],
                    "seed": seed,
                    "delay_weight": spec["delay_weight"],
                    "energy_weight": spec["energy_weight"],
                    "route_max_paths": spec["route_max_paths"],
                    "accelerator": accelerator,
                    "gpu": gpu_for_task(index, gpu_ids) if accelerator == "cuda" else None,
                    "output_dir": str(output_dir),
                    "expected_total": (
                        args.max_trace_slots
                        * (args.pretrain_cycles + args.joint_training_cycles)
                        if phase == "train"
                        else args.max_trace_slots
                    ),
                    "progress_file": str(progress_file),
                    "log_file": str(log_file),
                    "command": command,
                    "status": "pending",
                }
            )
    return jobs


def _progress(jobs, started_at: float) -> str:
    completed = 0
    total = 0
    finished = 0
    for job in jobs:
        try:
            state = json.loads(Path(job["progress_file"]).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            state = {}
        completed += int(state.get("completed", 0))
        total += int(state.get("total", job.get("expected_total", 0)))
        if job.get("status") == "succeeded":
            finished += 1
    fraction = completed / max(1, total)
    width = 26
    filled = min(width, int(width * fraction))
    bar = ("=" * filled + (">" if filled < width else "")).ljust(width, "-")
    elapsed = time.monotonic() - started_at
    eta = elapsed * (1.0 - fraction) / fraction if fraction > 0.0 else None
    return (
        f"[{bar}] {100 * fraction:6.2f}% tasks {finished}/{len(jobs)} "
        f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)}"
    )


def run_jobs(args, jobs: list[dict], task_limit: int | None = None) -> bool:
    task_limit = int(task_limit if task_limit is not None else args.tasks)
    if task_limit < 1:
        raise ValueError("task_limit must be at least one")
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        return True
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active = []
    started_at = time.monotonic()
    interactive = sys.stdout.isatty()
    last_render = 0.0
    try:
        while pending or active:
            while pending and len(active) < task_limit:
                job = pending.pop(0)
                if job["phase"] == "test":
                    checkpoint = (
                        Path(job["output_dir"]).parent / "train" / "ppo_final.pt"
                    )
                    if not checkpoint.is_file():
                        job["status"] = "failed"
                        job["return_code"] = 2
                        job["error"] = f"missing checkpoint: {checkpoint}"
                        continue
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                environment["CUDA_VISIBLE_DEVICES"] = job["gpu"] or ""
                if job["accelerator"] == "mps":
                    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
                handle = Path(job["log_file"]).open("w", encoding="utf-8")
                process = subprocess.Popen(
                    job["command"],
                    cwd=PROJECT_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                job["status"] = "running"
                active.append((job, process, handle))
            for job, process, handle in list(active):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                job["return_code"] = return_code
                job["status"] = "succeeded" if return_code == 0 else "failed"
                active.remove((job, process, handle))
            now = time.monotonic()
            if now - last_render >= (0.5 if interactive else 5.0) or not (pending or active):
                line = _progress(jobs, started_at)
                print(f"\r{line}" if interactive else line, end="" if interactive else "\n", flush=True)
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
            job["status"] = "interrupted"
            job["return_code"] = process.returncode
        return False
    return all(job.get("return_code") == 0 for job in jobs)


def write_summary(args, jobs: list[dict]) -> Path | None:
    test_jobs = [job for job in jobs if job["phase"] == "test"]
    rows = []
    for job in test_jobs:
        summary_path = Path(job["output_dir"]) / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "category": job["category"],
                "condition": job["condition"],
                "seed": job["seed"],
                "delay_weight": job["delay_weight"],
                "energy_weight": job["energy_weight"],
                "route_max_paths": job["route_max_paths"],
                "request_template_file": str(args.request_template_file),
                "request_data_scale": args.request_data_scale,
                "background_load_scale": args.background_load_scale,
                "ppo_update_interval_slots": args.ppo_update_interval_slots,
                "pretrain_cycles": args.pretrain_cycles,
                "joint_training_cycles": args.joint_training_cycles,
                "request_count": summary["episodes"],
                "success_rate": summary["success_rate"],
                "mean_return": summary["mean_return"],
                "mean_latency_s": summary["mean_latency_s"],
                "mean_energy_j": summary["mean_energy_j"],
                "mean_route_slot_crossings": summary["mean_route_slot_crossings"],
                "mean_route_phase_count": summary["mean_route_phase_count"],
                "mean_route_augmentation_count": summary[
                    "mean_route_augmentation_count"
                ],
            }
        )
    if not rows:
        return None
    path = args.output_root / "sensitivity_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_template_catalog(args)
    gpu_ids = detect_gpu_ids()
    try:
        accelerator = select_accelerator(args.device, gpu_ids, detect_mps())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_jobs = []
    phases = ("train", "test") if args.phase == "all" else (args.phase,)
    success = True
    for phase in phases:
        jobs = build_jobs(args, phase, accelerator, gpu_ids)
        all_jobs.extend(jobs)
        task_limit = args.train_tasks if phase == "train" else args.test_tasks
        print(
            f"{phase}: {len(jobs)} jobs, concurrency={task_limit}, "
            f"device={accelerator}"
        )
        if not run_jobs(args, jobs, task_limit):
            success = False
            if phase == "train":
                break
    manifest = {
        "request_template_file": str(args.request_template_file),
        "seeds": list(args.seeds),
        "weights": list(args.weights),
        "route_max_paths": list(args.route_max_paths),
        "ppo_update_interval_slots": args.ppo_update_interval_slots,
        "pretrain_cycles": args.pretrain_cycles,
        "joint_training_cycles": args.joint_training_cycles,
        "train_tasks": args.train_tasks,
        "test_tasks": args.test_tasks,
        "accelerator": accelerator,
        "gpu_ids": gpu_ids,
        "jobs": all_jobs,
    }
    (args.output_root / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    summary = write_summary(args, all_jobs)
    print(f"output root: {args.output_root}")
    if summary:
        print(f"summary: {summary}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
