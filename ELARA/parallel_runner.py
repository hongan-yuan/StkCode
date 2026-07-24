from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def detect_gpu_ids() -> list[str]:
    """Return CUDA device identifiers visible to this launcher."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        return [] if devices == ["-1"] else devices

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if devices:
                return devices
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def detect_mps() -> bool:
    try:
        import torch

        from .device import mps_is_available

        return mps_is_available(torch)
    except Exception:
        return False


def select_accelerator(requested: str, gpu_ids: list[str], mps_available: bool) -> str:
    if requested == "auto":
        if gpu_ids:
            return "cuda"
        return "mps" if mps_available else "cpu"
    if requested == "cuda" and not gpu_ids:
        raise RuntimeError("CUDA was requested, but no visible CUDA GPU was detected")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but MPS is not available")
    return requested


def gpu_for_task(task_index: int, gpu_ids: list[str]) -> str | None:
    if not gpu_ids:
        return None
    return gpu_ids[task_index % len(gpu_ids)]


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_line(jobs: list[dict], width: int = 28) -> str:
    completed = 0
    total = 0
    etas = []
    elapsed = []
    finished_tasks = 0
    units = []
    phases = []
    item_count = 0
    for job in jobs:
        state = {}
        try:
            state = json.loads(Path(job["progress_file"]).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
        completed += int(state.get("completed", 0))
        total += int(state.get("total", job.get("total", 0)))
        units.append(str(state.get("unit", job.get("unit", "episodes"))))
        phases.append(str(state.get("phase", "")))
        if state.get("item_count") is not None:
            item_count += int(state["item_count"])
        if state.get("eta_s") is not None:
            etas.append(float(state["eta_s"]))
        if state.get("elapsed_s") is not None:
            elapsed.append(float(state["elapsed_s"]))
        if job.get("return_code") == 0:
            finished_tasks += 1
    fraction = completed / max(1, total)
    filled = min(width, int(fraction * width))
    bar = "=" * filled + (">" if filled < width else "")
    bar = bar.ljust(width, "-")
    eta = max(etas) if etas else None
    unit = units[0] if units and len(set(units)) == 1 else "items"
    item_text = f" | requests {item_count}" if unit == "slots" else ""
    updating_count = sum(phase == "updating PPO" for phase in phases)
    phase_text = (
        f" | PPO updating {updating_count}/{len(jobs)}"
        if updating_count else ""
    )
    return (
        f"[{bar}] {fraction * 100:6.2f}% "
        f"{unit} {completed}/{total or '?'}{item_text} | "
        f"tasks {finished_tasks}/{len(jobs)}{phase_text} | "
        f"elapsed {format_duration(max(elapsed) if elapsed else 0.0)} | "
        f"ETA {format_duration(eta)}"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Launch independent ELARA jobs concurrently and distribute them "
            "round-robin over visible CUDA GPUs or a shared MPS device. Unknown arguments are forwarded "
            "to ELARA.train or ELARA.evaluate."
        )
    )
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument(
        "--tasks",
        type=int,
        help="number of concurrent jobs; defaults to 2 for training and 4 for evaluation",
    )
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="accelerator selection; auto prefers CUDA, then MPS, then CPU",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.tasks is None:
        args.tasks = 2 if args.mode == "train" else 4
    if args.tasks < 1:
        parser.error("--tasks must be at least 1")
    for reserved in ("--seed", "--output-dir"):
        if reserved in forwarded:
            parser.error(
                f"{reserved} is managed per task; use --base-seed or --output-root instead"
            )
    if args.mode == "train" and any(
        item == "--episodes" or item.startswith("--episodes=")
        for item in forwarded
    ):
        parser.error(
            "training request count is generated by the Poisson process over the configured constellation cycles; "
            "do not pass --episodes"
        )
    return args, forwarded


def _default_output_root(mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("ELARA") / "outputs" / f"parallel-{mode}" / timestamp


def _forwarded_int(arguments: list[str], option: str, default: int) -> int:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return int(arguments[index + 1])
        if argument.startswith(f"{option}="):
            return int(argument.split("=", 1)[1])
    return default


def _job_specs(
    args, forwarded: list[str], gpu_ids: list[str], accelerator: str
) -> list[dict]:
    base_seed = args.base_seed
    if base_seed is None:
        base_seed = 42 if args.mode == "train" else 1000
    output_root = args.output_root or _default_output_root(args.mode)
    module = "ELARA.train" if args.mode == "train" else "ELARA.evaluate"
    if args.mode == "train":
        cycle_slots = _forwarded_int(forwarded, "--max-trace-slots", 606)
        pretrain_cycles = _forwarded_int(forwarded, "--pretrain-cycles", 1)
        joint_cycles = _forwarded_int(
            forwarded, "--joint-training-cycles", 1
        )
        total = cycle_slots * (pretrain_cycles + joint_cycles)
        unit = "slots"
    else:
        total = _forwarded_int(forwarded, "--episodes", 100)
        unit = "episodes"
    jobs = []
    for task_index in range(args.tasks):
        seed = base_seed + task_index
        task_name = f"task-{task_index:02d}-seed-{seed}"
        output_dir = output_root / task_name
        gpu = gpu_for_task(task_index, gpu_ids) if accelerator == "cuda" else None
        command = [
            args.python,
            "-m",
            module,
            *forwarded,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--device",
            accelerator,
        ]
        jobs.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "seed": seed,
                "total": total,
                "unit": unit,
                "accelerator": accelerator,
                "gpu": gpu,
                "output_dir": str(output_dir),
                "log_file": str(output_root / "logs" / f"{task_name}.log"),
                "progress_file": str(output_root / "logs" / f"{task_name}.progress.json"),
                "command": command,
            }
        )
        command.extend(("--progress-file", jobs[-1]["progress_file"]))
    return jobs


def main(argv: list[str] | None = None) -> int:
    args, forwarded = parse_args(argv)
    gpu_ids = detect_gpu_ids()
    mps_available = detect_mps()
    try:
        accelerator = select_accelerator(args.device, gpu_ids, mps_available)
    except RuntimeError as error:
        print(f"Device error: {error}", file=sys.stderr)
        return 2
    jobs = _job_specs(args, forwarded, gpu_ids, accelerator)
    cuda_summary = ", ".join(gpu_ids) if gpu_ids else "none"
    print(f"Detected CUDA GPUs: {cuda_summary}; MPS: {'available' if mps_available else 'unavailable'}")
    print(f"Selected accelerator: {accelerator}")
    print(f"Launching {len(jobs)} concurrent {args.mode} jobs")
    for job in jobs:
        if job["gpu"] is not None:
            assigned = f"CUDA GPU {job['gpu']}"
        else:
            assigned = accelerator.upper()
        print(f"  {job['task_name']} -> {assigned}: {shlex.join(job['command'])}")
    if args.dry_run:
        return 0

    output_root = Path(jobs[0]["output_dir"]).parent
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "launcher_manifest.json"
    manifest = {
        "mode": args.mode,
        "gpu_ids": gpu_ids,
        "mps_available": mps_available,
        "accelerator": accelerator,
        "task_count": len(jobs),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": jobs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    processes = []
    log_handles = []
    try:
        for job in jobs:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["ELARA_TASK_INDEX"] = str(job["task_index"])
            environment["CUDA_VISIBLE_DEVICES"] = job["gpu"] or ""
            if job["accelerator"] == "mps":
                environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            log_handle = Path(job["log_file"]).open("w", encoding="utf-8")
            process = subprocess.Popen(
                job["command"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append((job, process))
            log_handles.append(log_handle)

        unfinished = list(processes)
        interactive = sys.stdout.isatty()
        last_render = 0.0
        while unfinished:
            for job, process in list(unfinished):
                return_code = process.poll()
                if return_code is not None:
                    job["return_code"] = return_code
                    job["status"] = "succeeded" if return_code == 0 else "failed"
                    unfinished.remove((job, process))
            now = time.monotonic()
            if now - last_render >= (0.5 if interactive else 5.0) or not unfinished:
                line = progress_line(jobs)
                if interactive:
                    print(f"\r{line}", end="", flush=True)
                else:
                    print(line, flush=True)
                last_render = now
            if unfinished:
                time.sleep(0.2)
        if interactive:
            print()
    except KeyboardInterrupt:
        print("Interrupted; terminating unfinished jobs...", file=sys.stderr)
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for job, process in processes:
            if process.poll() is None:
                process.wait()
            job["return_code"] = process.returncode
            job["status"] = "interrupted"
    finally:
        for handle in log_handles:
            handle.close()
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    failed = [job for job in jobs if job.get("return_code") != 0]
    print(f"Manifest: {manifest_path}")
    print(f"Completed: {len(jobs) - len(failed)}, failed: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
