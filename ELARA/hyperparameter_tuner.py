from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .parallel_runner import detect_gpu_ids, detect_mps, gpu_for_task, select_accelerator
from .sensitivity_runner import (
    DEFAULT_TEMPLATE_FILE,
    ELARA_ROOT,
    PROJECT_ROOT,
    _split_ints,
    ensure_template_catalog,
    run_jobs,
)


DEFAULT_SEARCH = ELARA_ROOT / "configs" / "hyperparameter_search.json"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Tune ELARA on validation seeds and export a frozen profile for "
            "final sensitivity experiments."
        )
    )
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--validation-seeds", type=_split_ints, default=(202, 203))
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--request-template-file", type=Path, default=DEFAULT_TEMPLATE_FILE)
    parser.add_argument("--template-seed", type=int, default=2026)
    parser.add_argument("--max-trace-slots", type=int, default=606)
    parser.add_argument("--arrival-lambda", type=float, default=0.35)
    parser.add_argument("--delay-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=0.5)
    parser.add_argument("--route-max-paths", type=int, default=3)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument(
        "--minimum-success-rate",
        type=float,
        default=0.95,
        help="profiles below this validation success rate cannot be selected",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.tasks < 1:
        parser.error("--tasks must be at least one")
    if abs(args.delay_weight + args.energy_weight - 1.0) > 1.0e-9:
        parser.error("--delay-weight and --energy-weight must sum to one")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("--minimum-success-rate must be in [0, 1]")
    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_root = ELARA_ROOT / "outputs" / "tuning" / stamp
    elif not args.output_root.is_absolute():
        args.output_root = (PROJECT_ROOT / args.output_root).resolve()
    if not args.search_config.is_absolute():
        args.search_config = (PROJECT_ROOT / args.search_config).resolve()
    if not args.request_template_file.is_absolute():
        args.request_template_file = (PROJECT_ROOT / args.request_template_file).resolve()
    return args


def load_profiles(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", [])
    if not profiles:
        raise ValueError("search configuration has no profiles")
    required = {
        "name", "compute_capacity_scale", "link_capacity_scale",
        "background_load_scale", "request_data_scale", "ppo_learning_rate",
        "ppo_entropy_coef", "ppo_epochs", "ppo_minibatch_size", "rollout_steps",
    }
    for profile in profiles:
        missing = required - set(profile)
        if missing:
            raise ValueError(
                f"profile {profile.get('name', '<unnamed>')} is missing {sorted(missing)}"
            )
    return payload.get("selection_protocol", {}), profiles


def _common(args, profile):
    return [
        "--max-trace-slots", str(args.max_trace_slots),
        "--request-template-file", str(args.request_template_file),
        "--arrival-lambda", str(args.arrival_lambda),
        "--delay-weight", str(args.delay_weight),
        "--energy-weight", str(args.energy_weight),
        "--route-max-paths", str(args.route_max_paths),
        "--compute-capacity-scale", str(profile["compute_capacity_scale"]),
        "--link-capacity-scale", str(profile["link_capacity_scale"]),
        "--background-load-scale", str(profile["background_load_scale"]),
        "--request-data-scale", str(profile["request_data_scale"]),
    ]


def build_jobs(args, profiles, phase, accelerator, gpu_ids):
    jobs = []
    for profile in profiles:
        for seed in args.validation_seeds:
            base = args.output_root / profile["name"] / f"seed_{seed}"
            output_dir = base / phase
            command = [
                sys.executable, "-m",
                "ELARA.train" if phase == "train" else "ELARA.evaluate",
                *_common(args, profile),
                "--seed", str(seed),
                "--output-dir", str(output_dir),
                "--device", accelerator,
            ]
            if phase == "train":
                command.extend(
                    (
                        "--ppo-learning-rate", str(profile["ppo_learning_rate"]),
                        "--ppo-entropy-coef", str(profile["ppo_entropy_coef"]),
                        "--ppo-epochs", str(profile["ppo_epochs"]),
                        "--ppo-minibatch-size", str(profile["ppo_minibatch_size"]),
                        "--rollout-steps", str(profile["rollout_steps"]),
                    )
                )
            else:
                command.extend(
                    (
                        "--policy", "ppo",
                        "--checkpoint", str(base / "train" / "ppo_final.pt"),
                        "--full-cycle",
                    )
                )
            index = len(jobs)
            progress = args.output_root / "logs" / f"{phase}_{profile['name']}_seed{seed}.progress.json"
            command.extend(("--progress-file", str(progress)))
            jobs.append(
                {
                    "index": index,
                    "phase": phase,
                    "category": "hyperparameter_tuning",
                    "condition": profile["name"],
                    "profile": profile,
                    "seed": seed,
                    "accelerator": accelerator,
                    "gpu": gpu_for_task(index, gpu_ids) if accelerator == "cuda" else None,
                    "output_dir": str(output_dir),
                    "expected_total": args.max_trace_slots,
                    "progress_file": str(progress),
                    "log_file": str(progress.with_suffix(".log")),
                    "command": command,
                    "status": "pending",
                }
            )
    return jobs


def aggregate(args, protocol, profiles, jobs):
    by_profile = {profile["name"]: [] for profile in profiles}
    for job in jobs:
        if job["phase"] != "test":
            continue
        path = Path(job["output_dir"]) / "summary.json"
        if path.is_file():
            by_profile[job["condition"]].append(
                json.loads(path.read_text(encoding="utf-8"))
            )
    valid = {name: rows for name, rows in by_profile.items() if rows}
    if not valid:
        return None
    raw_latency = [
        row["mean_latency_s"] for rows in valid.values() for row in rows
        if row["mean_latency_s"] is not None
    ]
    raw_energy = [
        row["mean_energy_j"] for rows in valid.values() for row in rows
        if row["mean_energy_j"] is not None
    ]
    latency_reference = float(np.median(raw_latency)) if raw_latency else 1.0
    energy_reference = float(np.median(raw_energy)) if raw_energy else 1.0
    records = []
    profile_by_name = {profile["name"]: profile for profile in profiles}
    for name, rows in valid.items():
        latency_values = [
            row["mean_latency_s"] for row in rows if row["mean_latency_s"] is not None
        ]
        energy_values = [
            row["mean_energy_j"] for row in rows if row["mean_energy_j"] is not None
        ]
        latency = float(np.mean(latency_values)) if latency_values else math.inf
        energy = float(np.mean(energy_values)) if energy_values else math.inf
        success = float(np.mean([row["success_rate"] for row in rows]))
        selectable = (
            math.isfinite(latency)
            and math.isfinite(energy)
            and success >= args.minimum_success_rate
        )
        score = (
            args.delay_weight * latency / max(latency_reference, 1.0e-9)
            + args.energy_weight * energy / max(energy_reference, 1.0e-9)
            + args.failure_penalty * (1.0 - success)
            if selectable else math.inf
        )
        records.append(
            {
                "profile": name,
                "validation_seed_count": len(rows),
                "mean_latency_s": latency,
                "mean_energy_j": energy,
                "success_rate": success,
                "selectable": int(selectable),
                "selection_score": score,
            }
        )
    records.sort(key=lambda row: row["selection_score"])
    results_path = args.output_root / "tuning_results.csv"
    with results_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    selectable_records = [row for row in records if row["selectable"]]
    if not selectable_records:
        return results_path, None
    winner = selectable_records[0]
    best_path = args.output_root / "best_profile.json"
    best_path.write_text(
        json.dumps(
            {
                "selected_profile": winner["profile"],
                "profile": profile_by_name[winner["profile"]],
                "validation_metrics": winner,
                "validation_seeds": list(args.validation_seeds),
                "request_template_file": str(args.request_template_file),
                "selection_protocol": protocol,
                "selection_score": (
                    "weighted latency and energy normalized by validation medians, "
                    "plus failure penalty"
                ),
                "warning": (
                    "Freeze this profile before running final seeds. Apply scenario "
                    "parameters to every compared method."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results_path, best_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol, profiles = load_profiles(args.search_config)
    ensure_template_catalog(args)
    gpu_ids = detect_gpu_ids()
    accelerator = select_accelerator(args.device, gpu_ids, detect_mps())
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_jobs = []
    for phase in ("train", "test"):
        jobs = build_jobs(args, profiles, phase, accelerator, gpu_ids)
        all_jobs.extend(jobs)
        print(f"{phase}: {len(jobs)} validation jobs")
        if not run_jobs(args, jobs):
            return 1
    if args.dry_run:
        return 0
    outputs = aggregate(args, protocol, profiles, all_jobs)
    (args.output_root / "tuning_manifest.json").write_text(
        json.dumps(
            {
                "search_config": str(args.search_config),
                "request_template_file": str(args.request_template_file),
                "validation_seeds": list(args.validation_seeds),
                "jobs": all_jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if outputs:
        print(f"tuning results: {outputs[0]}")
        if outputs[1] is not None:
            print(f"best frozen profile: {outputs[1]}")
        else:
            print(
                "no profile met the minimum validation success rate; "
                "best_profile.json was not written",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
