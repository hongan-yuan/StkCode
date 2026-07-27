from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from . import comparison_runner
from .comparison_policies import BASELINES
from .config import PROJECT_ROOT
from .request_templates import DEFAULT_TEMPLATE_PATH


ELARA_ROOT = PROJECT_ROOT / "ELARA"
DEFAULT_MODEL_ROOT = (
    ELARA_ROOT
    / "outputs"
    / "sensitivity"
    / "20260725-040404"
    / "latency_energy_weights"
    / "d50_e50"
)
DEFAULT_OUTPUT_PARENT = ELARA_ROOT / "outputs" / "baseline-tests4"


def _split_ints(value: str) -> list[int]:
    try:
        values = [
            int(item)
            for item in value.replace(",", " ").split()
            if item
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return list(dict.fromkeys(values))


def _timestamped_output_root() -> Path:
    return DEFAULT_OUTPUT_PARENT / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run all seven ELARA baselines with shared 0.5:0.5 PPO weights, "
            "initial placement, request streams, and Markov-background seeds."
        )
    )
    parser.add_argument(
        "--model-seeds", type=_split_ints, default=[42, 43, 44, 45]
    )
    parser.add_argument(
        "--test-seeds", type=_split_ints, default=[142, 143, 144, 145]
    )
    parser.add_argument("--background-seeds", type=_split_ints)
    parser.add_argument(
        "--chain-lengths", type=_split_ints, default=[5, 10, 15]
    )
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--request-template-file",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
    )
    parser.add_argument("--total-arrival-lambda", type=float)
    parser.add_argument("--max-slots", type=int, default=606)
    parser.add_argument(
        "--policy-checkpoint-name",
        default="ppo_final.pt",
        help="shared 0.5:0.5 PPO weights used within every test scenario",
    )
    parser.add_argument(
        "--initial-control-checkpoint-name",
        default="ppo_pretrained.pt",
        help=(
            "checkpoint supplying the common pre-adaptation placement and "
            "Bandit state"
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
        parser.error("test seeds must differ from model seeds")
    if set(args.model_seeds) & set(args.background_seeds):
        parser.error("background seeds must differ from model seeds")
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    if args.max_slots < 1:
        parser.error("--max-slots must be positive")
    if any(length not in {5, 10, 15} for length in args.chain_lengths):
        parser.error("--chain-lengths may only contain 5, 10, and 15")
    if (
        args.total_arrival_lambda is not None
        and args.total_arrival_lambda <= 0.0
    ):
        parser.error("--total-arrival-lambda must be positive")

    args.model_root = args.model_root.expanduser().resolve()
    args.request_template_file = (
        args.request_template_file.expanduser().resolve()
    )
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _timestamped_output_root().resolve()
    )
    if ELARA_ROOT.resolve() not in args.output_root.parents:
        parser.error("--output-root must be below the ELARA directory")
    return args


def _joined(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _comparison_argv(args) -> list[str]:
    forwarded = [
        "--baselines",
        ",".join(BASELINES),
        "--model-seeds",
        _joined(args.model_seeds),
        "--test-seeds",
        _joined(args.test_seeds),
        "--background-seeds",
        _joined(args.background_seeds),
        "--chain-lengths",
        _joined(args.chain_lengths),
        "--tasks",
        str(args.tasks),
        "--device",
        args.device,
        "--model-root",
        str(args.model_root),
        "--output-root",
        str(args.output_root),
        "--request-template-file",
        str(args.request_template_file),
        "--max-slots",
        str(args.max_slots),
        "--checkpoint-name",
        args.policy_checkpoint_name,
        "--control-state-checkpoint-name",
        args.initial_control_checkpoint_name,
    ]
    if args.total_arrival_lambda is not None:
        forwarded.extend(
            ["--total-arrival-lambda", str(args.total_arrival_lambda)]
        )
    if args.dry_run:
        forwarded.append("--dry-run")
    return forwarded


def _read_comparison_rows(output_root: Path) -> list[dict[str, str]]:
    path = output_root / "comparison_summary.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_fairness(args) -> dict:
    rows = _read_comparison_rows(args.output_root)
    grouped: dict[tuple[int, int, int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            int(row["model_seed"]),
            int(row["test_seed"]),
            int(row["background_seed"]),
            int(row["chain_length"]),
        )
        grouped.setdefault(key, []).append(row)

    expected_scenarios = (
        len(args.model_seeds) * len(args.chain_lengths)
    )
    if len(grouped) != expected_scenarios:
        raise RuntimeError(
            f"expected {expected_scenarios} scenarios, found {len(grouped)}"
        )

    verified_scenarios = []
    for key, scenario_rows in sorted(grouped.items()):
        methods = {row["ablation"] for row in scenario_rows}
        if methods != set(BASELINES):
            raise RuntimeError(
                f"scenario {key} has incomplete baselines: {sorted(methods)}"
            )

        checks = {
            "request_stream_hash": {
                row["request_stream_hash"] for row in scenario_rows
            },
            "request_count": {
                row["request_count"] for row in scenario_rows
            },
            "initial_control_state_hash": {
                row["initial_control_state_hash"] for row in scenario_rows
            },
            "initial_placement_hash": {
                row["initial_placement_hash"] for row in scenario_rows
            },
            "policy_checkpoint": {
                row["policy_checkpoint"] for row in scenario_rows
            },
            "control_state_checkpoint": {
                row["control_state_checkpoint"] for row in scenario_rows
            },
            "delay_weight": {
                float(row["delay_weight"]) for row in scenario_rows
            },
            "energy_weight": {
                float(row["energy_weight"]) for row in scenario_rows
            },
            "route_max_paths_per_slot": {
                int(row["route_max_paths_per_slot"])
                for row in scenario_rows
            },
        }
        inconsistent = [
            name for name, values in checks.items() if len(values) != 1
        ]
        if inconsistent:
            raise RuntimeError(
                f"scenario {key} does not share {', '.join(inconsistent)}"
            )
        if checks["delay_weight"] != {0.5}:
            raise RuntimeError(f"scenario {key} is not delay weight 0.5")
        if checks["energy_weight"] != {0.5}:
            raise RuntimeError(f"scenario {key} is not energy weight 0.5")
        if checks["route_max_paths_per_slot"] != {3}:
            raise RuntimeError(f"scenario {key} does not use paths=3")

        policy_path = Path(next(iter(checks["policy_checkpoint"])))
        control_path = Path(
            next(iter(checks["control_state_checkpoint"]))
        )
        if policy_path.name != args.policy_checkpoint_name:
            raise RuntimeError(
                f"scenario {key} used unexpected policy checkpoint"
            )
        if control_path.name != args.initial_control_checkpoint_name:
            raise RuntimeError(
                f"scenario {key} used unexpected control checkpoint"
            )
        verified_scenarios.append(
            {
                "model_seed": key[0],
                "test_seed": key[1],
                "background_seed": key[2],
                "chain_length": key[3],
                "request_stream_hash": next(
                    iter(checks["request_stream_hash"])
                ),
                "initial_control_state_hash": next(
                    iter(checks["initial_control_state_hash"])
                ),
                "initial_placement_hash": next(
                    iter(checks["initial_placement_hash"])
                ),
                "policy_checkpoint": str(policy_path),
                "control_state_checkpoint": str(control_path),
            }
        )

    report = {
        "status": "verified",
        "experiment": "shared_state_seven_baseline_comparison",
        "baselines": list(BASELINES),
        "scenario_count": len(verified_scenarios),
        "shared_within_each_scenario": {
            "ppo_policy_checkpoint": True,
            "initial_control_state": True,
            "initial_placement": True,
            "request_stream": True,
            "test_seed": True,
            "background_seed": True,
            "latency_energy_weights": "0.5:0.5",
            "route_max_paths_per_slot": 3,
        },
        "policy_checkpoint_name": args.policy_checkpoint_name,
        "initial_control_checkpoint_name": (
            args.initial_control_checkpoint_name
        ),
        "scenarios": verified_scenarios,
    }
    path = args.output_root / "fairness_verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _annotate_manifest(args, verification: dict | None = None) -> None:
    path = args.output_root / "comparison_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "shared_state_seven_baseline_comparison",
            "shared_policy_checkpoint_name": (
                args.policy_checkpoint_name
            ),
            "shared_initial_control_checkpoint_name": (
                args.initial_control_checkpoint_name
            ),
            "shared_initial_placement_required": True,
            "shared_request_and_background_seeds_required": True,
            "fairness_verification": (
                str(args.output_root / "fairness_verification.json")
                if verification is not None
                else None
            ),
        }
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = comparison_runner.main(_comparison_argv(args))
    if result != 0:
        _annotate_manifest(args)
        return result
    if args.dry_run:
        _annotate_manifest(args)
        return 0
    try:
        verification = verify_fairness(args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        manifest_path = args.output_root / "comparison_manifest.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["status"] = "failed"
        manifest["fairness_verification_error"] = str(exc)
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Fairness verification failed: {exc}", file=sys.stderr)
        return 1
    _annotate_manifest(args, verification)
    print(
        "Shared-state seven-baseline comparison verified: "
        f"{args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
