from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

from . import comparison_runner
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
DEFAULT_OUTPUT_PARENT = ELARA_ROOT / "outputs" / "bandit-ablation"
ABLATIONS = ("ELARA", "ELARA-NB")


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
    return DEFAULT_OUTPUT_PARENT / datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired ELARA versus ELARA-NB Bandit ablation using the "
            "040404 models. Both methods use final PPO weights and the same "
            "pre-joint-training placement and Bandit state."
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
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
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
        help="PPO weights used by both ELARA and ELARA-NB",
    )
    parser.add_argument(
        "--initial-control-checkpoint-name",
        default="ppo_pretrained.pt",
        help=(
            "shared pre-joint-training placement and Bandit state used to "
            "initialize both methods"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.background_seeds is None:
        args.background_seeds = [
            seed + 100_000 for seed in args.test_seeds
        ]
    if len(args.model_seeds) != len(args.test_seeds):
        parser.error("--model-seeds and --test-seeds must have equal lengths")
    if len(args.background_seeds) != len(args.model_seeds):
        parser.error("--background-seeds must match the model-seed count")
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    if args.max_slots < 1:
        parser.error("--max-slots must be positive")
    if any(length not in {5, 10, 15} for length in args.chain_lengths):
        parser.error("--chain-lengths may only contain 5, 10, and 15")
    args.model_root = args.model_root.expanduser().resolve()
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _timestamped_output_root().resolve()
    )
    args.request_template_file = (
        args.request_template_file.expanduser().resolve()
    )
    return args


def _joined(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _comparison_argv(args) -> list[str]:
    forwarded = [
        "--baselines",
        ",".join(ABLATIONS),
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


def _mean_std_ci95(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": 1.96 * std / math.sqrt(len(values)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_ablation(output_root: Path) -> None:
    source = output_root / "comparison_summary.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    scenarios: dict[tuple[int, int, int, int], dict[str, dict]] = {}
    for record in records:
        key = (
            int(record["model_seed"]),
            int(record["test_seed"]),
            int(record["background_seed"]),
            int(record["chain_length"]),
        )
        scenarios.setdefault(key, {})[record["ablation"]] = record

    paired = []
    for key, methods in sorted(scenarios.items()):
        if set(methods) != set(ABLATIONS):
            raise RuntimeError(
                f"incomplete Bandit ablation scenario {key}: {set(methods)}"
            )
        elara = methods["ELARA"]
        no_bandit = methods["ELARA-NB"]
        if (
            elara["request_stream_hash"]
            != no_bandit["request_stream_hash"]
        ):
            raise RuntimeError(
                f"request streams differ in Bandit ablation scenario {key}"
            )
        if (
            elara["initial_control_state_hash"]
            != no_bandit["initial_control_state_hash"]
        ):
            raise RuntimeError(
                f"initial control states differ in scenario {key}"
            )
        if elara["policy_checkpoint"] != no_bandit["policy_checkpoint"]:
            raise RuntimeError(
                f"PPO policy checkpoints differ in scenario {key}"
            )
        if (
            elara["control_state_checkpoint"]
            != no_bandit["control_state_checkpoint"]
        ):
            raise RuntimeError(
                f"control-state checkpoints differ in scenario {key}"
            )
        if elara["route_strategy"] != no_bandit["route_strategy"]:
            raise RuntimeError(
                f"routing strategies differ in scenario {key}"
            )
        if str(elara["adaptation_enabled"]).lower() != "true":
            raise RuntimeError(f"ELARA adaptation is disabled in {key}")
        if str(no_bandit["adaptation_enabled"]).lower() != "false":
            raise RuntimeError(f"ELARA-NB adaptation is enabled in {key}")
        no_bandit_migrations = sum(
            int(no_bandit[f"migration_{action}_count"])
            for action in ("no_op", "relocate", "scale_out", "scale_in")
        )
        if no_bandit_migrations:
            raise RuntimeError(
                f"ELARA-NB performed {no_bandit_migrations} migrations in {key}"
            )

        def reduction(metric: str) -> float:
            reference = float(no_bandit[metric])
            return (
                (reference - float(elara[metric]))
                / max(abs(reference), 1.0e-12)
                * 100.0
            )

        row = {
            "model_seed": key[0],
            "test_seed": key[1],
            "background_seed": key[2],
            "chain_length": key[3],
            "request_count": int(elara["request_count"]),
            "request_stream_hash": elara["request_stream_hash"],
            "initial_control_state_hash": elara[
                "initial_control_state_hash"
            ],
            "success_rate_delta_pp": (
                float(elara["success_rate"])
                - float(no_bandit["success_rate"])
            )
            * 100.0,
            "return_delta": (
                float(elara["mean_return"])
                - float(no_bandit["mean_return"])
            ),
            "latency_reduction_pct": reduction("mean_latency_s"),
            "p95_latency_reduction_pct": reduction("p95_latency_s"),
            "energy_reduction_pct": reduction("mean_energy_j"),
            "elara_return_win": int(
                float(elara["mean_return"])
                > float(no_bandit["mean_return"])
            ),
        }
        paired.append(row)

    _write_csv(output_root / "bandit_ablation_paired.csv", paired)
    metrics = (
        "success_rate_delta_pp",
        "return_delta",
        "latency_reduction_pct",
        "p95_latency_reduction_pct",
        "energy_reduction_pct",
    )

    def summarize(rows: list[dict]) -> dict:
        result = {
            "scenario_count": len(rows),
            "request_count": sum(int(row["request_count"]) for row in rows),
            "elara_return_wins": sum(
                int(row["elara_return_win"]) for row in rows
            ),
        }
        for metric in metrics:
            result[metric] = _mean_std_ci95(
                [float(row[metric]) for row in rows]
            )
        return result

    aggregate = {
        "experiment": "ELARA versus ELARA-NB",
        "policy_checkpoint": records[0]["policy_checkpoint"],
        "shared_initial_control_checkpoint": records[0][
            "control_state_checkpoint"
        ],
        "shared_request_stream_verified": True,
        "shared_initial_control_state_verified": True,
        "shared_routing_strategy_verified": True,
        "elara_nb_zero_migrations_verified": True,
        "positive_delta_or_reduction_means_elara_is_better": True,
        "overall": summarize(paired),
        "by_chain_length": {
            str(chain_length): summarize(
                [
                    row
                    for row in paired
                    if int(row["chain_length"]) == chain_length
                ]
            )
            for chain_length in sorted(
                {int(row["chain_length"]) for row in paired}
            )
        },
    }
    (output_root / "bandit_ablation_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )


def _annotate_manifest(args) -> None:
    path = args.output_root / "comparison_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "paired_bandit_ablation",
            "ablation_methods": list(ABLATIONS),
            "shared_policy_checkpoint_name": (
                args.policy_checkpoint_name
            ),
            "shared_initial_control_checkpoint_name": (
                args.initial_control_checkpoint_name
            ),
            "shared_initial_state_required": True,
            "interpretation": (
                "ELARA and ELARA-NB use identical PPO weights and identical "
                "pre-joint-training placement and Bandit state. Only ELARA "
                "continues replica adaptation during testing."
            ),
        }
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = comparison_runner.main(_comparison_argv(args))
    _annotate_manifest(args)
    if result == 0 and not args.dry_run:
        try:
            _aggregate_ablation(args.output_root)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            manifest_path = (
                args.output_root / "comparison_manifest.json"
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["status"] = "failed"
            manifest["bandit_ablation_aggregation_error"] = str(exc)
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            print(
                f"Bandit ablation aggregation failed: {exc}",
                file=sys.stderr,
            )
            return 1
        print(f"Bandit ablation complete: {args.output_root}")
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
