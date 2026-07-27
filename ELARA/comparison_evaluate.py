from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np

from .comparison_policies import (
    ADAPTATION_BASELINES,
    BASELINES,
    BASELINE_DESCRIPTIONS,
    ROUTE_STRATEGIES,
    BaselineServingPolicy,
)
from .config import ELARAConfig, PROJECT_ROOT
from .environment import ELARAEnvironment
from .progress import ProgressReporter


ELARA_ROOT = PROJECT_ROOT / "ELARA"
DEFAULT_TEMPLATE_FILE = ELARA_ROOT / "data" / "request_templates_seed2026.json"
DEFAULT_TRACE_FILE = (
    PROJECT_ROOT
    / "WalkerDeltaConstellationSimu"
    / "Walker_Delta_ISL_Simu.csv"
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one ELARA baseline on an independent, reproducible "
            "request-arrival and Markov-background scenario."
        )
    )
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--background-seed", type=int, required=True)
    parser.add_argument("--chain-length", type=int, choices=(5, 10, 15), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="ppo_final.pt")
    parser.add_argument(
        "--control-state-checkpoint-name",
        help=(
            "checkpoint whose placement and Bandit state initialize the "
            "environment; defaults to --checkpoint-name"
        ),
    )
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--request-template-file", type=Path)
    parser.add_argument("--total-arrival-lambda", type=float, default=4.9)
    parser.add_argument("--max-slots", type=int, default=606)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args(argv)
    if args.total_arrival_lambda <= 0.0:
        parser.error("--total-arrival-lambda must be positive")
    if args.max_slots < 1:
        parser.error("--max-slots must be positive")
    return args


def _tuple_value(value):
    if isinstance(value, list):
        return tuple(_tuple_value(item) for item in value)
    return value


def _local_path(value, fallback: Path) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"required experiment input is missing: {fallback}")


def load_training_config(path: Path) -> ELARAConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    valid = {item.name for item in fields(ELARAConfig)}
    tuple_names = {
        "replica_count_range",
        "request_template_chain_lengths",
        "compute_capacity_choices_gflops",
        "compute_load_states",
        "link_load_states",
    }
    kwargs = {}
    for key, value in raw.items():
        if key not in valid:
            continue
        if key in tuple_names:
            kwargs[key] = _tuple_value(value)
        elif key == "compute_power_by_capacity_w":
            kwargs[key] = {
                float(capacity): float(power)
                for capacity, power in value.items()
            }
        else:
            kwargs[key] = value
    kwargs["trace_csv"] = _local_path(
        raw.get("trace_csv"), DEFAULT_TRACE_FILE
    )
    kwargs["request_template_file"] = _local_path(
        raw.get("request_template_file"), DEFAULT_TEMPLATE_FILE
    )
    kwargs["output_dir"] = ELARA_ROOT / "outputs"
    return ELARAConfig(**kwargs)


def build_evaluation_config(args) -> ELARAConfig:
    config_path = args.model_dir / args.config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"training config is missing: {config_path}")
    config = load_training_config(config_path)
    if config.seed != args.model_seed:
        raise ValueError(
            f"model seed {args.model_seed} does not match config seed {config.seed}"
        )
    if (
        abs(config.delay_weight - 0.5) > 1.0e-9
        or abs(config.energy_weight - 0.5) > 1.0e-9
        or config.route_max_paths_per_slot != 3
    ):
        raise ValueError(
            "comparison requires a 0.5:0.5 checkpoint trained with paths=3"
        )
    config.request_seed = args.test_seed
    config.background_seed = args.background_seed
    config.request_chain_length_filter = args.chain_length
    config.request_arrival_lambda_total_per_slot = args.total_arrival_lambda
    config.max_trace_slots = args.max_slots
    config.route_strategy = ROUTE_STRATEGIES[args.baseline]
    config.adaptation_enabled = args.baseline in ADAPTATION_BASELINES
    config.output_dir = args.output_dir
    if args.request_template_file is not None:
        config.request_template_file = args.request_template_file.resolve()
    config.__post_init__()
    return config


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _finite_percentile(values, percentile: float) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.percentile(finite, percentile)) if finite else None


def _state_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args) -> dict:
    config = build_evaluation_config(args)
    policy_checkpoint = args.model_dir / args.checkpoint_name
    control_state_checkpoint = args.model_dir / (
        args.control_state_checkpoint_name or args.checkpoint_name
    )
    if not policy_checkpoint.is_file():
        raise FileNotFoundError(
            f"trained policy checkpoint is missing: {policy_checkpoint}"
        )
    if not control_state_checkpoint.is_file():
        raise FileNotFoundError(
            "initial control-state checkpoint is missing: "
            f"{control_state_checkpoint}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = ELARAEnvironment(config)
    from .ppo import PPOAgent

    trained_agent = PPOAgent(config, args.device)
    control_state = trained_agent.load(policy_checkpoint)
    if control_state_checkpoint != policy_checkpoint:
        control_state = trained_agent.load_control_state(
            control_state_checkpoint
        )
    if control_state:
        environment.load_control_state_dict(control_state)
    environment.replica_adapter.start_fresh_window(0.0)
    initial_control_state = environment.control_state_dict()
    initial_control_state_hash = _state_hash(initial_control_state)
    initial_placement_hash = _state_hash(
        initial_control_state["service_replicas"]
    )
    policy = BaselineServingPolicy(
        args.baseline,
        config,
        trained_agent if args.baseline in {"ELARA", "ELARA-NB", "ELARA-SH"} else None,
    )

    progress = ProgressReporter(
        args.progress_file, args.max_slots, unit="slots"
    )
    progress.update(0, item_count=0)
    request_rows: list[dict] = []
    request_hop_rows: list[dict] = []
    slot_rows: list[dict] = []
    request_stream = hashlib.sha256()
    migration_totals = {
        "no_op": 0,
        "relocate": 0,
        "scale_out": 0,
        "scale_in": 0,
    }

    for absolute_slot, slot_requests in environment.iter_request_batches(
        slot_count=args.max_slots
    ):
        for request in slot_requests:
            request_stream.update(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "template_id": request.template_id,
                        "arrival_time_s": request.arrival_time_s,
                        "source": request.source,
                        "destination": request.destination,
                        "services": request.services,
                        "data_volumes_gb": request.data_volumes_gb,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        sessions = environment.start_request_sessions(slot_requests)
        trackers = [
            {
                "return": 0.0,
                "final_info": {},
                "slot_crossings": 0,
                "route_phases": 0,
                "used_paths": 0,
                "hop_metrics": [],
            }
            for _ in sessions
        ]
        active = list(range(len(sessions)))
        while active:
            states = [sessions[index].last_state for index in active]
            actions = policy.act_batch(states)
            next_active = []
            for index, action in zip(active, actions):
                environment.restore_request_session(sessions[index])
                state, reward, terminated, truncated, info = environment.step(
                    action
                )
                tracker = trackers[index]
                tracker["return"] += reward
                tracker["final_info"] = info
                if info.get("hop_metrics"):
                    tracker["hop_metrics"].append(dict(info["hop_metrics"]))
                for route_key in ("route", "final_route"):
                    route = info.get(route_key) or {}
                    tracker["slot_crossings"] += int(
                        route.get("slot_crossings", 0)
                    )
                    phases = route.get("slot_phases", [])
                    tracker["route_phases"] += len(phases)
                    tracker["used_paths"] += sum(
                        len(phase.get("paths", ())) for phase in phases
                    )
                sessions[index] = environment.capture_request_session()
                if state is not None and not (terminated or truncated):
                    next_active.append(index)
            active = next_active

        environment.finalize_request_sessions()
        for request, tracker in zip(slot_requests, trackers):
            info = tracker["final_info"]
            success = bool(info.get("success"))
            request_rows.append(
                {
                    "ablation": args.baseline,
                    "seed": args.test_seed,
                    "model_seed": args.model_seed,
                    "test_seed": args.test_seed,
                    "background_seed": args.background_seed,
                    "chain_length": args.chain_length,
                    "chain_length_filter": args.chain_length,
                    "arrival_mode": "total_per_slot",
                    "cycle": 1,
                    "epoch": absolute_slot + 1,
                    "slot_mod": absolute_slot,
                    "absolute_slot": absolute_slot,
                    "request_id": request.request_id,
                    "template_id": request.template_id,
                    "arrival_time_s": request.arrival_time_s,
                    "source_node": request.source,
                    "destination_node": request.destination,
                    "feasible": int(success),
                    "failure_reason": "" if success else info.get("reason", ""),
                    "total_delay_s": (
                        info.get("total_latency_s", float("nan"))
                        if success
                        else ""
                    ),
                    "total_energy_j": (
                        info.get("total_energy_j", float("nan"))
                        if success
                        else ""
                    ),
                    "reward": tracker["return"],
                    "route_slot_crossings": tracker["slot_crossings"],
                    "route_phase_count": tracker["route_phases"],
                    "route_used_path_count": tracker["used_paths"],
                    "serving_history": json.dumps(
                        info.get("serving_history", ())
                    ),
                }
            )
            for hop in tracker["hop_metrics"]:
                request_hop_rows.append(
                    {
                        "ablation": args.baseline,
                        "seed": args.test_seed,
                        "model_seed": args.model_seed,
                        "test_seed": args.test_seed,
                        "background_seed": args.background_seed,
                        "chain_length": args.chain_length,
                        "chain_length_filter": args.chain_length,
                        "absolute_slot": absolute_slot,
                        "request_id": request.request_id,
                        "template_id": request.template_id,
                        "arrival_time_s": request.arrival_time_s,
                        "request_source_node": request.source,
                        "request_destination_node": request.destination,
                        "request_feasible": int(success),
                        "request_failure_reason": (
                            "" if success else info.get("reason", "")
                        ),
                        **hop,
                    }
                )

        actions = environment.finish_time_slot(absolute_slot)
        policy.finish_time_slot()
        for action in actions:
            migration_totals[action.action] += 1
        successful = [
            row
            for row in request_rows
            if row["absolute_slot"] == absolute_slot and row["feasible"]
        ]
        slot_count = len(slot_requests)
        slot_rows.append(
            {
                "ablation": args.baseline,
                "seed": args.test_seed,
                "model_seed": args.model_seed,
                "test_seed": args.test_seed,
                "background_seed": args.background_seed,
                "chain_length": args.chain_length,
                "chain_length_filter": args.chain_length,
                "arrival_mode": "total_per_slot",
                "cycle": 1,
                "epoch": absolute_slot + 1,
                "slot_mod": absolute_slot,
                "absolute_slot": absolute_slot,
                "request_count": slot_count,
                "feasible_count": len(successful),
                "failure_count": slot_count - len(successful),
                "success_rate": (
                    len(successful) / slot_count if slot_count else ""
                ),
                "average_end_to_end_delay_s": _finite_mean(
                    row["total_delay_s"] for row in successful
                ),
                "average_energy_j": _finite_mean(
                    row["total_energy_j"] for row in successful
                ),
                "migration_action_count": len(actions),
            }
        )
        progress.update(absolute_slot + 1, item_count=len(request_rows))

    successful = [row for row in request_rows if row["feasible"]]
    successful_hops = [
        row for row in request_hop_rows if row["request_feasible"]
    ]
    summary = {
        "ablation": args.baseline,
        "method_definition": BASELINE_DESCRIPTIONS[args.baseline],
        "seed": args.test_seed,
        "model_seed": args.model_seed,
        "test_seed": args.test_seed,
        "background_seed": args.background_seed,
        "chain_length": args.chain_length,
        "arrival_mode": "total_per_slot",
        "request_template_file": str(config.request_template_file),
        "request_stream_hash": request_stream.hexdigest(),
        "request_count": len(request_rows),
        "feasible_count": len(successful),
        "failure_count": len(request_rows) - len(successful),
        "success_rate": len(successful) / max(1, len(request_rows)),
        "mean_return": _finite_mean(row["reward"] for row in request_rows),
        "mean_latency_s": _finite_mean(
            row["total_delay_s"] for row in successful
        ),
        "p95_latency_s": _finite_percentile(
            (row["total_delay_s"] for row in successful), 95.0
        ),
        "mean_energy_j": _finite_mean(
            row["total_energy_j"] for row in successful
        ),
        "mean_route_slot_crossings": _finite_mean(
            row["route_slot_crossings"] for row in request_rows
        ),
        "mean_route_phase_count": _finite_mean(
            row["route_phase_count"] for row in request_rows
        ),
        "mean_route_used_path_count": _finite_mean(
            row["route_used_path_count"] for row in request_rows
        ),
        "mean_communication_delay_per_hop_s": _finite_mean(
            row["communication_delay_s"] for row in successful_hops
        ),
        "mean_computation_queue_delay_per_hop_s": _finite_mean(
            row["computation_queue_delay_s"] for row in successful_hops
        ),
        "mean_execution_delay_per_hop_s": _finite_mean(
            row["execution_delay_s"] for row in successful_hops
        ),
        "mean_computation_delay_per_hop_s": _finite_mean(
            row["computation_delay_s"] for row in successful_hops
        ),
        "mean_communication_energy_per_hop_j": _finite_mean(
            row["communication_energy_j"] for row in successful_hops
        ),
        "mean_computation_energy_per_hop_j": _finite_mean(
            row["computation_energy_j"] for row in successful_hops
        ),
        "migration_actions": migration_totals,
        "bandit": environment.replica_adapter.summary(),
        "route_strategy": config.route_strategy,
        "adaptation_enabled": config.adaptation_enabled,
        "delay_weight": config.delay_weight,
        "energy_weight": config.energy_weight,
        "route_max_paths_per_slot": config.route_max_paths_per_slot,
        "total_arrival_lambda_per_slot": (
            config.request_arrival_lambda_total_per_slot
        ),
        "constellation_slots": args.max_slots,
        "checkpoint": str(policy_checkpoint),
        "policy_checkpoint": str(policy_checkpoint),
        "control_state_checkpoint": str(control_state_checkpoint),
        "initial_control_state_hash": initial_control_state_hash,
        "initial_placement_hash": initial_placement_hash,
    }
    _write_csv(args.output_dir / "request_metrics.csv", request_rows)
    _write_csv(
        args.output_dir / "request_hop_metrics.csv", request_hop_rows
    )
    _write_csv(args.output_dir / "slot_metrics.csv", slot_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    progress.update(
        args.max_slots,
        status="succeeded",
        item_count=len(request_rows),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
