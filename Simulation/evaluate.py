from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .agents.migration import ReplicaPlacementMigrationAgent
from .agents.ppo_gnn_agent import PPOGNNExecutionAgent
from .config import SimulationConfig
from .core.env import SimulationEnvironment
from .core.metrics import summarize_results
from .domain.constellation import node_id_to_sat_name
from .domain.request import generate_slot_arrivals, load_request_templates


DEFAULT_MODEL_DIR = SimulationConfig().output_dir.parent / "fix_rep_pattern_train_data"
DEFAULT_OUTPUT_DIR = SimulationConfig().output_dir.parent / "eval_outputs"


def make_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value):
        return make_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: make_jsonable(row.get(key, "")) for key in fieldnames})


def flatten_route_counts(route_mode_counts: dict) -> dict:
    return {
        f"route_mode_{mode}_count": count
        for mode, count in sorted(route_mode_counts.items())
        if mode
    }


def request_metric_row(epoch: int, cycle: int, slot_mod: int, result: dict) -> dict:
    request = result["request"]
    return {
        "epoch": epoch,
        "cycle": cycle,
        "slot_mod": slot_mod,
        "template_id": request.get("template_id"),
        "request_id": request["request_id"],
        "chain_length": len(request["services"]),
        "start_time_s": request["start_time"],
        "source_node": request["source_node"],
        "source_satellite": node_id_to_sat_name(request["source_node"]),
        "destination_node": request["destination_node"],
        "destination_satellite": node_id_to_sat_name(request["destination_node"]),
        "services": request["services"],
        "feasible": result["feasible"],
        "failed_stage": result["failed_stage"],
        "failure_reason": result.get("failure_reason", ""),
        "finish_time_s": result["finish_time_s"]
        if math.isfinite(result["finish_time_s"])
        else "",
        "total_delay_s": result["total_delay_s"]
        if math.isfinite(result["total_delay_s"])
        else "",
        "total_energy_j": result["total_energy_j"]
        if math.isfinite(result["total_energy_j"])
        else "",
        "reward": result["reward"],
    }


def route_detail_rows(result: dict) -> list[dict]:
    rows = []
    for route in result.get("route_details", []):
        rows.append(
            {
                "request_id": route["request_id"],
                "stage": route["stage"],
                "source_node": route["source_node"],
                "target_node": route["target_node"],
                "data_gb": route["data_gb"],
                "route_mode": route["route_mode"],
                "communication_delay_s": route["communication_delay_s"],
                "communication_energy_j": route["communication_energy_j"],
                "slot_crossings": route["slot_crossings"],
                "path": json.dumps(route["path"]),
                "slot_paths": json.dumps(make_jsonable(route["slot_paths"])),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained three-layer LEO microservice simulator over one "
            "constellation period with the same Poisson request-arrival process as training."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--request-template-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--bandit-arm-stats", type=Path, default=None)
    parser.add_argument(
        "--no-load-bandit",
        action="store_true",
        help="Start evaluation with an empty Bandit policy instead of loading trained arm stats.",
    )
    parser.add_argument("--isl-csv", type=Path, default=None)
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Optional smoke-test slot limit. Omit it to evaluate the full constellation period.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--arrival-lambda",
        type=float,
        default=SimulationConfig().request_arrival_lambda_per_pattern_per_slot,
        help="Poisson lambda for each fixed request template in each time slot.",
    )
    parser.add_argument(
        "--bandit-period-slots",
        type=int,
        default=10,
        help="Apply the slow-layer Bandit deployment adjustment every N time slots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bandit_period_slots = max(1, args.bandit_period_slots)
    template_csv = args.request_template_csv or args.model_dir / "request_templates.csv"
    checkpoint = args.checkpoint or args.model_dir / "ppo_gnn_latest.pth"
    bandit_arm_stats = args.bandit_arm_stats or args.model_dir / "bandit_arm_stats.csv"
    if not template_csv.exists():
        raise FileNotFoundError(f"Request template CSV not found: {template_csv}")

    config = SimulationConfig(
        random_seed=args.seed,
        isl_log_csv=args.isl_csv or SimulationConfig().isl_log_csv,
        max_slots_to_load=args.max_slots,
        process_single_request=False,
        request_arrival_lambda_per_pattern_per_slot=args.arrival_lambda,
        output_dir=args.output_dir,
    )
    bandit_agent = ReplicaPlacementMigrationAgent(config)
    bandit_policy_loaded = False
    bandit_loaded_arm_count = 0
    if not args.no_load_bandit and bandit_arm_stats.exists():
        bandit_loaded_arm_count = bandit_agent.load_arm_stats(bandit_arm_stats)
        bandit_policy_loaded = bandit_loaded_arm_count > 0
    env = SimulationEnvironment(
        config,
        migration_agent=bandit_agent,
        auto_generate_requests=False,
        auto_apply_migration=False,
    ).build()
    slot_count = env.context["slot_count"]
    slot_duration = env.context["slot_duration"]

    agent = PPOGNNExecutionAgent(
        config,
        hidden_dim=config.ppo_hidden_dim,
        train_mode=False,
        device=args.device,
    )
    checkpoint_loaded = False
    checkpoint_load_error = ""
    if checkpoint.exists():
        try:
            agent.load(checkpoint)
            checkpoint_loaded = True
        except Exception as exc:
            checkpoint_load_error = f"{type(exc).__name__}: {exc}"

    request_templates = load_request_templates(template_csv)
    arrival_rng = random.Random(args.seed + 10_000)
    next_request_id = 1
    slot_rows: list[dict] = []
    request_rows: list[dict] = []
    route_rows: list[dict] = []
    bandit_action_rows: list[dict] = []
    all_results: list[dict] = []
    bandit_window_requests = []
    bandit_feedback_results = []
    pending_migration_actions = []

    for epoch in range(1, slot_count + 1):
        absolute_slot = epoch - 1
        cycle = absolute_slot // slot_count + 1
        slot_mod = absolute_slot % slot_count
        arrivals, next_request_id, arrival_info = generate_slot_arrivals(
            arrival_rng,
            env.context,
            request_templates,
            absolute_slot,
            next_request_id,
        )
        results = env.execute_requests(arrivals, agent)
        bandit_agent.observe_failed_replicas(results)
        all_results.extend(results)
        bandit_window_requests.extend(arrivals)
        bandit_feedback_results.extend(results)

        bandit_updated = False
        migration_actions = []
        is_cycle_last_slot = slot_mod == slot_count - 1
        if (slot_mod + 1) % args.bandit_period_slots == 0 and not is_cycle_last_slot:
            if pending_migration_actions:
                bandit_agent.observe_execution_feedback(
                    pending_migration_actions, bandit_feedback_results
                )
            else:
                bandit_agent.observe_service_pressure_feedback(bandit_feedback_results)
            migration_actions = env.apply_migration(bandit_window_requests)
            pending_migration_actions = migration_actions
            bandit_feedback_results = []
            bandit_window_requests = []
            bandit_updated = True

        summary = summarize_results(results)
        total_reward = sum(result.get("reward", 0.0) for result in results)
        processed_request_count = len(results)
        completed_hop_count = sum(len(result.get("execution_plan", [])) for result in results)
        route_record_count = sum(len(result.get("route_details", [])) for result in results)
        row = {
            "epoch": epoch,
            "cycle": cycle,
            "slot_mod": slot_mod,
            "slot_start_time_s": absolute_slot * slot_duration,
            "slot_duration_s": slot_duration,
            "arrival_count": arrival_info["arrival_count"],
            "processed_request_count": processed_request_count,
            "completed_hop_count": completed_hop_count,
            "route_record_count": route_record_count,
            "total_reward": total_reward,
            "average_reward_per_request": total_reward / processed_request_count
            if processed_request_count
            else None,
            "average_reward_per_hop": total_reward / completed_hop_count
            if completed_hop_count
            else None,
            "request_count": summary["request_count"],
            "feasible_count": summary["feasible_count"],
            "success_rate": summary["success_rate"],
            "average_end_to_end_delay_s": summary["average_end_to_end_delay_s"],
            "p95_end_to_end_delay_s": summary["p95_end_to_end_delay_s"],
            "average_energy_j": summary["average_energy_j"],
            "p95_energy_j": summary["p95_energy_j"],
            "average_communication_delay_s": summary["average_communication_delay_s"],
            "average_slot_crossings": summary["average_slot_crossings"],
            **flatten_route_counts(summary["route_mode_counts"]),
            "bandit_updated": bandit_updated,
            "bandit_action_count": len(migration_actions),
            **{f"bandit_{key}": value for key, value in bandit_agent.summary().items()},
            **{
                f"routing_{key}": value
                for key, value in env.context["routing_cache"]["stats"].items()
            },
        }
        slot_rows.append(row)
        for result in results:
            request_rows.append(request_metric_row(epoch, cycle, slot_mod, result))
            route_rows.extend(route_detail_rows(result))
        for action in migration_actions:
            bandit_action_rows.append({"epoch": epoch, **asdict(action)})

        print(json.dumps(make_jsonable(row), ensure_ascii=False))
        env.context["routing_cache"]["route_results"].clear()
        env.context.get("route_estimate_cache", {}).clear()

    if pending_migration_actions and bandit_feedback_results:
        bandit_agent.observe_execution_feedback(pending_migration_actions, bandit_feedback_results)

    overall_summary = summarize_results(all_results)
    final_summary = {
        "evaluation_slots": slot_count,
        "slot_duration_s": slot_duration,
        "request_template_csv": template_csv,
        "request_template_count": len(request_templates),
        "checkpoint": checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_load_error": checkpoint_load_error,
        "bandit_arm_stats": bandit_arm_stats,
        "bandit_policy_loaded": bandit_policy_loaded,
        "bandit_loaded_arm_count": bandit_loaded_arm_count,
        "arrival_lambda_per_pattern_per_slot": args.arrival_lambda,
        "bandit_period_slots": args.bandit_period_slots,
        "request_count": len(all_results),
        "summary": overall_summary,
        "bandit_summary": bandit_agent.summary(),
        "routing_cache_summary": dict(env.context["routing_cache"]["stats"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluation_metrics.csv", slot_rows)
    write_csv(args.output_dir / "request_metrics.csv", request_rows)
    write_csv(args.output_dir / "route_details.csv", route_rows)
    write_csv(args.output_dir / "bandit_actions.csv", bandit_action_rows)
    write_csv(args.output_dir / "bandit_arm_stats.csv", bandit_agent.export_arm_stats())
    (args.output_dir / "Evaluation_Run_Summary.json").write_text(
        json.dumps(make_jsonable(final_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(make_jsonable(final_summary), ensure_ascii=False, indent=2))
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
