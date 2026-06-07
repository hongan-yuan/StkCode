from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from .config import SimulationConfig
from .env import SimulationEnvironment
from .metrics import summarize_results
from .migration import ReplicaPlacementMigrationAgent
from .ppo_gnn_agent import PPOGNNExecutionAgent
from .request import generate_request_templates, generate_slot_arrivals


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
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
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def request_template_rows(templates) -> list[dict]:
    return [
        {
            "template_id": template.request_id,
            "request_template_id": template.template_id,
            "chain_length": len(template.services),
            "source_node": template.source_node,
            "destination_node": template.destination_node,
            "services": template.services,
            "service_workload_cycles": template.service_workload_cycles,
            "input_data_gb": template.input_data_gb,
            "data_gb_between_services": template.data_gb_between_services,
            "output_data_gb": template.output_data_gb,
        }
        for template in templates
    ]


def training_summary_row(
    args: argparse.Namespace,
    base_config: SimulationConfig,
    model_dir: Path,
    slot_count: int,
    slot_duration: float,
    agent: PPOGNNExecutionAgent,
    completed_epochs: int,
    status: str,
) -> dict:
    return {
        "status": status,
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "slot_count": slot_count,
        "slot_duration_s": slot_duration,
        "constellation_cycles": completed_epochs / slot_count if slot_count else 0,
        "model_dir": model_dir,
        "training_available": agent.training_available,
        "ppo_device": str(agent.device) if agent.device is not None else "fallback",
        "log_every": args.log_every,
        "checkpoint_policy": "save final trained model only",
        "model_file": (
            model_dir / "ppo_gnn_latest.pth"
            if agent.training_available
            else model_dir / "ppo_gnn_latest.json"
        ),
        "ppo_update_slots": args.ppo_update_slots,
        "ppo_batch_size": base_config.ppo_batch_size,
        "bandit_period_slots": args.bandit_period_slots,
        "route_horizon_slots": args.route_horizon_slots,
        "arrival_lambda_per_pattern_per_slot": args.arrival_lambda,
        "output_granularity": f"CSV/JSON outputs flushed every {args.log_every} epochs.",
        "deployment_reset_each_cycle": False,
    }


def write_training_outputs(
    model_dir: Path,
    training_rows: list[dict],
    request_rows: list[dict],
    bandit_action_rows: list[dict],
    bandit_agent: ReplicaPlacementMigrationAgent,
    request_templates,
    base_config: SimulationConfig,
    summary_row: dict,
) -> None:
    write_csv(model_dir / "training_metrics.csv", training_rows)
    write_csv(model_dir / "request_metrics.csv", request_rows)
    write_csv(model_dir / "bandit_actions.csv", bandit_action_rows)
    write_csv(model_dir / "bandit_arm_stats.csv", bandit_agent.export_arm_stats())
    write_csv(model_dir / "request_templates.csv", request_template_rows(request_templates))
    write_csv(model_dir / "training_run_summary.csv", [summary_row])
    (model_dir / "training_config.json").write_text(
        json.dumps(jsonable(asdict(base_config)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def failed_request_diagnostics(epoch: int, results: list[dict]) -> list[dict]:
    diagnostics = []
    for result in results:
        if result.get("feasible", True):
            continue
        request = result["request"]
        failed_stage = result.get("failed_stage")
        services = request.get("services", [])
        if isinstance(failed_stage, int) and failed_stage < len(services):
            failed_service_id = services[failed_stage]
            failed_stage_type = "microservice_execution"
        else:
            failed_service_id = None
            failed_stage_type = "destination_return"

        diagnostics.append(
            {
                "event": "failed_request_diagnostic",
                "epoch": epoch,
                "request_id": request.get("request_id"),
                "template_id": request.get("template_id"),
                "chain_length": len(services),
                "services": services,
                "source_node": request.get("source_node"),
                "destination_node": request.get("destination_node"),
                "failed_stage": failed_stage,
                "failed_stage_type": failed_stage_type,
                "failed_service_id": failed_service_id,
                "failure_reason": result.get("failure_reason", "unknown_failure"),
                "completed_stage_count": len(result.get("execution_plan", [])),
                "failed_route": result.get("failed_route"),
                "last_route": (
                    result.get("route_details", [])[-1]
                    if result.get("route_details")
                    else None
                ),
            }
        )
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the PPO-GNN fast-layer agent over Poisson request arrivals "
            "on a slot-by-slot constellation timeline."
        )
    )
    parser.add_argument("--epochs", "--episodes", dest="epochs", type=int, default=6060)
    parser.add_argument("--max-slots", type=int, default=606)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
        help="Training device for PPO-GNN: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--model-dir", type=Path,
        default=SimulationConfig().output_dir.parent / "dyn_train_data_260605",
        help="Directory for checkpoints and training logs.",
    )
    parser.add_argument("--log-every", type=int, default=500,
        help="Flush training CSV files every N epochs; console logging still runs every epoch.",
    )
    parser.add_argument("--single-request-debug", action="store_true",
        help="Legacy debug flag; slot-arrival training ignores it.",
    )
    parser.add_argument("--ppo-update-slots", type=int, default=5,
        help="Run one PPO update after collecting this many time slots.",
    )
    parser.add_argument("--bandit-period-slots", type=int, default=10,
        help="Run the Bandit placement/migration layer every N time slots.",
    )
    parser.add_argument("--arrival-lambda", type=float,
        default=SimulationConfig().request_arrival_lambda_per_pattern_per_slot,
        help="Poisson lambda for each fixed request template in each time slot.",
    )
    parser.add_argument("--route-horizon-slots", type=int, default=3,
        help="Maximum number of future time slots a route may use before failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_every = max(1, args.log_every)
    args.ppo_update_slots = max(1, args.ppo_update_slots)
    args.bandit_period_slots = max(1, args.bandit_period_slots)
    args.route_horizon_slots = max(1, args.route_horizon_slots)
    model_dir: Path = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    for stale_checkpoint in model_dir.glob("ppo_gnn_epoch_*.pth"):
        stale_checkpoint.unlink()
    for stale_model in (
        model_dir / "ppo_gnn_latest.pth",
        model_dir / "ppo_gnn_latest.json",
    ):
        if stale_model.exists():
            stale_model.unlink()

    base_config = SimulationConfig(
        random_seed=args.seed,
        max_slots_to_load=args.max_slots,
        process_single_request=False,
        request_arrival_lambda_per_pattern_per_slot=args.arrival_lambda,
        route_horizon_slots=args.route_horizon_slots,
        output_dir=model_dir,
    )
    agent = PPOGNNExecutionAgent(
        base_config,
        hidden_dim=base_config.ppo_hidden_dim,
        train_mode=True,
        device=args.device,
    )
    bandit_agent = ReplicaPlacementMigrationAgent(base_config)
    arrival_rng = random.Random(args.seed + 10_000)

    training_rows: list[dict] = []
    request_rows: list[dict] = []
    bandit_action_rows: list[dict] = []

    env = SimulationEnvironment(
        base_config,
        migration_agent=bandit_agent,
        auto_generate_requests=False,
        auto_apply_migration=False,
    ).build()

    slot_count = env.context["slot_count"]
    slot_duration = env.context["slot_duration"]
    request_templates = generate_request_templates(arrival_rng, env.context)

    next_request_id = 1
    bandit_window_requests = []
    bandit_feedback_results = []
    pending_migration_actions = []
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        completed_epochs = epoch
        absolute_slot = epoch - 1
        cycle_index = absolute_slot // slot_count
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
        summary = summarize_results(results)
        bandit_window_requests.extend(arrivals)
        bandit_feedback_results.extend(results)

        update_stats = {
            "updated": False,
            "reason": f"waiting for {args.ppo_update_slots} time slots",
            "transition_count": len(getattr(agent, "transitions", [])),
        }
        if agent.training_available and epoch % args.ppo_update_slots == 0:
            update_stats = agent.ppo_update(
                clip_epsilon=base_config.ppo_clip_epsilon,
                gamma=base_config.ppo_gamma,
                epochs=1,
            )
        elif not agent.training_available:
            update_stats = {
                "updated": False,
                "reason": "PyTorch is not installed; ran fallback masked-softmax policy.",
                "transition_count": 0,
            }

        bandit_updated = False
        migration_actions = []
        is_cycle_last_slot = slot_mod == slot_count - 1
        if (slot_mod + 1) % args.bandit_period_slots == 0 and not is_cycle_last_slot:
            if pending_migration_actions:
                bandit_agent.observe_execution_feedback(
                    pending_migration_actions, bandit_feedback_results
                )
            migration_actions = env.apply_migration(bandit_window_requests)
            pending_migration_actions = migration_actions
            bandit_feedback_results = []
            bandit_window_requests = []
            bandit_updated = True

        if is_cycle_last_slot:
            if pending_migration_actions and bandit_feedback_results:
                bandit_agent.observe_execution_feedback(
                    pending_migration_actions, bandit_feedback_results
                )
            pending_migration_actions = []
            bandit_feedback_results = []
            bandit_window_requests = []

        payload = env.payload_for_results(arrivals, results, agent, migration_actions)
        total_reward = sum(result.get("reward", 0.0) for result in results)
        processed_request_count = payload["request_count"]
        completed_hop_count = sum(len(result.get("execution_plan", [])) for result in results)
        route_record_count = sum(len(result.get("route_details", [])) for result in results)
        average_reward_per_request = (
            total_reward / processed_request_count if processed_request_count else None
        )
        average_reward_per_hop = (
            total_reward / completed_hop_count if completed_hop_count else None
        )
        row = {
            "epoch": epoch,
            "cycle": cycle_index + 1,
            "slot_mod": slot_mod,
            "slot_start_time_s": absolute_slot * slot_duration,
            "slot_duration_s": slot_duration,
            "arrival_count": arrival_info["arrival_count"],
            "nonempty_slot": processed_request_count > 0,
            "request_pool_count": len(request_templates),
            "processed_request_count": processed_request_count,
            "completed_hop_count": completed_hop_count,
            "route_record_count": route_record_count,
            "covered_all_request_patterns": all(
                count > 0 for count in arrival_info["arrival_counts_by_template"].values()
            ),
            "selected_request_ids": json.dumps(payload["selected_request_ids"]),
            "total_reward": total_reward,
            "average_reward_per_request": average_reward_per_request,
            "average_reward_per_hop": average_reward_per_hop,
            **summary,
            "ppo_updated": update_stats.get("updated", False),
            "ppo_loss": update_stats.get("loss", ""),
            "ppo_policy_loss": update_stats.get("policy_loss", ""),
            "ppo_value_loss": update_stats.get("value_loss", ""),
            "ppo_entropy": update_stats.get("entropy", ""),
            "ppo_transition_count": update_stats.get("transition_count", 0),
            "ppo_batch_size": update_stats.get("batch_size", base_config.ppo_batch_size),
            "ppo_sampled_transition_count": update_stats.get("sampled_transition_count", 0),
            "ppo_update_reason": update_stats.get("reason", ""),
            "ppo_device": str(agent.device) if agent.device is not None else "fallback",
            "bandit_updated": bandit_updated,
            "bandit_action_count": len(migration_actions),
            **{f"bandit_{k}": v for k, v in payload["bandit_summary"].items()},
            **{f"routing_{k}": v for k, v in payload["routing_cache_summary"].items()},
        }
        training_rows.append(row)

        if summary["feasible_count"] < summary["request_count"]:
            for diagnostic in failed_request_diagnostics(epoch, results):
                print(json.dumps(jsonable(diagnostic), ensure_ascii=False))

        for result in results:
            request = result["request"]
            request_rows.append(
                {
                    "epoch": epoch,
                    "cycle": cycle_index + 1,
                    "slot_mod": slot_mod,
                    "template_id": request["template_id"],
                    "request_id": request["request_id"],
                    "chain_length": len(request["services"]),
                    "start_time_s": request["start_time"],
                    "source_node": request["source_node"],
                    "destination_node": request["destination_node"],
                    "feasible": result["feasible"],
                    "reward": result["reward"],
                    "total_delay_s": result["total_delay_s"]
                    if math.isfinite(result["total_delay_s"])
                    else "",
                    "total_energy_j": result["total_energy_j"]
                    if math.isfinite(result["total_energy_j"])
                    else "",
                    "failed_stage": result["failed_stage"],
                    "failure_reason": result.get("failure_reason", ""),
                }
            )

        for action in payload["migration_actions"]:
            action_row = {"epoch": epoch, **action}
            bandit_action_rows.append(action_row)

        should_flush_logs = epoch % args.log_every == 0
        if should_flush_logs:
            write_training_outputs(
                model_dir,
                training_rows,
                request_rows,
                bandit_action_rows,
                bandit_agent,
                request_templates,
                base_config,
                training_summary_row(
                    args,
                    base_config,
                    model_dir,
                    slot_count,
                    slot_duration,
                    agent,
                    completed_epochs,
                    "running" if epoch < args.epochs else "complete",
                ),
            )

        print(json.dumps(jsonable(row), ensure_ascii=False))
        env.context["routing_cache"]["route_results"].clear()

    agent.save(model_dir / "ppo_gnn_latest.pth")
    final_summary = training_summary_row(
        args,
        base_config,
        model_dir,
        slot_count,
        slot_duration,
        agent,
        completed_epochs,
        "complete",
    )
    write_training_outputs(
        model_dir,
        training_rows,
        request_rows,
        bandit_action_rows,
        bandit_agent,
        request_templates,
        base_config,
        final_summary,
    )
    print(json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
