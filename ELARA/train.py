from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from .config import ELARAConfig
from .environment import ELARAEnvironment
from .model import require_torch, torch
from .ppo import PPOAgent, PPOTransition
from .progress import ProgressReporter


def parse_args():
    parser = argparse.ArgumentParser(description="Train the independent ELARA PPO agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-trace-slots",
        type=int,
        default=606,
        help="constellation-cycle length; training admits every request arriving in these slots",
    )
    parser.add_argument("--chain-length", type=int)
    parser.add_argument("--replicas", type=int, help="fixed replica-count compatibility override")
    parser.add_argument("--replica-min", type=int, default=5)
    parser.add_argument("--replica-max", type=int, default=10)
    parser.add_argument("--request-template-lengths", default="5,10,15")
    parser.add_argument("--arrival-lambda", type=float, default=0.35,
                        help="Poisson lambda per request template per slot")
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--route-horizon", type=int, default=3)
    parser.add_argument("--route-max-paths", type=int, default=3)
    parser.add_argument("--adaptation-window-slots", "--deployment-window", type=int, default=10)
    parser.add_argument(
        "--adaptation-top-k",
        "--adaption-top-k",
        dest="adaptation_top_k",
        type=int,
        default=10,
    )
    parser.add_argument("--disable-adaptation", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("ELARA/outputs/train"))
    parser.add_argument("--progress-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    template_lengths = tuple(
        int(value.strip()) for value in args.request_template_lengths.split(",")
        if value.strip()
    )
    config = ELARAConfig(
        seed=args.seed,
        max_trace_slots=args.max_trace_slots,
        chain_length=args.chain_length,
        replicas_per_service=args.replicas,
        replica_count_range=(args.replica_min, args.replica_max),
        request_template_chain_lengths=template_lengths,
        request_arrival_lambda_per_template_per_slot=args.arrival_lambda,
        future_topology_horizon=args.future_horizon,
        route_horizon_slots=args.route_horizon,
        route_max_paths_per_slot=args.route_max_paths,
        adaptation_window_slots=args.adaptation_window_slots,
        adaptation_top_k_services=args.adaptation_top_k,
        adaptation_enabled=not args.disable_adaptation,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    environment = ELARAEnvironment(config)
    agent = PPOAgent(config, args.device)
    cycle_slots = environment.topology.slot_count
    cycle_duration_s = cycle_slots * environment.topology.slot_duration_s
    progress = ProgressReporter(args.progress_file, cycle_slots, unit="slots")
    progress.update(0, item_count=0)
    metrics_path = args.output_dir / "training_metrics.csv"
    fields = (
        "episode", "request_id", "template_id", "arrival_time_s", "chain_length",
        "return", "latency_s", "energy_j", "steps", "relay_count",
        "route_slot_crossings", "route_phase_count", "migration_action_count",
        "no_op_count", "relocation_count", "scale_out_count", "scale_in_count",
        "policy_loss", "value_loss", "entropy",
    )
    total_steps = 0
    processed_requests = 0
    last_arrival_time_s = 0.0
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while True:
            request = environment.sample_request()
            if request.arrival_time_s >= cycle_duration_s:
                break
            episode = processed_requests
            state = environment.reset(request)
            episode_return = 0.0
            steps = 0
            final_info = {}
            losses = {}
            route_slot_crossings = 0
            route_phase_count = 0
            while state is not None:
                action, log_prob, value = agent.act(state)
                next_state, reward, terminated, truncated, info = environment.step(action)
                agent.remember(
                    PPOTransition(state, action, log_prob, value, reward, terminated or truncated)
                )
                episode_return += reward
                steps += 1
                total_steps += 1
                final_info = info
                for route_key in ("route", "final_route"):
                    route = info.get(route_key) or {}
                    route_slot_crossings += int(route.get("slot_crossings", 0))
                    route_phase_count += len(route.get("slot_phases", []))
                state = next_state
                if len(agent.buffer) >= config.rollout_steps:
                    losses = agent.update(state)
            row = {
                "episode": episode,
                "request_id": final_info.get("request_id", ""),
                "template_id": final_info.get("template_id", ""),
                "arrival_time_s": final_info.get("arrival_time_s", ""),
                "chain_length": final_info.get("chain_length", steps),
                "return": episode_return,
                "latency_s": final_info.get("total_latency_s", float("nan")),
                "energy_j": final_info.get("total_energy_j", float("nan")),
                "steps": steps,
                "relay_count": final_info.get("relay_count", 0),
                "route_slot_crossings": route_slot_crossings,
                "route_phase_count": route_phase_count,
                "migration_action_count": len(final_info.get("migration_actions", [])),
                "no_op_count": sum(item.get("action") == "no_op" for item in final_info.get("migration_actions", [])),
                "relocation_count": sum(item.get("action") == "relocate" for item in final_info.get("migration_actions", [])),
                "scale_out_count": sum(item.get("action") == "scale_out" for item in final_info.get("migration_actions", [])),
                "scale_in_count": sum(item.get("action") == "scale_in" for item in final_info.get("migration_actions", [])),
                "policy_loss": losses.get("policy_loss", ""),
                "value_loss": losses.get("value_loss", ""),
                "entropy": losses.get("entropy", ""),
            }
            writer.writerow(row)
            handle.flush()
            processed_requests += 1
            last_arrival_time_s = request.arrival_time_s
            completed_slots = min(
                cycle_slots,
                int(request.arrival_time_s // environment.topology.slot_duration_s) + 1,
            )
            progress.update(completed_slots, item_count=processed_requests)
            if processed_requests % 10 == 0:
                print(
                    f"requests={processed_requests} slot={completed_slots}/{cycle_slots} "
                    f"steps={total_steps} "
                    f"return={episode_return:.4f} latency={row['latency_s']:.4f}s"
                )
            if processed_requests % 50 == 0:
                agent.save(
                    args.output_dir / "ppo_latest.pt",
                    environment.control_state_dict(),
                )
    if agent.buffer:
        agent.update(None)
    agent.save(args.output_dir / "ppo_final.pt", environment.control_state_dict())
    with (args.output_dir / "orchestration_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "constellation_cycle_slots": cycle_slots,
                "constellation_cycle_duration_s": cycle_duration_s,
                "processed_request_count": processed_requests,
                "last_processed_arrival_time_s": last_arrival_time_s,
                "bandit": environment.replica_adapter.summary(),
                "service_replicas": {
                    service_id: service.replicas
                    for service_id, service in environment.services.items()
                },
            },
            handle,
            indent=2,
        )
    progress.update(cycle_slots, status="succeeded", item_count=processed_requests)
    print(
        f"training complete: requests={processed_requests} slots={cycle_slots} "
        f"checkpoint={args.output_dir / 'ppo_final.pt'}"
    )


if __name__ == "__main__":
    main()
