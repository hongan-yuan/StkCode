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
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trace-slots", type=int, default=120)
    parser.add_argument("--chain-length", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--future-horizon", type=int, default=3)
    parser.add_argument("--route-horizon", type=int, default=3)
    parser.add_argument("--route-max-paths", type=int, default=3)
    parser.add_argument("--deployment-window", type=int, default=20)
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
    progress = ProgressReporter(args.progress_file, args.episodes)
    progress.update(0)
    require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ELARAConfig(
        seed=args.seed,
        max_trace_slots=args.max_trace_slots,
        chain_length=args.chain_length,
        replicas_per_service=args.replicas,
        future_topology_horizon=args.future_horizon,
        route_horizon_slots=args.route_horizon,
        route_max_paths_per_slot=args.route_max_paths,
        deployment_window_requests=args.deployment_window,
        adaptation_top_k_services=args.adaptation_top_k,
        adaptation_enabled=not args.disable_adaptation,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    environment = ELARAEnvironment(config)
    agent = PPOAgent(config, args.device)
    metrics_path = args.output_dir / "training_metrics.csv"
    fields = (
        "episode", "return", "latency_s", "energy_j", "steps", "relay_count",
        "route_slot_crossings", "route_phase_count", "migration_action_count",
        "relocation_count", "policy_loss", "value_loss", "entropy",
    )
    total_steps = 0
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in range(args.episodes):
            state = environment.reset()
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
                "return": episode_return,
                "latency_s": final_info.get("total_latency_s", float("nan")),
                "energy_j": final_info.get("total_energy_j", float("nan")),
                "steps": steps,
                "relay_count": final_info.get("relay_count", 0),
                "route_slot_crossings": route_slot_crossings,
                "route_phase_count": route_phase_count,
                "migration_action_count": len(final_info.get("migration_actions", [])),
                "relocation_count": sum(
                    item.get("action") == "relocate"
                    for item in final_info.get("migration_actions", [])
                ),
                "policy_loss": losses.get("policy_loss", ""),
                "value_loss": losses.get("value_loss", ""),
                "entropy": losses.get("entropy", ""),
            }
            writer.writerow(row)
            handle.flush()
            progress.update(episode + 1)
            if (episode + 1) % 10 == 0:
                print(
                    f"episode={episode + 1} steps={total_steps} "
                    f"return={episode_return:.4f} latency={row['latency_s']:.4f}s"
                )
            if (episode + 1) % 50 == 0:
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
                "bandit": environment.replica_adapter.summary(),
                "service_replicas": {
                    service_id: service.replicas
                    for service_id, service in environment.services.items()
                },
            },
            handle,
            indent=2,
        )
    progress.update(args.episodes, status="succeeded")
    print(f"training complete: {args.output_dir / 'ppo_final.pt'}")


if __name__ == "__main__":
    main()
