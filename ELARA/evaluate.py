from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from .config import ELARAConfig
from .environment import ELARAEnvironment
from .progress import ProgressReporter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ELARA or lightweight baselines")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy", choices=("random", "greedy", "ppo"), default="greedy")
    parser.add_argument("--checkpoint", type=Path)
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
    parser.add_argument("--output-dir", type=Path, default=Path("ELARA/outputs/eval"))
    parser.add_argument("--progress-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def greedy_action(state) -> int:
    # Lower hops and queue are preferred; higher bottleneck rate is preferred.
    score = (
        state.candidate_features[:, 0]
        + state.candidate_features[:, 2]
        - state.candidate_features[:, 1]
    )
    score = np.where(state.action_mask, score, np.inf)
    return int(np.argmin(score))


def main() -> None:
    args = parse_args()
    progress = ProgressReporter(args.progress_file, args.episodes)
    progress.update(0)
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
    rng = random.Random(args.seed)
    environment = ELARAEnvironment(config)
    agent = None
    if args.policy == "ppo":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --policy ppo")
        from .ppo import PPOAgent

        agent = PPOAgent(config, args.device)
        control_state = agent.load(args.checkpoint)
        if control_state:
            environment.load_control_state_dict(control_state)

    records = []
    for episode in range(args.episodes):
        state = environment.reset()
        episode_return = 0.0
        final_info = {}
        route_slot_crossings = 0
        route_phase_count = 0
        while state is not None:
            if args.policy == "random":
                valid = np.flatnonzero(state.action_mask)
                action = int(rng.choice(valid.tolist()))
            elif args.policy == "greedy":
                action = greedy_action(state)
            else:
                action, _, _ = agent.act(state, deterministic=True)
            state, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
            final_info = info
            for route_key in ("route", "final_route"):
                route = info.get(route_key) or {}
                route_slot_crossings += int(route.get("slot_crossings", 0))
                route_phase_count += len(route.get("slot_phases", []))
        records.append(
            {
                "episode": episode,
                "success": int(bool(final_info.get("success"))),
                "return": episode_return,
                "latency_s": final_info.get("total_latency_s", float("nan")),
                "energy_j": final_info.get("total_energy_j", float("nan")),
                "relay_count": final_info.get("relay_count", 0),
                "subgraph_node_count": final_info.get("subgraph_node_count", 0),
                "route_slot_crossings": route_slot_crossings,
                "route_phase_count": route_phase_count,
                "migration_action_count": len(final_info.get("migration_actions", [])),
                "relocation_count": sum(
                    item.get("action") == "relocate"
                    for item in final_info.get("migration_actions", [])
                ),
            }
        )
        progress.update(episode + 1)
    with (args.output_dir / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    success = [row for row in records if row["success"]]
    summary = {
        "policy": args.policy,
        "episodes": len(records),
        "success_rate": len(success) / max(1, len(records)),
        "mean_return": float(np.mean([row["return"] for row in records])),
        "mean_latency_s": float(np.mean([row["latency_s"] for row in success])) if success else None,
        "mean_energy_j": float(np.mean([row["energy_j"] for row in success])) if success else None,
        "mean_relay_count": float(np.mean([row["relay_count"] for row in records])),
        "mean_route_slot_crossings": float(
            np.mean([row["route_slot_crossings"] for row in records])
        ),
        "mean_route_phase_count": float(
            np.mean([row["route_phase_count"] for row in records])
        ),
        "migration_action_count": int(
            sum(row["migration_action_count"] for row in records)
        ),
        "relocation_count": int(sum(row["relocation_count"] for row in records)),
        "bandit": environment.replica_adapter.summary(),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    progress.update(args.episodes, status="succeeded")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
