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
from .request_templates import DEFAULT_TEMPLATE_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ELARA or lightweight baselines")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="ignore --episodes and evaluate every Poisson arrival in one loaded constellation cycle",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy", choices=("random", "greedy", "ppo"), default="greedy")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-trace-slots", type=int, default=120)
    parser.add_argument("--chain-length", type=int)
    parser.add_argument("--replicas", type=int, help="fixed replica-count compatibility override")
    parser.add_argument("--replica-min", type=int, default=5)
    parser.add_argument("--replica-max", type=int, default=10)
    parser.add_argument("--request-template-lengths", default="5,10,15")
    parser.add_argument("--request-template-file", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--request-data-scale", type=float, default=1.0)
    parser.add_argument("--arrival-lambda", type=float, default=0.35,
                        help="Poisson lambda per request template per slot")
    parser.add_argument("--delay-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=0.5)
    parser.add_argument("--compute-capacity-scale", type=float, default=1.0)
    parser.add_argument("--link-capacity-scale", type=float, default=1.0)
    parser.add_argument("--background-load-scale", type=float, default=0.5)
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
        request_template_file=args.request_template_file,
        request_data_scale=args.request_data_scale,
        request_arrival_lambda_per_template_per_slot=args.arrival_lambda,
        delay_weight=args.delay_weight,
        energy_weight=args.energy_weight,
        compute_capacity_scale=args.compute_capacity_scale,
        link_capacity_scale=args.link_capacity_scale,
        background_load_scale=args.background_load_scale,
        future_topology_horizon=args.future_horizon,
        route_horizon_slots=args.route_horizon,
        route_max_paths_per_slot=args.route_max_paths,
        adaptation_window_slots=args.adaptation_window_slots,
        adaptation_top_k_services=args.adaptation_top_k,
        adaptation_enabled=not args.disable_adaptation,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    environment = ELARAEnvironment(config)
    cycle_slots = environment.topology.slot_count
    progress = ProgressReporter(
        args.progress_file,
        cycle_slots if args.full_cycle else args.episodes,
        unit="slots" if args.full_cycle else "episodes",
    )
    progress.update(0, item_count=0)
    agent = None
    if args.policy == "ppo":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --policy ppo")
        from .ppo import PPOAgent

        agent = PPOAgent(config, args.device)
        control_state = agent.load(args.checkpoint)
        if control_state:
            environment.load_control_state_dict(control_state)
            # Evaluation restarts its own cycle at time zero.  Preserve the
            # learned LinUCB parameters and final placement, but do not carry a
            # partially collected training window or its absolute slot clock.
            environment.replica_adapter.start_fresh_window(0.0)

    records = []
    hop_records = []
    episode = 0
    batches = environment.iter_request_batches(
        slot_count=cycle_slots if args.full_cycle else None,
        request_count=None if args.full_cycle else args.episodes,
    )
    for absolute_slot, slot_requests in batches:
        slot_records = []
        sessions = environment.start_request_sessions(slot_requests)
        trackers = [
            {
                "return": 0.0,
                "final_info": {},
                "route_slot_crossings": 0,
                "route_phase_count": 0,
                "route_augmentation_count": 0,
                "hop_metrics": [],
            }
            for _ in sessions
        ]
        active = list(range(len(sessions)))
        while active:
            states = [sessions[index].last_state for index in active]
            if args.policy == "ppo":
                decisions = agent.act_batch(states, deterministic=True)
            else:
                decisions = []
                for state in states:
                    if args.policy == "random":
                        valid = np.flatnonzero(state.action_mask)
                        action = int(rng.choice(valid.tolist()))
                    else:
                        action = greedy_action(state)
                    decisions.append((action, 0.0, 0.0))

            next_active = []
            for index, (action, _, _) in zip(active, decisions):
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
                    tracker["route_slot_crossings"] += int(
                        route.get("slot_crossings", 0)
                    )
                    tracker["route_phase_count"] += len(
                        route.get("slot_phases", [])
                    )
                    tracker["route_augmentation_count"] += int(
                        route.get("min_cost_flow_augmentations", 0)
                    )
                sessions[index] = environment.capture_request_session()
                if state is not None:
                    next_active.append(index)
            active = next_active

        for request, tracker in zip(slot_requests, trackers):
            final_info = tracker["final_info"]
            request_success = int(bool(final_info.get("success")))
            slot_records.append(
                {
                    "episode": episode,
                    "request_id": final_info.get("request_id", ""),
                    "template_id": final_info.get("template_id", ""),
                    "arrival_time_s": final_info.get("arrival_time_s", ""),
                    "chain_length": final_info.get("chain_length", ""),
                    "success": request_success,
                    "return": tracker["return"],
                    "latency_s": final_info.get("total_latency_s", float("nan")),
                    "energy_j": final_info.get("total_energy_j", float("nan")),
                    "relay_count": final_info.get("relay_count", 0),
                    "subgraph_node_count": final_info.get("subgraph_node_count", 0),
                    "route_slot_crossings": tracker["route_slot_crossings"],
                    "route_phase_count": tracker["route_phase_count"],
                    "route_augmentation_count": tracker[
                        "route_augmentation_count"
                    ],
                    "migration_action_count": 0,
                    "no_op_count": 0,
                    "relocation_count": 0,
                    "scale_out_count": 0,
                    "scale_in_count": 0,
                }
            )
            for hop in tracker["hop_metrics"]:
                hop_records.append(
                    {
                        "policy": args.policy,
                        "seed": args.seed,
                        "absolute_slot": absolute_slot,
                        "request_id": request.request_id,
                        "template_id": request.template_id,
                        "arrival_time_s": request.arrival_time_s,
                        "chain_length": len(request.services),
                        "request_source_node": request.source,
                        "request_destination_node": request.destination,
                        "request_success": request_success,
                        "request_failure_reason": (
                            ""
                            if request_success
                            else final_info.get("reason", "")
                        ),
                        **hop,
                    }
                )
            episode += 1

        environment.finalize_request_sessions()
        migration_actions = environment.finish_time_slot(absolute_slot)
        if slot_records:
            slot_records[-1].update(
                migration_action_count=len(migration_actions),
                no_op_count=sum(
                    item.action == "no_op" for item in migration_actions
                ),
                relocation_count=sum(
                    item.action == "relocate" for item in migration_actions
                ),
                scale_out_count=sum(
                    item.action == "scale_out" for item in migration_actions
                ),
                scale_in_count=sum(
                    item.action == "scale_in" for item in migration_actions
                ),
            )
            records.extend(slot_records)
        completed = (
            absolute_slot + 1
            if args.full_cycle
            else episode
        )
        progress.update(completed, item_count=episode)
    with (args.output_dir / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    if hop_records:
        with (
            args.output_dir / "request_hop_metrics.csv"
        ).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=hop_records[0].keys()
            )
            writer.writeheader()
            writer.writerows(hop_records)
    success = [row for row in records if row["success"]]
    summary = {
        "policy": args.policy,
        "episodes": len(records),
        "full_cycle": args.full_cycle,
        "constellation_cycle_slots": cycle_slots if args.full_cycle else None,
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
        "mean_route_augmentation_count": float(
            np.mean([row["route_augmentation_count"] for row in records])
        ),
        "mean_communication_delay_per_hop_s": (
            float(
                np.mean(
                    [
                        row["communication_delay_s"]
                        for row in hop_records
                        if row["request_success"]
                    ]
                )
            )
            if any(row["request_success"] for row in hop_records)
            else None
        ),
        "mean_computation_queue_delay_per_hop_s": (
            float(
                np.mean(
                    [
                        row["computation_queue_delay_s"]
                        for row in hop_records
                        if row["request_success"]
                    ]
                )
            )
            if any(row["request_success"] for row in hop_records)
            else None
        ),
        "mean_execution_delay_per_hop_s": (
            float(
                np.mean(
                    [
                        row["execution_delay_s"]
                        for row in hop_records
                        if row["request_success"]
                    ]
                )
            )
            if any(row["request_success"] for row in hop_records)
            else None
        ),
        "mean_communication_energy_per_hop_j": (
            float(
                np.mean(
                    [
                        row["communication_energy_j"]
                        for row in hop_records
                        if row["request_success"]
                    ]
                )
            )
            if any(row["request_success"] for row in hop_records)
            else None
        ),
        "mean_computation_energy_per_hop_j": (
            float(
                np.mean(
                    [
                        row["computation_energy_j"]
                        for row in hop_records
                        if row["request_success"]
                    ]
                )
            )
            if any(row["request_success"] for row in hop_records)
            else None
        ),
        "migration_action_count": int(
            sum(row["migration_action_count"] for row in records)
        ),
        "relocation_count": int(sum(row["relocation_count"] for row in records)),
        "scale_out_count": int(sum(row["scale_out_count"] for row in records)),
        "scale_in_count": int(sum(row["scale_in_count"] for row in records)),
        "no_op_count": int(sum(row["no_op_count"] for row in records)),
        "bandit": environment.replica_adapter.summary(),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    progress.update(
        cycle_slots if args.full_cycle else args.episodes,
        status="succeeded",
        item_count=len(records),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
