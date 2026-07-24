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
from .request_templates import DEFAULT_TEMPLATE_PATH


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
    parser.add_argument("--ppo-minibatch-size", type=int, default=16)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=128,
        help="legacy checkpoint/config field; PPO updates are scheduled by time slots",
    )
    parser.add_argument("--ppo-update-interval-slots", type=int, default=5)
    parser.add_argument("--ppo-transaction-history-slots", type=int)
    parser.add_argument("--ppo-transaction-max-reuse", type=int, default=2)
    parser.add_argument("--pretrain-cycles", type=int, default=1)
    parser.add_argument("--joint-training-cycles", type=int, default=1)
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
        request_template_file=args.request_template_file,
        request_data_scale=args.request_data_scale,
        request_arrival_lambda_per_template_per_slot=args.arrival_lambda,
        delay_weight=args.delay_weight,
        energy_weight=args.energy_weight,
        compute_capacity_scale=args.compute_capacity_scale,
        link_capacity_scale=args.link_capacity_scale,
        background_load_scale=args.background_load_scale,
        future_topology_horizon=args.future_horizon,
        ppo_minibatch_size=args.ppo_minibatch_size,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_entropy_coef=args.ppo_entropy_coef,
        ppo_epochs=args.ppo_epochs,
        rollout_steps=args.rollout_steps,
        ppo_update_interval_slots=args.ppo_update_interval_slots,
        ppo_transaction_history_slots=(
            args.ppo_transaction_history_slots
            if args.ppo_transaction_history_slots is not None
            else 2 * args.ppo_update_interval_slots
        ),
        ppo_transaction_max_reuse=args.ppo_transaction_max_reuse,
        ppo_pretrain_cycles=args.pretrain_cycles,
        ppo_joint_training_cycles=args.joint_training_cycles,
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
    total_cycles = config.ppo_pretrain_cycles + config.ppo_joint_training_cycles
    total_training_slots = total_cycles * cycle_slots
    joint_adaptation_enabled = config.adaptation_enabled
    config.adaptation_enabled = (
        joint_adaptation_enabled and config.ppo_pretrain_cycles == 0
    )
    progress = ProgressReporter(
        args.progress_file, total_training_slots, unit="slots"
    )
    progress.update(0, item_count=0)
    metrics_path = args.output_dir / "training_metrics.csv"
    fields = (
        "episode", "phase", "cycle_index", "absolute_slot", "cycle_slot",
        "request_id", "template_id", "arrival_time_s", "chain_length",
        "success", "failure_reason", "failed_stage",
        "return", "latency_s", "energy_j", "steps", "relay_count",
        "route_slot_crossings", "route_phase_count", "migration_action_count",
        "no_op_count", "relocation_count", "scale_out_count", "scale_in_count",
        "ppo_update", "ppo_update_samples", "policy_loss", "value_loss", "entropy",
    )
    update_fields = (
        "ppo_update", "phase", "trigger_absolute_slot", "cycle_slot",
        "samples", "policy_loss", "value_loss", "entropy",
    )
    total_steps = 0
    processed_requests = 0
    last_arrival_time_s = 0.0
    ppo_update_count = 0
    last_ppo_update_trigger_slot = -1
    phase_request_counts = {"ppo_pretrain": 0, "joint_training": 0}
    current_cycle_index = 0
    current_phase = (
        "ppo_pretrain" if config.ppo_pretrain_cycles > 0 else "joint_training"
    )
    next_update_slot = config.ppo_update_interval_slots
    updates_path = args.output_dir / "ppo_update_metrics.csv"

    with (
        metrics_path.open("w", encoding="utf-8", newline="") as handle,
        updates_path.open("w", encoding="utf-8", newline="") as update_handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=fields)
        update_writer = csv.DictWriter(update_handle, fieldnames=update_fields)
        writer.writeheader()
        update_writer.writeheader()

        def update_ppo(trigger_slot: int, phase: str) -> tuple[dict, int]:
            nonlocal ppo_update_count, last_ppo_update_trigger_slot
            if trigger_slot == last_ppo_update_trigger_slot:
                return {}, 0
            sample_count = agent.eligible_transition_count(trigger_slot)
            if sample_count == 0:
                return {}, 0
            progress.update(
                min(trigger_slot, total_training_slots),
                item_count=processed_requests,
                phase="updating PPO",
            )
            losses = agent.update(None, current_slot=trigger_slot)
            ppo_update_count += 1
            last_ppo_update_trigger_slot = trigger_slot
            update_writer.writerow(
                {
                    "ppo_update": ppo_update_count,
                    "phase": phase,
                    "trigger_absolute_slot": trigger_slot,
                    "cycle_slot": trigger_slot % cycle_slots,
                    "samples": sample_count,
                    "policy_loss": losses.get("policy_loss", ""),
                    "value_loss": losses.get("value_loss", ""),
                    "entropy": losses.get("entropy", ""),
                }
            )
            update_handle.flush()
            progress.update(
                min(trigger_slot, total_training_slots),
                item_count=processed_requests,
                phase="processing requests",
            )
            return losses, sample_count

        for absolute_slot, slot_requests in environment.iter_request_batches(
            slot_count=total_training_slots
        ):
            request_cycle_index = min(
                total_cycles - 1, absolute_slot // cycle_slots
            )
            if request_cycle_index != current_cycle_index:
                previous_cycle_index = current_cycle_index
                boundary_slot = absolute_slot
                update_ppo(boundary_slot, current_phase)
                current_cycle_index = request_cycle_index
                current_phase = (
                    "ppo_pretrain"
                    if current_cycle_index < config.ppo_pretrain_cycles
                    else "joint_training"
                )
                if (
                    previous_cycle_index < config.ppo_pretrain_cycles
                    <= current_cycle_index
                    and config.ppo_pretrain_cycles > 0
                ):
                    # Do not carry pretraining trajectories into the changed
                    # placement distribution used by joint training.
                    agent.clear_transactions()
                    pretrain_boundary_time = (
                        config.ppo_pretrain_cycles * cycle_duration_s
                    )
                    environment.replica_adapter.start_fresh_window(
                        pretrain_boundary_time
                    )
                    agent.save(
                        args.output_dir / "ppo_pretrained.pt",
                        environment.control_state_dict(),
                    )
                config.adaptation_enabled = (
                    joint_adaptation_enabled and current_phase == "joint_training"
                )
                next_update_slot = (
                    current_cycle_index * cycle_slots
                    + config.ppo_update_interval_slots
                )

            cycle_slot = absolute_slot % cycle_slots
            completed_slots = absolute_slot + 1
            slot_rows = []
            sessions = environment.start_request_sessions(slot_requests)
            episode_base = processed_requests
            trackers = [
                {
                    "return": 0.0,
                    "steps": 0,
                    "final_info": {},
                    "route_slot_crossings": 0,
                    "route_phase_count": 0,
                    "transitions": [],
                }
                for _ in sessions
            ]
            active = list(range(len(sessions)))
            while active:
                decisions = agent.act_batch(
                    [sessions[index].last_state for index in active]
                )
                next_active = []
                for index, (action, log_prob, value) in zip(
                    active, decisions
                ):
                    environment.restore_request_session(sessions[index])
                    state = sessions[index].last_state
                    next_state, reward, terminated, truncated, info = (
                        environment.step(action)
                    )
                    transition = (
                        PPOTransition(
                            state,
                            action,
                            log_prob,
                            value,
                            reward,
                            terminated or truncated,
                            collection_slot=absolute_slot,
                        )
                    )
                    tracker = trackers[index]
                    tracker["transitions"].append(transition)
                    tracker["return"] += reward
                    tracker["steps"] += 1
                    total_steps += 1
                    tracker["final_info"] = info
                    for route_key in ("route", "final_route"):
                        route = info.get(route_key) or {}
                        tracker["route_slot_crossings"] += int(
                            route.get("slot_crossings", 0)
                        )
                        tracker["route_phase_count"] += len(
                            route.get("slot_phases", [])
                        )
                    sessions[index] = environment.capture_request_session()
                    if next_state is not None:
                        next_active.append(index)
                active = next_active

            # Batch inference interleaves stages from independent requests.
            # Restore the original trajectory-contiguous buffer order before
            # GAE is computed so one request can never bootstrap from another.
            for tracker in trackers:
                for transition in tracker["transitions"]:
                    agent.remember(transition)

            for local_episode, (request, tracker) in enumerate(
                zip(slot_requests, trackers)
            ):
                episode = episode_base + local_episode
                final_info = tracker["final_info"]
                steps = tracker["steps"]
                row = {
                    "episode": episode,
                    "phase": current_phase,
                    "cycle_index": current_cycle_index,
                    "absolute_slot": absolute_slot,
                    "cycle_slot": cycle_slot,
                    "request_id": final_info.get("request_id", ""),
                    "template_id": final_info.get("template_id", ""),
                    "arrival_time_s": final_info.get("arrival_time_s", ""),
                    "chain_length": final_info.get("chain_length", steps),
                    "success": int(bool(final_info.get("success"))),
                    "failure_reason": final_info.get("reason", ""),
                    "failed_stage": (
                        "" if final_info.get("success") else max(0, steps - 1)
                    ),
                    "return": tracker["return"],
                    "latency_s": final_info.get("total_latency_s", float("nan")),
                    "energy_j": final_info.get("total_energy_j", float("nan")),
                    "steps": steps,
                    "relay_count": final_info.get("relay_count", 0),
                    "route_slot_crossings": tracker["route_slot_crossings"],
                    "route_phase_count": tracker["route_phase_count"],
                    "migration_action_count": 0,
                    "no_op_count": 0,
                    "relocation_count": 0,
                    "scale_out_count": 0,
                    "scale_in_count": 0,
                    "ppo_update": "",
                    "ppo_update_samples": "",
                    "policy_loss": "",
                    "value_loss": "",
                    "entropy": "",
                }
                slot_rows.append(row)
                processed_requests += 1
                phase_request_counts[current_phase] += 1
                last_arrival_time_s = request.arrival_time_s

            # Placement changes occur only after every target request arriving
            # in this slot has completed. They affect the next slot, never a
            # peer target request from the current slot.
            environment.finalize_request_sessions()
            migration_actions = environment.finish_time_slot(absolute_slot)
            losses = {}
            update_samples = 0
            if completed_slots >= next_update_slot:
                losses, update_samples = update_ppo(
                    completed_slots, current_phase
                )
                while next_update_slot <= completed_slots:
                    next_update_slot += config.ppo_update_interval_slots

            if slot_rows:
                last_row = slot_rows[-1]
                last_row.update(
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
                    ppo_update=ppo_update_count if losses else "",
                    ppo_update_samples=update_samples if losses else "",
                    policy_loss=losses.get("policy_loss", ""),
                    value_loss=losses.get("value_loss", ""),
                    entropy=losses.get("entropy", ""),
                )
                writer.writerows(slot_rows)
                handle.flush()

            progress.update(completed_slots, item_count=processed_requests)
            if slot_rows and processed_requests % 10 == 0:
                print(
                    f"requests={processed_requests} phase={current_phase} "
                    f"slot={completed_slots}/{total_training_slots} "
                    f"steps={total_steps} "
                    f"return={slot_rows[-1]['return']:.4f} "
                    f"latency={slot_rows[-1]['latency_s']:.4f}s"
                )
            if slot_rows and processed_requests % 50 == 0:
                agent.save(
                    args.output_dir / "ppo_latest.pt",
                    environment.control_state_dict(),
                )
        update_ppo(total_training_slots, current_phase)
    if config.ppo_pretrain_cycles > 0 and not (
        args.output_dir / "ppo_pretrained.pt"
    ).is_file():
        environment.replica_adapter.start_fresh_window(
            config.ppo_pretrain_cycles * cycle_duration_s
        )
        agent.save(
            args.output_dir / "ppo_pretrained.pt",
            environment.control_state_dict(),
        )
    agent.save(args.output_dir / "ppo_final.pt", environment.control_state_dict())
    with (args.output_dir / "orchestration_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "constellation_cycle_slots": cycle_slots,
                "constellation_cycle_duration_s": cycle_duration_s,
                "pretrain_cycles": config.ppo_pretrain_cycles,
                "joint_training_cycles": config.ppo_joint_training_cycles,
                "joint_adaptation_enabled": joint_adaptation_enabled,
                "total_training_slots": total_training_slots,
                "ppo_update_interval_slots": config.ppo_update_interval_slots,
                "ppo_transaction_history_slots": (
                    config.ppo_transaction_history_slots
                ),
                "ppo_transaction_max_reuse": config.ppo_transaction_max_reuse,
                "ppo_update_count": ppo_update_count,
                "processed_request_count": processed_requests,
                "phase_request_counts": phase_request_counts,
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
    progress.update(
        total_training_slots, status="succeeded", item_count=processed_requests
    )
    print(
        f"training complete: requests={processed_requests} "
        f"slots={total_training_slots} updates={ppo_update_count} "
        f"checkpoint={args.output_dir / 'ppo_final.pt'}"
    )


if __name__ == "__main__":
    main()
