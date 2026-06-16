from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path

from ..ablation_names import ABLATION_NAME_MAP, OFFICIAL_ABLATIONS, canonical_ablation_name
from ..agents.migration import ReplicaPlacementMigrationAgent
from ..core.env import SimulationEnvironment
from ..core.metrics import summarize_results
from ..domain.constellation import node_id_to_sat_name
from ..domain.request import SFCRequest
from .full_cycle_seed_distribution import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_OUTPUT_DIR,
    build_config,
    build_execution_agent,
    canonicalize_ablation_row,
    delay_margin_summary,
    filter_templates_by_chain_length,
    generate_arrivals_for_slot,
    jsonable,
    load_or_generate_templates,
    parse_seed_list,
    read_csv,
    request_hop_metric_rows,
    request_metric_row,
    resolve_isl_csv_path,
    route_failure_count,
    route_mode_counts_for_results,
    run_dir_for_seed,
    total_arrival_lambda,
    write_csv,
)


DEFAULT_OUTPUT_ROOT = (
    DEFAULT_OUTPUT_DIR.parent / "bandit_redeployment_replay_experiments"
)
DEFAULT_REDEPLOY_ABLATIONS = "ELARA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an identical 10-slot request trace before and after a "
            "service redeployment action to measure bandit redeployment impact."
        )
    )
    parser.add_argument("--seeds", default="41 42 43 44")
    parser.add_argument(
        "--ablation",
        choices=tuple(dict.fromkeys((*OFFICIAL_ABLATIONS, *ABLATION_NAME_MAP))),
        default="ELARA",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--model-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--request-template-csv", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default="ppo_gnn_latest.pth")
    parser.add_argument("--bandit-stats-name", default="bandit_arm_stats.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--isl-csv", type=Path, default=None)
    parser.add_argument("--arrival-lambda", type=float, default=None)
    parser.add_argument(
        "--arrival-mode",
        choices=("per_template", "total_per_slot"),
        default="per_template",
    )
    parser.add_argument("--total-arrival-lambda", type=float, default=None)
    parser.add_argument("--chain-length-filter", type=int, choices=(5, 10, 15), default=None)
    parser.add_argument("--window-slots", type=int, default=10)
    parser.add_argument("--start-slot", type=int, default=0)
    parser.add_argument("--max-slots", type=int, default=None)
    parser.add_argument("--no-load-checkpoint", action="store_true")
    parser.add_argument("--no-load-bandit", action="store_true")
    parser.add_argument(
        "--redeploy-ablations",
        default=DEFAULT_REDEPLOY_ABLATIONS,
        help=(
            "Ablations that execute bandit-based service redeployment between "
            "the before and after replay windows. Defaults to ELARA only."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.ablation = canonical_ablation_name(args.ablation)
    args.redeploy_ablations = {
        canonical_ablation_name(item)
        for item in str(args.redeploy_ablations).replace(",", " ").split()
        if item
    }
    if args.window_slots <= 0:
        raise SystemExit("--window-slots must be positive.")
    return args


def clone_request_for_replay(
    request: SFCRequest,
    original_absolute_slot: int,
    replay_absolute_slot: int,
    slot_duration: float,
    slot_count: int,
) -> SFCRequest:
    offset = max(0.0, request.start_time - original_absolute_slot * slot_duration)
    return replace(
        request,
        start_time=replay_absolute_slot * slot_duration + offset,
        start_slot=replay_absolute_slot % slot_count,
    )


def request_trace_rows(
    seed: int,
    ablation: str,
    before_slots: list[list[SFCRequest]],
    slot_duration: float,
) -> list[dict]:
    rows = []
    for slot_index, requests in enumerate(before_slots, start=1):
        original_absolute_slot = slot_index - 1
        for request in requests:
            rows.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "slot_index": slot_index,
                    "original_absolute_slot": original_absolute_slot,
                    "request_id": request.request_id,
                    "template_id": request.template_id,
                    "chain_length": len(request.services),
                    "services": list(request.services),
                    "start_offset_s": request.start_time - original_absolute_slot * slot_duration,
                    "source_node": request.source_node,
                    "source_satellite": node_id_to_sat_name(request.source_node),
                    "destination_node": request.destination_node,
                    "destination_satellite": node_id_to_sat_name(request.destination_node),
                    "input_data_gb": request.input_data_gb,
                    "data_gb_between_services": list(request.data_gb_between_services),
                    "output_data_gb": request.output_data_gb,
                }
            )
    return rows


def phase_summary(results: list[dict]) -> dict:
    summary = summarize_results(results)
    feasible = [result for result in results if result.get("feasible", False)]
    return {
        "request_count": summary["request_count"],
        "feasible_count": summary["feasible_count"],
        "failure_count": max(0, summary["request_count"] - summary["feasible_count"]),
        "success_rate": summary["success_rate"],
        "average_end_to_end_delay_s": summary["average_end_to_end_delay_s"],
        "p95_end_to_end_delay_s": summary["p95_end_to_end_delay_s"],
        "average_energy_j": summary["average_energy_j"],
        "average_communication_delay_s": summary["average_communication_delay_s"],
        "average_slot_crossings": summary["average_slot_crossings"],
        "feasible_request_count_for_average": len(feasible),
    }


def delta(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after):
        return math.nan
    return before - after


def reduction_ratio(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or abs(before) <= 1.0e-12:
        return math.nan
    return (before - after) / before


def replay_phase(
    *,
    phase: str,
    seed: int,
    ablation: str,
    env: SimulationEnvironment,
    agent,
    slot_requests: list[list[SFCRequest]],
    slot_offset: int,
    cycle: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    slot_rows: list[dict] = []
    request_rows: list[dict] = []
    hop_rows: list[dict] = []
    all_results: list[dict] = []
    slot_duration = env.context["slot_duration"]
    slot_count = env.context["slot_count"]

    for slot_index, requests in enumerate(slot_requests, start=1):
        absolute_slot = slot_offset + slot_index - 1
        slot_mod = absolute_slot % slot_count
        results = env.execute_requests(requests, agent)
        all_results.extend(results)
        summary = summarize_results(results)
        route_mode_counts = route_mode_counts_for_results(results)
        total_reward = sum(float(result.get("reward", 0.0)) for result in results)
        row = {
            "ablation": ablation,
            "seed": seed,
            "phase": phase,
            "cycle": cycle,
            "slot_index": slot_index,
            "absolute_slot": absolute_slot,
            "slot_mod": slot_mod,
            "slot_start_time_s": absolute_slot * slot_duration,
            "slot_duration_s": slot_duration,
            "arrival_count": len(requests),
            "request_count": summary["request_count"],
            "feasible_count": summary["feasible_count"],
            "failure_count": route_failure_count(results),
            "success_rate": summary["success_rate"],
            "average_end_to_end_delay_s": summary["average_end_to_end_delay_s"],
            "p95_end_to_end_delay_s": summary["p95_end_to_end_delay_s"],
            "average_energy_j": summary["average_energy_j"],
            "average_communication_delay_s": summary["average_communication_delay_s"],
            "average_slot_crossings": summary["average_slot_crossings"],
            "total_reward": total_reward,
            "average_reward_per_request": total_reward / len(results) if results else None,
            "service_routing_strategy": env.config.service_routing_strategy,
            "execution_agent": agent.__class__.__name__,
            **{
                f"route_mode_{mode}_count": count
                for mode, count in sorted(route_mode_counts.items())
                if mode
            },
        }
        slot_rows.append(row)
        for result in results:
            request_row = request_metric_row(ablation, seed, cycle, absolute_slot + 1, slot_mod, result)
            request_row["phase"] = phase
            request_row["slot_index"] = slot_index
            request_row["absolute_slot"] = absolute_slot
            request_rows.append(request_row)
            for hop_row in request_hop_metric_rows(ablation, seed, cycle, absolute_slot + 1, slot_mod, result):
                hop_row["phase"] = phase
                hop_row["slot_index"] = slot_index
                hop_row["absolute_slot"] = absolute_slot
                hop_rows.append(hop_row)
        env.context["routing_cache"]["route_results"].clear()
        env.context.get("route_estimate_cache", {}).clear()

    return slot_rows, request_rows, hop_rows, all_results


def run_seed(
    args: argparse.Namespace,
    seed: int,
    index: int,
    seeds: list[int],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    run_dir = run_dir_for_seed(args, seeds, seed, index)
    seed_output_dir = args.output_dir / f"seed_{seed}"
    config = build_config(args, seed, run_dir, seed_output_dir)
    if args.max_slots is not None:
        min_slots = args.start_slot + 2 * args.window_slots
        if args.max_slots < min_slots:
            raise SystemExit(
                f"--max-slots must be at least start-slot + 2*window-slots "
                f"({min_slots}) for this replay experiment."
            )

    bandit_agent = ReplicaPlacementMigrationAgent(config)
    bandit_stats = run_dir / args.bandit_stats_name
    redeployment_enabled = args.ablation in args.redeploy_ablations
    bandit_loaded_count = 0
    if redeployment_enabled and not args.no_load_bandit and bandit_stats.exists():
        bandit_loaded_count = bandit_agent.load_arm_stats(bandit_stats)

    env = SimulationEnvironment(
        config,
        migration_agent=bandit_agent,
        auto_generate_requests=False,
        auto_apply_migration=False,
    ).build()
    slot_count = env.context["slot_count"]
    slot_duration = env.context["slot_duration"]
    arrival_rng = random.Random(seed + 20_000)
    templates, template_csv, templates_loaded = load_or_generate_templates(
        args, run_dir, arrival_rng, env.context
    )
    original_template_count = len(templates)
    templates = filter_templates_by_chain_length(templates, args.chain_length_filter)

    agent, checkpoint_loaded, checkpoint_load_error = build_execution_agent(
        args, config, run_dir
    )

    before_slots: list[list[SFCRequest]] = []
    next_request_id = 1
    for slot_offset in range(args.window_slots):
        absolute_slot = args.start_slot + slot_offset
        arrivals, next_request_id, _arrival_info = generate_arrivals_for_slot(
            args, arrival_rng, env.context, templates, absolute_slot, next_request_id
        )
        before_slots.append(arrivals)

    trace_rows = request_trace_rows(seed, args.ablation, before_slots, slot_duration)
    before_slot_rows, before_request_rows, before_hop_rows, before_results = replay_phase(
        phase="before",
        seed=seed,
        ablation=args.ablation,
        env=env,
        agent=agent,
        slot_requests=before_slots,
        slot_offset=args.start_slot,
        cycle=1,
    )

    migration_actions = []
    if redeployment_enabled:
        bandit_agent.observe_failed_replicas(before_results)
        bandit_agent.observe_service_pressure_feedback(before_results)
        migration_actions = env.apply_migration(
            [request for slot_requests in before_slots for request in slot_requests]
        )

    replay_slots: list[list[SFCRequest]] = []
    for slot_offset, requests in enumerate(before_slots):
        original_absolute_slot = args.start_slot + slot_offset
        replay_absolute_slot = args.start_slot + args.window_slots + slot_offset
        replay_slots.append(
            [
                clone_request_for_replay(
                    request,
                    original_absolute_slot,
                    replay_absolute_slot,
                    slot_duration,
                    slot_count,
                )
                for request in requests
            ]
        )

    after_slot_rows, after_request_rows, after_hop_rows, after_results = replay_phase(
        phase="after",
        seed=seed,
        ablation=args.ablation,
        env=env,
        agent=agent,
        slot_requests=replay_slots,
        slot_offset=args.start_slot + args.window_slots,
        cycle=1,
    )

    before_summary = phase_summary(before_results)
    after_summary = phase_summary(after_results)
    delay_before = float(before_summary["average_end_to_end_delay_s"])
    delay_after = float(after_summary["average_end_to_end_delay_s"])
    energy_before = float(before_summary["average_energy_j"])
    energy_after = float(after_summary["average_energy_j"])
    summary_row = {
        "ablation": args.ablation,
        "seed": seed,
        "window_slots": args.window_slots,
        "start_slot": args.start_slot,
        "request_count_before": before_summary["request_count"],
        "request_count_after": after_summary["request_count"],
        "same_request_trace": True,
        "redeployment_enabled": redeployment_enabled,
        "redeployment_action_count": len(migration_actions),
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_load_error": checkpoint_load_error,
        "bandit_loaded_arm_count": bandit_loaded_count,
        "execution_agent": agent.__class__.__name__,
        "service_routing_strategy": config.service_routing_strategy,
        "before_success_rate": before_summary["success_rate"],
        "after_success_rate": after_summary["success_rate"],
        "before_average_end_to_end_delay_s": delay_before,
        "after_average_end_to_end_delay_s": delay_after,
        "delay_reduction_s": delta(delay_before, delay_after),
        "delay_reduction_ratio": reduction_ratio(delay_before, delay_after),
        "before_p95_end_to_end_delay_s": before_summary["p95_end_to_end_delay_s"],
        "after_p95_end_to_end_delay_s": after_summary["p95_end_to_end_delay_s"],
        "before_average_energy_j": energy_before,
        "after_average_energy_j": energy_after,
        "energy_reduction_j": delta(energy_before, energy_after),
        "energy_reduction_ratio": reduction_ratio(energy_before, energy_after),
        "before_average_communication_delay_s": before_summary["average_communication_delay_s"],
        "after_average_communication_delay_s": after_summary["average_communication_delay_s"],
        "before_average_slot_crossings": before_summary["average_slot_crossings"],
        "after_average_slot_crossings": after_summary["average_slot_crossings"],
    }
    summary = {
        **summary_row,
        "run_dir": run_dir,
        "request_template_csv": template_csv,
        "request_templates_loaded": templates_loaded,
        "request_template_count": len(templates),
        "original_request_template_count": original_template_count,
        "arrival_mode": args.arrival_mode,
        "arrival_lambda_total_per_slot": (
            total_arrival_lambda(args, config, len(templates))
            if args.arrival_mode == "total_per_slot"
            else None
        ),
        "before_delay_margin_summary": delay_margin_summary(before_results),
        "after_delay_margin_summary": delay_margin_summary(after_results),
        "bandit_summary": bandit_agent.summary(),
    }

    slot_rows = before_slot_rows + after_slot_rows
    request_rows = before_request_rows + after_request_rows
    hop_rows = before_hop_rows + after_hop_rows
    migration_rows = [asdict(action) for action in migration_actions]

    write_csv(seed_output_dir / "request_trace.csv", trace_rows)
    write_csv(seed_output_dir / "window_slot_metrics.csv", slot_rows)
    write_csv(seed_output_dir / "window_request_metrics.csv", request_rows)
    write_csv(seed_output_dir / "window_request_hop_metrics.csv", hop_rows)
    write_csv(seed_output_dir / "redeployment_summary_metrics.csv", [summary_row])
    write_csv(seed_output_dir / "migration_actions.csv", migration_rows)
    (seed_output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return trace_rows, slot_rows, request_rows, hop_rows, [summary_row], summary


def load_seed_outputs(
    output_dir: Path, seeds: list[int]
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    trace_rows = []
    slot_rows = []
    request_rows = []
    hop_rows = []
    summary_rows = []
    summaries = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        trace_rows.extend(read_csv(seed_dir / "request_trace.csv"))
        slot_rows.extend(read_csv(seed_dir / "window_slot_metrics.csv"))
        request_rows.extend(read_csv(seed_dir / "window_request_metrics.csv"))
        hop_rows.extend(read_csv(seed_dir / "window_request_hop_metrics.csv"))
        summary_rows.extend(read_csv(seed_dir / "redeployment_summary_metrics.csv"))
        summary_path = seed_dir / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not summary_rows:
        raise SystemExit(f"No seed outputs found under {output_dir}")
    return trace_rows, slot_rows, request_rows, hop_rows, summary_rows, summaries


def aggregate_outputs(
    args: argparse.Namespace,
    seeds: list[int],
    trace_rows: list[dict],
    slot_rows: list[dict],
    request_rows: list[dict],
    hop_rows: list[dict],
    summary_rows: list[dict],
    summaries: list[dict],
) -> dict:
    for rows in (trace_rows, slot_rows, request_rows, hop_rows, summary_rows):
        for row in rows:
            canonicalize_ablation_row(row)
    write_csv(args.output_dir / "request_trace_by_seed.csv", trace_rows)
    write_csv(args.output_dir / "window_slot_metrics_by_seed.csv", slot_rows)
    write_csv(args.output_dir / "window_request_metrics_by_seed.csv", request_rows)
    write_csv(args.output_dir / "window_request_hop_metrics_by_seed.csv", hop_rows)
    write_csv(args.output_dir / "redeployment_summary_by_seed.csv", summary_rows)
    final_summary = {
        "ablation": args.ablation,
        "seeds": seeds,
        "output_dir": args.output_dir,
        "request_trace_csv": args.output_dir / "request_trace_by_seed.csv",
        "window_slot_metrics_csv": args.output_dir / "window_slot_metrics_by_seed.csv",
        "window_request_metrics_csv": args.output_dir / "window_request_metrics_by_seed.csv",
        "window_request_hop_metrics_csv": args.output_dir / "window_request_hop_metrics_by_seed.csv",
        "redeployment_summary_csv": args.output_dir / "redeployment_summary_by_seed.csv",
        "seed_summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_summary


def main() -> None:
    args = parse_args()
    args.isl_csv = resolve_isl_csv_path(args.isl_csv)
    seeds = parse_seed_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        final_summary = aggregate_outputs(
            args,
            seeds,
            *load_seed_outputs(args.output_dir, seeds),
        )
        print(json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2))
        return

    all_trace_rows = []
    all_slot_rows = []
    all_request_rows = []
    all_hop_rows = []
    all_summary_rows = []
    summaries = []
    for index, seed in enumerate(seeds):
        trace_rows, slot_rows, request_rows, hop_rows, summary_rows, summary = run_seed(
            args, seed, index, seeds
        )
        all_trace_rows.extend(trace_rows)
        all_slot_rows.extend(slot_rows)
        all_request_rows.extend(request_rows)
        all_hop_rows.extend(hop_rows)
        all_summary_rows.extend(summary_rows)
        summaries.append(summary)

    if args.skip_aggregate:
        print(json.dumps(jsonable({"seeds": seeds, "seed_summaries": summaries}), ensure_ascii=False, indent=2))
        return

    final_summary = aggregate_outputs(
        args,
        seeds,
        all_trace_rows,
        all_slot_rows,
        all_request_rows,
        all_hop_rows,
        all_summary_rows,
        summaries,
    )
    print(json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
