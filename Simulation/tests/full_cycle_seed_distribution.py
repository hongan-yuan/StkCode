from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, fields, is_dataclass
from html import escape
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on local plotting env
    plt = None

from ..agents.migration import ReplicaPlacementMigrationAgent
from ..agents.baselines import (
    NearestReplicaExecutionAgent,
    SCNFVChainingOrbitExecutionAgent,
    ServicePressureExecutionAgent,
)
from ..agents.ppo_gnn_agent import PPOGNNExecutionAgent
from ..config import SimulationConfig
from ..core.env import SimulationEnvironment
from ..core.metrics import summarize_results
from ..domain.constellation import node_id_to_sat_name
from ..domain.request import (
    SFCRequest,
    generate_request_templates,
    generate_slot_arrivals,
    generate_slot_arrivals_total_poisson,
    load_request_templates,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = ROOT_DIR / "Simulation" / "multi_seed_runs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "full_cycle_seed_distribution"
DEFAULT_ISL_CSV = ROOT_DIR / "WalkerDeltaConstellationSimu" / "Walker_Delta_ISL_Simu.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full-satellite-cycle test for multiple seeds and plot the "
            "slot-mean delay/energy distributions grouped by seed."
        )
    )
    parser.add_argument("--seeds", default="42 43 44 45")
    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "no_bandit",
            "shortest_hop_routing",
            "nearest_replica",
            "service_pressure",
            "sc_nfv",
        ),
        default="full",
        help="Evaluation variant to run.",
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Explicit seed run directory. Can be repeated. When omitted, "
            "--model-root/seed_<seed> is used."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--request-template-csv",
        type=Path,
        default=None,
        help=(
            "Optional global request_templates.csv. If omitted, each seed uses "
            "its own training request_templates.csv when available."
        ),
    )
    parser.add_argument("--checkpoint-name", default="ppo_gnn_latest.pth")
    parser.add_argument("--bandit-stats-name", default="bandit_arm_stats.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--isl-csv", type=Path, default=None)
    parser.add_argument(
        "--arrival-lambda",
        type=float,
        default=None,
        help=(
            "Override the Poisson lambda per request-chain template per slot. "
            "By default the value from training_config.json or SimulationConfig is used."
        ),
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("per_template", "total_per_slot"),
        default="per_template",
        help=(
            "per_template samples one Poisson arrival count per request template. "
            "total_per_slot samples one total Poisson count per slot and then "
            "draws templates uniformly from the selected template pool."
        ),
    )
    parser.add_argument(
        "--total-arrival-lambda",
        type=float,
        default=None,
        help=(
            "Poisson lambda for total arrivals per slot when --arrival-mode "
            "total_per_slot is used. Defaults to per-template lambda times the "
            "selected template count."
        ),
    )
    parser.add_argument(
        "--chain-length-filter",
        type=int,
        choices=(5, 10, 15),
        default=None,
        help="Restrict evaluation to request templates with this chain length.",
    )
    parser.add_argument(
        "--bandit-period-slots",
        type=int,
        default=10,
        help="Apply Bandit migration every N slots, matching train.py default.",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Smoke-test slot limit. Omit this argument to cover a full satellite period.",
    )
    parser.add_argument("--no-load-checkpoint", action="store_true")
    parser.add_argument("--no-load-bandit", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print per-task progress every N slots.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run seed simulations but do not write root-level merged CSV/plots.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing seed_<seed> outputs and only regenerate merged CSV/plots.",
    )
    return parser.parse_args()


class TaskProgress:
    def __init__(self, label: str, total: int, every: int):
        self.label = label
        self.total = max(1, int(total))
        self.every = max(1, int(every))
        self.started_at = time.monotonic()
        self.last_line_length = 0
        self.is_tty = sys.stderr.isatty()

    def update(self, current: int, suffix: str = "") -> None:
        if current != 1 and current != self.total and current % self.every != 0:
            return
        elapsed = time.monotonic() - self.started_at
        progress = min(1.0, max(0.0, current / self.total))
        eta = elapsed * (1.0 - progress) / progress if progress > 0.0 else math.inf
        line = (
            f"{self.label} [{self._bar(progress)}] {progress * 100:6.2f}% "
            f"{current}/{self.total} elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta)} {suffix}".rstrip()
        )
        if self.is_tty:
            padding = " " * max(0, self.last_line_length - len(line))
            sys.stderr.write("\r" + line + padding)
            sys.stderr.flush()
            self.last_line_length = len(line)
            if current >= self.total:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self.last_line_length = 0
        else:
            print(line, file=sys.stderr, flush=True)

    def _bar(self, progress: float) -> str:
        width = 24
        filled = int(round(progress * width))
        return "#" * filled + "-" * (width - filled)


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item) for item in value.replace(",", " ").split() if item]
    if not seeds:
        raise SystemExit("At least one seed is required.")
    return seeds


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value):
        return jsonable(asdict(value))
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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def run_dir_for_seed(args: argparse.Namespace, seeds: list[int], seed: int, index: int) -> Path:
    if args.model_dir:
        if len(args.model_dir) == 1:
            return args.model_dir[0]
        if len(args.model_dir) != len(seeds):
            raise SystemExit("--model-dir must be provided once or once per seed.")
        return args.model_dir[index]
    return args.model_root / f"seed_{seed}"


def convert_tuple_like(value):
    if isinstance(value, list):
        return tuple(convert_tuple_like(item) for item in value)
    return value


def config_kwargs_from_training_json(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    valid_names = {field.name for field in fields(SimulationConfig)}
    tuple_names = {
        "replica_count_range",
        "microservice_workload_range_cycles",
        "microservice_image_size_range_gb",
        "microservice_storage_range_gb",
        "cpu_freq_choices_ghz",
        "cpu_discount_range",
        "compute_load_states",
        "request_chain_plan",
        "compute_load_utilization_ranges",
        "compute_load_discount_ranges",
    }
    kwargs = {}
    for key, value in raw.items():
        if key not in valid_names:
            continue
        if key in {"isl_log_csv", "output_dir"} and value:
            kwargs[key] = Path(value)
        elif key == "cpu_power_by_freq_w" and isinstance(value, dict):
            kwargs[key] = {float(k): float(v) for k, v in value.items()}
        elif key in tuple_names:
            kwargs[key] = convert_tuple_like(value)
        else:
            kwargs[key] = value
    return kwargs


def resolve_isl_csv_path(path: Path | None) -> Path:
    if path is None:
        return DEFAULT_ISL_CSV
    path = Path(path)
    if path.exists():
        return path
    if path.name == DEFAULT_ISL_CSV.name and DEFAULT_ISL_CSV.exists():
        return DEFAULT_ISL_CSV
    return path


def build_config(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    output_dir: Path,
) -> SimulationConfig:
    kwargs = config_kwargs_from_training_json(run_dir / "training_config.json")
    isl_csv = resolve_isl_csv_path(
        args.isl_csv or kwargs.get("isl_log_csv", SimulationConfig().isl_log_csv)
    )
    kwargs.update(
        {
            "random_seed": seed,
            "isl_log_csv": isl_csv,
            "max_slots_to_load": args.max_slots,
            "process_single_request": False,
            "output_dir": output_dir,
        }
    )
    if args.arrival_lambda is not None:
        kwargs["request_arrival_lambda_per_pattern_per_slot"] = args.arrival_lambda
    if args.ablation == "shortest_hop_routing":
        kwargs["service_routing_strategy"] = "shortest_hop_per_slot"
    elif args.ablation == "service_pressure":
        kwargs["service_routing_strategy"] = "service_pressure"
    else:
        kwargs["service_routing_strategy"] = "min_cost_max_flow"
    return SimulationConfig(**kwargs)


def load_or_generate_templates(
    args: argparse.Namespace,
    run_dir: Path,
    rng: random.Random,
    context: dict,
) -> tuple[list[SFCRequest], Path | None, bool]:
    template_csv = args.request_template_csv or run_dir / "request_templates.csv"
    if template_csv.exists():
        return load_request_templates(template_csv), template_csv, True
    return generate_request_templates(rng, context), None, False


def filter_templates_by_chain_length(
    templates: list[SFCRequest],
    chain_length: int | None,
) -> list[SFCRequest]:
    if chain_length is None:
        return list(templates)
    filtered = [
        template for template in templates if len(template.services) == chain_length
    ]
    if not filtered:
        raise SystemExit(f"No request templates found with chain length {chain_length}.")
    return filtered


def total_arrival_lambda(
    args: argparse.Namespace,
    config: SimulationConfig,
    template_count: int,
) -> float:
    if args.total_arrival_lambda is not None:
        return args.total_arrival_lambda
    return config.request_arrival_lambda_per_pattern_per_slot * template_count


def generate_arrivals_for_slot(
    args: argparse.Namespace,
    rng: random.Random,
    context: dict,
    templates: list[SFCRequest],
    absolute_slot: int,
    next_request_id: int,
) -> tuple[list[SFCRequest], int, dict]:
    config: SimulationConfig = context["config"]
    if args.arrival_mode == "total_per_slot":
        return generate_slot_arrivals_total_poisson(
            rng,
            context,
            templates,
            absolute_slot,
            next_request_id,
            total_arrival_lambda(args, config, len(templates)),
        )
    return generate_slot_arrivals(
        rng,
        context,
        templates,
        absolute_slot,
        next_request_id,
    )


def request_metric_row(
    ablation: str,
    seed: int,
    epoch: int,
    slot_mod: int,
    result: dict,
) -> dict:
    request = result["request"]
    return {
        "ablation": ablation,
        "seed": seed,
        "epoch": epoch,
        "slot_mod": slot_mod,
        "template_id": request.get("template_id"),
        "request_id": request["request_id"],
        "chain_length": len(request["services"]),
        "source_node": request["source_node"],
        "source_satellite": node_id_to_sat_name(request["source_node"]),
        "destination_node": request["destination_node"],
        "destination_satellite": node_id_to_sat_name(request["destination_node"]),
        "feasible": result["feasible"],
        "failed_stage": result["failed_stage"],
        "failure_reason": result.get("failure_reason", ""),
        "total_delay_s": (
            result["total_delay_s"] if math.isfinite(result["total_delay_s"]) else ""
        ),
        "total_energy_j": (
            result["total_energy_j"] if math.isfinite(result["total_energy_j"]) else ""
        ),
        "reward": result["reward"],
    }


def flatten_route_counts(route_mode_counts: dict) -> dict:
    return {
        f"route_mode_{mode}_count": count
        for mode, count in sorted(route_mode_counts.items())
        if mode
    }


def cycle_metric_row(summary: dict) -> dict:
    overall = summary.get("overall_summary", {})
    route_mode_counts = overall.get("route_mode_counts", {})
    request_count = int(overall.get("request_count") or 0)
    feasible_count = int(overall.get("feasible_count") or 0)
    task_completion_rate = (
        feasible_count / request_count if request_count > 0 else 0.0
    )
    return {
        "ablation": summary.get("ablation", ""),
        "seed": summary.get("seed", ""),
        "chain_length_filter": summary.get("chain_length_filter", ""),
        "arrival_mode": summary.get("arrival_mode", ""),
        "arrival_lambda_total_per_slot": summary.get(
            "arrival_lambda_total_per_slot", ""
        ),
        "template_count": summary.get("request_template_count", ""),
        "slot_count": summary.get("slot_count", ""),
        "request_count": request_count,
        "feasible_count": feasible_count,
        "failure_count": max(0, request_count - feasible_count),
        "success_rate": task_completion_rate,
        "task_completion_rate": task_completion_rate,
        "average_end_to_end_delay_s": overall.get("average_end_to_end_delay_s", ""),
        "average_energy_j": overall.get("average_energy_j", ""),
        "p95_end_to_end_delay_s": overall.get("p95_end_to_end_delay_s", ""),
        "average_communication_delay_s": overall.get(
            "average_communication_delay_s", ""
        ),
        "average_slot_crossings": overall.get("average_slot_crossings", ""),
        **flatten_route_counts(route_mode_counts),
    }


def route_failure_count(results: list[dict]) -> int:
    return sum(1 for result in results if not result.get("feasible", False))


def route_mode_counts_for_results(results: list[dict]) -> Counter:
    counts = Counter()
    for result in results:
        for route in result.get("route_details", []):
            mode = route.get("route_mode", "")
            if mode:
                counts[mode] += 1
    return counts


def build_execution_agent(args: argparse.Namespace, config: SimulationConfig, run_dir: Path):
    if args.ablation == "nearest_replica":
        return NearestReplicaExecutionAgent(config), False, ""
    if args.ablation == "service_pressure":
        return ServicePressureExecutionAgent(config), False, ""
    if args.ablation == "sc_nfv":
        return SCNFVChainingOrbitExecutionAgent(config), False, ""

    agent = PPOGNNExecutionAgent(
        config,
        hidden_dim=config.ppo_hidden_dim,
        train_mode=False,
        device=args.device,
    )
    checkpoint = run_dir / args.checkpoint_name
    checkpoint_loaded = False
    checkpoint_load_error = ""
    if not args.no_load_checkpoint and checkpoint.exists():
        try:
            agent.load(checkpoint)
            checkpoint_loaded = True
        except Exception as exc:
            checkpoint_load_error = f"{type(exc).__name__}: {exc}"
    return agent, checkpoint_loaded, checkpoint_load_error


def finite_values(values: list[float | None]) -> list[float]:
    finite = []
    for value in values:
        if value in ("", None, "None", "null"):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return finite


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return math.inf
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def plot_distribution_matplotlib(
    output_path: Path,
    seeds: list[int],
    values_by_seed: dict[int, list[float]],
    ylabel: str,
    title: str,
) -> None:
    data = [values_by_seed.get(seed, []) for seed in seeds]
    fig, ax = plt.subplots(figsize=(max(7.0, len(seeds) * 1.3), 4.8))
    ax.boxplot(data, labels=[str(seed) for seed in seeds], showmeans=True)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution_svg(
    output_path: Path,
    seeds: list[int],
    values_by_seed: dict[int, list[float]],
    ylabel: str,
    title: str,
) -> None:
    width = max(760, 120 * len(seeds))
    height = 480
    margin_left, margin_right, margin_top, margin_bottom = 78, 32, 52, 74
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_values = [value for seed in seeds for value in values_by_seed.get(seed, [])]
    if not all_values:
        all_values = [0.0, 1.0]
    y_min = min(all_values)
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    def sx(index: int) -> float:
        return margin_left + (index + 0.5) * plot_w / max(1, len(seeds))

    def sy(value: float) -> float:
        return margin_top + (y_max - value) * plot_h / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">Seed</text>',
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">{escape(ylabel)}</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.3g}</text>')
    for index, seed in enumerate(seeds):
        values = sorted(values_by_seed.get(seed, []))
        x = sx(index)
        parts.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{seed}</text>')
        if not values:
            continue
        q1 = quantile(values, 0.25)
        median = quantile(values, 0.50)
        q3 = quantile(values, 0.75)
        low = values[0]
        high = values[-1]
        mean = sum(values) / len(values)
        box_w = min(56, plot_w / max(1, len(seeds)) * 0.48)
        parts.append(f'<line x1="{x:.1f}" y1="{sy(low):.1f}" x2="{x:.1f}" y2="{sy(high):.1f}" stroke="#475569" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x - box_w / 3:.1f}" y1="{sy(low):.1f}" x2="{x + box_w / 3:.1f}" y2="{sy(low):.1f}" stroke="#475569" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x - box_w / 3:.1f}" y1="{sy(high):.1f}" x2="{x + box_w / 3:.1f}" y2="{sy(high):.1f}" stroke="#475569" stroke-width="1.5"/>')
        parts.append(f'<rect x="{x - box_w / 2:.1f}" y="{sy(q3):.1f}" width="{box_w:.1f}" height="{max(1.0, sy(q1) - sy(q3)):.1f}" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x - box_w / 2:.1f}" y1="{sy(median):.1f}" x2="{x + box_w / 2:.1f}" y2="{sy(median):.1f}" stroke="#1e3a8a" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{sy(mean):.1f}" r="3.2" fill="#dc2626"/>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def plot_distribution(
    output_dir: Path,
    seeds: list[int],
    rows: list[dict],
    column: str,
    ylabel: str,
    filename_stem: str,
    title: str,
) -> Path:
    values_by_seed = {
        seed: finite_values(
            [
                row.get(column)
                for row in rows
                if int(row.get("seed", -1)) == seed
            ]
        )
        for seed in seeds
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None:
        path = output_dir / f"{filename_stem}.png"
        plot_distribution_matplotlib(path, seeds, values_by_seed, ylabel, title)
    else:
        path = output_dir / f"{filename_stem}.svg"
        plot_distribution_svg(path, seeds, values_by_seed, ylabel, title)
    return path


def aggregate_outputs(
    args: argparse.Namespace,
    seeds: list[int],
    all_slot_rows: list[dict],
    all_request_rows: list[dict],
    summaries: list[dict],
) -> dict:
    write_csv(args.output_dir / "slot_metrics_by_seed.csv", all_slot_rows)
    write_csv(args.output_dir / "request_metrics_by_seed.csv", all_request_rows)
    cycle_rows = [cycle_metric_row(summary) for summary in summaries]
    write_csv(args.output_dir / "cycle_metrics_by_seed.csv", cycle_rows)
    delay_plot = plot_distribution(
        args.output_dir,
        seeds,
        all_slot_rows,
        "average_end_to_end_delay_s",
        "Slot mean end-to-end delay (s)",
        "slot_mean_delay_distribution_by_seed",
        "Full-cycle slot-mean delay distribution by seed",
    )
    energy_plot = plot_distribution(
        args.output_dir,
        seeds,
        all_slot_rows,
        "average_energy_j",
        "Slot mean energy (J)",
        "slot_mean_energy_distribution_by_seed",
        "Full-cycle slot-mean energy distribution by seed",
    )
    final_summary = {
        "ablation": args.ablation,
        "seeds": seeds,
        "output_dir": args.output_dir,
        "slot_metrics_csv": args.output_dir / "slot_metrics_by_seed.csv",
        "request_metrics_csv": args.output_dir / "request_metrics_by_seed.csv",
        "cycle_metrics_csv": args.output_dir / "cycle_metrics_by_seed.csv",
        "delay_plot": delay_plot,
        "energy_plot": energy_plot,
        "seed_summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_summary


def load_seed_outputs(output_dir: Path, seeds: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    all_slot_rows = []
    all_request_rows = []
    summaries = []
    for seed in seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        slot_rows = read_csv(seed_output_dir / "slot_metrics.csv")
        request_rows = read_csv(seed_output_dir / "request_metrics.csv")
        if not slot_rows:
            raise SystemExit(f"Missing seed slot metrics: {seed_output_dir / 'slot_metrics.csv'}")
        all_slot_rows.extend(slot_rows)
        all_request_rows.extend(request_rows)
        summary_path = seed_output_dir / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return all_slot_rows, all_request_rows, summaries


def run_seed(
    args: argparse.Namespace,
    seed: int,
    index: int,
    seeds: list[int],
) -> tuple[list[dict], list[dict], dict]:
    run_dir = run_dir_for_seed(args, seeds, seed, index)
    seed_output_dir = args.output_dir / f"seed_{seed}"
    config = build_config(args, seed, run_dir, seed_output_dir)
    bandit_agent = ReplicaPlacementMigrationAgent(config)
    bandit_stats = run_dir / args.bandit_stats_name
    bandit_loaded_count = 0
    bandit_enabled = args.ablation != "no_bandit"
    if bandit_enabled and not args.no_load_bandit and bandit_stats.exists():
        bandit_loaded_count = bandit_agent.load_arm_stats(bandit_stats)

    env = SimulationEnvironment(
        config,
        migration_agent=bandit_agent,
        auto_generate_requests=False,
        auto_apply_migration=False,
    ).build()
    slot_count = env.context["slot_count"]
    slot_duration = env.context["slot_duration"]
    arrival_rng = random.Random(seed + 10_000)
    templates, template_csv, templates_loaded = load_or_generate_templates(
        args, run_dir, arrival_rng, env.context
    )
    original_template_count = len(templates)
    templates = filter_templates_by_chain_length(templates, args.chain_length_filter)
    if args.chain_length_filter is None and len(templates) != 14:
        print(f"Warning: seed={seed} has {len(templates)} request templates, expected 14.")

    checkpoint = run_dir / args.checkpoint_name
    agent, checkpoint_loaded, checkpoint_load_error = build_execution_agent(
        args, config, run_dir
    )

    slot_rows: list[dict] = []
    request_rows: list[dict] = []
    all_results: list[dict] = []
    next_request_id = 1
    cumulative_request_count = 0
    cumulative_feasible_count = 0
    bandit_window_requests = []
    bandit_feedback_results = []
    pending_migration_actions = []
    bandit_period_slots = max(1, int(args.bandit_period_slots))
    progress = TaskProgress(
        f"ablation={args.ablation} seed={seed}",
        slot_count,
        args.progress_every,
    )

    for epoch in range(1, slot_count + 1):
        absolute_slot = epoch - 1
        slot_mod = absolute_slot % slot_count
        arrivals, next_request_id, arrival_info = generate_arrivals_for_slot(
            args, arrival_rng, env.context, templates, absolute_slot, next_request_id
        )
        results = env.execute_requests(arrivals, agent)
        if bandit_enabled:
            bandit_agent.observe_failed_replicas(results)
        all_results.extend(results)
        if bandit_enabled:
            bandit_window_requests.extend(arrivals)
            bandit_feedback_results.extend(results)

        bandit_updated = False
        migration_actions = []
        is_cycle_last_slot = slot_mod == slot_count - 1
        if (
            bandit_enabled
            and (slot_mod + 1) % bandit_period_slots == 0
            and not is_cycle_last_slot
        ):
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
        cumulative_request_count += summary["request_count"]
        cumulative_feasible_count += summary["feasible_count"]
        cumulative_task_completion_rate = (
            cumulative_feasible_count / cumulative_request_count
            if cumulative_request_count
            else 0.0
        )
        total_reward = sum(float(result.get("reward", 0.0)) for result in results)
        route_mode_counts = route_mode_counts_for_results(results)
        row = {
            "ablation": args.ablation,
            "seed": seed,
            "epoch": epoch,
            "slot_mod": slot_mod,
            "slot_start_time_s": absolute_slot * slot_duration,
            "slot_duration_s": slot_duration,
            "arrival_count": arrival_info["arrival_count"],
            "processed_request_count": len(results),
            "request_count": summary["request_count"],
            "feasible_count": summary["feasible_count"],
            "failure_count": route_failure_count(results),
            "success_rate": summary["success_rate"],
            "slot_success_rate": summary["success_rate"],
            "cumulative_request_count": cumulative_request_count,
            "cumulative_feasible_count": cumulative_feasible_count,
            "task_completion_rate": cumulative_task_completion_rate,
            "cumulative_task_completion_rate": cumulative_task_completion_rate,
            "average_end_to_end_delay_s": summary["average_end_to_end_delay_s"],
            "average_energy_j": summary["average_energy_j"],
            "total_reward": total_reward,
            "average_reward_per_request": total_reward / len(results) if results else None,
            "p95_end_to_end_delay_s": summary["p95_end_to_end_delay_s"],
            "average_communication_delay_s": summary["average_communication_delay_s"],
            "average_slot_crossings": summary["average_slot_crossings"],
            "bandit_updated": bandit_updated,
            "bandit_action_count": len(migration_actions),
            "service_routing_strategy": config.service_routing_strategy,
            "execution_agent": agent.__class__.__name__,
            **flatten_route_counts(route_mode_counts),
            **{
                f"routing_{key}": value
                for key, value in env.context["routing_cache"]["stats"].items()
            },
            **{
                f"route_estimate_cache_{key}": value
                for key, value in env.context.get("route_estimate_cache_stats", {}).items()
            },
        }
        slot_rows.append(row)
        request_rows.extend(
            request_metric_row(args.ablation, seed, epoch, slot_mod, result)
            for result in results
        )
        env.context["routing_cache"]["route_results"].clear()
        env.context.get("route_estimate_cache", {}).clear()

        progress.update(
            epoch,
            f"arrivals={len(arrivals)} success={row['success_rate']:.3f}",
        )

    if bandit_enabled and pending_migration_actions and bandit_feedback_results:
        bandit_agent.observe_execution_feedback(
            pending_migration_actions, bandit_feedback_results
        )

    summary = {
        "ablation": args.ablation,
        "seed": seed,
        "run_dir": run_dir,
        "slot_count": slot_count,
        "slot_duration_s": slot_duration,
        "request_template_csv": template_csv,
        "request_templates_loaded": templates_loaded,
        "request_template_count": len(templates),
        "original_request_template_count": original_template_count,
        "chain_length_filter": args.chain_length_filter,
        "arrival_mode": args.arrival_mode,
        "arrival_lambda_total_per_slot": (
            total_arrival_lambda(args, config, len(templates))
            if args.arrival_mode == "total_per_slot"
            else None
        ),
        "checkpoint": checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_load_error": checkpoint_load_error,
        "execution_agent": agent.__class__.__name__,
        "service_routing_strategy": config.service_routing_strategy,
        "bandit_redeployment_enabled": bandit_enabled,
        "bandit_arm_stats": bandit_stats,
        "bandit_loaded_arm_count": bandit_loaded_count,
        "arrival_lambda_per_pattern_per_slot": (
            config.request_arrival_lambda_per_pattern_per_slot
        ),
        "overall_summary": summarize_results(all_results),
        "bandit_summary": bandit_agent.summary(),
        "routing_cache_summary": dict(env.context["routing_cache"]["stats"]),
        "route_estimate_cache_summary": dict(
            env.context.get("route_estimate_cache_stats", {})
        ),
    }
    write_csv(seed_output_dir / "slot_metrics.csv", slot_rows)
    write_csv(seed_output_dir / "request_metrics.csv", request_rows)
    (seed_output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return slot_rows, request_rows, summary


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        all_slot_rows, all_request_rows, summaries = load_seed_outputs(args.output_dir, seeds)
        final_summary = aggregate_outputs(
            args, seeds, all_slot_rows, all_request_rows, summaries
        )
        print(json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2))
        return

    all_slot_rows: list[dict] = []
    all_request_rows: list[dict] = []
    summaries: list[dict] = []
    for index, seed in enumerate(seeds):
        slot_rows, request_rows, summary = run_seed(args, seed, index, seeds)
        all_slot_rows.extend(slot_rows)
        all_request_rows.extend(request_rows)
        summaries.append(summary)

    if args.skip_aggregate:
        print(json.dumps(jsonable({"seeds": seeds, "seed_summaries": summaries}), ensure_ascii=False, indent=2))
        return

    final_summary = aggregate_outputs(
        args, seeds, all_slot_rows, all_request_rows, summaries
    )
    print(json.dumps(jsonable(final_summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
