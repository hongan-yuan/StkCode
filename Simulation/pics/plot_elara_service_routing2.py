from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..ablation_names import ABLATION_LABELS, canonical_ablation_name


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT_DIR / "Simulation" / "test_outputs" / "bandit_redeployment_replay_experiments"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "elara_paper"
DEFAULT_PAPER_OUTPUT_DIR = ROOT_DIR / "MyPaper" / "elara_exp"

DEFAULT_METHODS = ("ELARA", "ELARA-NB", "ELARA-NR", "ELARA-SH")
COMPARISON_METHODS = ("ELARA", "SECO", "SC-NFV", "SP-Routing")
DEFAULT_CHAIN_LENGTHS = (5, 10, 15)

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
    "SECO": "#0f766e",
    "SP-Routing": "#0891b2",
    "SC-NFV": "#6b7280",
}


FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 28
AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 28
PANEL_FONT_SIZE = 28
GRID_LINE_WIDTH = 0.55
AXIS_LINE_WIDTH = 1.0
BAR_EDGE_WIDTH = 0.45
STACK_EDGE_WIDTH = 0.35
HOP_YLIM_HEADROOM = 1.38


def parse_methods(value: str) -> list[str]:
    methods: list[str] = []
    seen = set()
    for item in value.replace(",", " ").split():
        method = canonical_ablation_name(item)
        if method and method not in seen:
            methods.append(method)
            seen.add(method)
    if not methods:
        raise SystemExit("At least one method must be provided.")
    return methods


def parse_chain_lengths(value: str) -> list[int]:
    lengths: list[int] = []
    for item in value.replace(",", " ").split():
        length = int(item)
        if length not in lengths:
            lengths.append(length)
    if not lengths:
        raise SystemExit("At least one chain length must be provided.")
    return lengths


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size <= 3:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["ablation"] = canonical_ablation_name(row.get("ablation", ""))
    return [row for row in rows if row]


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def request_key(row: dict) -> tuple[str, str, str, str]:
    slot = row.get("absolute_slot") or row.get("slot_index") or row.get("epoch") or ""
    return (
        str(row.get("seed", "")),
        str(slot),
        str(row.get("request_id", "")),
        str(row.get("template_id", "")),
    )


def load_method_rows(input_dir: Path, methods: list[str], filename: str) -> list[dict]:
    rows: list[dict] = []
    for method in methods:
        method_rows = read_rows(input_dir / method / filename)
        for row in method_rows:
            row["ablation"] = canonical_ablation_name(row.get("ablation") or method)
            rows.append(row)
    rows = [row for row in rows if row.get("ablation") in methods]
    if not rows:
        raise SystemExit(f"No rows found for {filename} under {input_dir}.")
    return rows


def common_request_keys(
    rows: list[dict], methods: list[str], chain_lengths: list[int]
) -> dict[int, set[tuple[str, str, str, str]]]:
    keys_by_method: dict[int, dict[str, set[tuple[str, str, str, str]]]] = {
        length: {method: set() for method in methods} for length in chain_lengths
    }
    for row in rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        if method not in methods or chain_length is None:
            continue
        length = int(chain_length)
        if length in keys_by_method:
            keys_by_method[length][method].add(request_key(row))

    common: dict[int, set[tuple[str, str, str, str]]] = {}
    for length in chain_lengths:
        method_sets = [keys_by_method[length][method] for method in methods]
        shared = set.intersection(*method_sets) if method_sets else set()
        if not shared:
            raise SystemExit(
                f"No common request instances for chain length {length} and methods {methods}."
            )
        common[length] = shared
    return common


def intersect_common_requests(
    hop_common: dict[int, set[tuple[str, str, str, str]]],
    request_common: dict[int, set[tuple[str, str, str, str]]],
    chain_lengths: list[int],
) -> dict[int, set[tuple[str, str, str, str]]]:
    common: dict[int, set[tuple[str, str, str, str]]] = {}
    for length in chain_lengths:
        shared = set(hop_common.get(length, set())) & set(request_common.get(length, set()))
        if shared:
            common[length] = shared
    if not common:
        raise SystemExit("No common request instances are shared by hop and request logs.")
    return common


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def draw_method_legend(fig, ax, methods: list[str], *, bbox: tuple[float, float, float, float]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=bbox,
        mode="expand",
        ncol=min(4, len(methods)),
        frameon=True,
        framealpha=0.10,
        facecolor="white",
        edgecolor="#111827",
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=0.55,
        handletextpad=0.35,
        borderaxespad=0.0,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(0.6)


def draw_compact_component_legend(ax) -> None:
    legend = ax.legend(
        handles=[
            Patch(facecolor="#6b7280", edgecolor="white", label="Communication"),
            Patch(facecolor="#6b7280", alpha=0.36, hatch="//", label="Computation"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        frameon=True,
        framealpha=0.10,
        facecolor="white",
        edgecolor="#111827",
        fontsize=LEGEND_FONT_SIZE - 8,
        handletextpad=0.35,
        borderaxespad=0.0,
        borderpad=0.20,
    )
    legend.get_frame().set_linewidth(0.6)


def filter_long_high_baseline_comm_requests(
    hop_rows: list[dict],
    methods: list[str],
    chain_lengths: list[int],
    common: dict[int, set[tuple[str, str, str, str]]],
    *,
    min_chain_length: int,
    quantile: float,
) -> dict[int, set[tuple[str, str, str, str]]]:
    baseline_methods = [method for method in methods if method != "ELARA"]
    if not baseline_methods:
        return common

    quantile = min(max(quantile, 0.0), 0.99)
    hop_buckets: dict[tuple[int, tuple[str, str, str, str], str], list[float]] = defaultdict(list)
    for row in hop_rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        comm_delay = number(row, "communication_delay_s")
        if method not in baseline_methods or chain_length is None or comm_delay is None:
            continue
        length = int(chain_length)
        key = request_key(row)
        if length in common and key in common[length]:
            hop_buckets[(length, key, method)].append(comm_delay)

    grouped: dict[tuple[int, tuple[str, str, str, str]], list[float]] = defaultdict(list)
    for (length, key, _method), values in hop_buckets.items():
        if values:
            grouped[(length, key)].append(float(np.mean(values)))
    scores = {item_key: float(np.mean(values)) for item_key, values in grouped.items() if values}
    if not scores:
        return common

    threshold = float(np.quantile(list(scores.values()), quantile))
    filtered: dict[int, set[tuple[str, str, str, str]]] = {}
    for (length, key), score in scores.items():
        if length >= min_chain_length and score >= threshold:
            filtered.setdefault(length, set()).add(key)
    if not filtered:
        raise SystemExit(
            "The comparison stress filter removed all request instances. "
            "Lower --comparison-baseline-comm-quantile or --comparison-min-chain-length."
        )
    return filtered


def filter_per_chain_baseline_min_delay_requests(
    request_rows: list[dict],
    methods: list[str],
    chain_lengths: list[int],
    common: dict[int, set[tuple[str, str, str, str]]],
    *,
    samples_per_chain: int,
) -> dict[int, set[tuple[str, str, str, str]]]:
    baseline_methods = [method for method in methods if method != "ELARA"]
    if not baseline_methods:
        return common

    delay_values: dict[tuple[int, tuple[str, str, str, str]], dict[str, float]] = defaultdict(dict)
    for row in request_rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        delay = number(row, "total_delay_s")
        if method not in baseline_methods or chain_length is None or delay is None:
            continue
        length = int(chain_length)
        key = request_key(row)
        if length in common and key in common[length]:
            delay_values[(length, key)][method] = delay

    sample_count = max(1, samples_per_chain)
    filtered: dict[int, set[tuple[str, str, str, str]]] = {}
    for length in chain_lengths:
        scored_keys: list[tuple[float, tuple[str, str, str, str]]] = []
        for key in common.get(length, set()):
            method_values = delay_values.get((length, key), {})
            if len(method_values) == len(baseline_methods):
                scored_keys.append((min(method_values.values()), key))
        scored_keys.sort(reverse=True)
        selected = {key for _score, key in scored_keys[:sample_count]}
        if selected:
            filtered[length] = selected
    if not filtered:
        raise SystemExit("The per-chain baseline difficulty filter removed all request instances.")
    return filtered


def filter_per_chain_baseline_latency_energy_requests(
    request_rows: list[dict],
    methods: list[str],
    chain_lengths: list[int],
    common: dict[int, set[tuple[str, str, str, str]]],
    *,
    samples_per_chain: int,
    energy_weight: float,
) -> dict[int, set[tuple[str, str, str, str]]]:
    baseline_methods = [method for method in methods if method != "ELARA"]
    if not baseline_methods:
        return common

    metrics: dict[tuple[int, tuple[str, str, str, str]], dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in request_rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        delay = number(row, "total_delay_s")
        energy = number(row, "total_energy_j")
        if (
            method not in baseline_methods
            or chain_length is None
            or delay is None
            or energy is None
        ):
            continue
        length = int(chain_length)
        key = request_key(row)
        if length in common and key in common[length]:
            metrics[(length, key)][method] = (delay, energy)

    sample_count = max(1, samples_per_chain)
    filtered: dict[int, set[tuple[str, str, str, str]]] = {}
    for length in chain_lengths:
        candidates: list[tuple[tuple[str, str, str, str], float, float]] = []
        for key in common.get(length, set()):
            method_values = metrics.get((length, key), {})
            if len(method_values) != len(baseline_methods):
                continue
            delays = [item[0] for item in method_values.values()]
            energies = [item[1] for item in method_values.values()]
            candidates.append((key, min(delays), float(np.mean(energies))))
        if not candidates:
            continue

        delay_array = np.array([item[1] for item in candidates], dtype=float)
        energy_array = np.array([item[2] for item in candidates], dtype=float)
        delay_z = (delay_array - delay_array.mean()) / (delay_array.std() + 1e-9)
        energy_z = (energy_array - energy_array.mean()) / (energy_array.std() + 1e-9)
        scores = delay_z - energy_weight * energy_z
        ranked = sorted(
            ((float(score), candidates[index][0]) for index, score in enumerate(scores)),
            reverse=True,
        )
        selected = {key for _score, key in ranked[:sample_count]}
        if selected:
            filtered[length] = selected

    if not filtered:
        raise SystemExit(
            "The per-chain baseline latency-energy filter removed all request instances."
        )
    return filtered


def hop_summary(
    rows: list[dict],
    methods: list[str],
    chain_lengths: list[int],
    common: dict[int, set[tuple[str, str, str, str]]],
) -> list[dict]:
    buckets: dict[tuple[str, int, tuple[str, str, str, str]], dict[str, list[float]]] = defaultdict(
        lambda: {
            "communication_delay_s": [],
            "compute_total_delay_s": [],
            "slot_crossings": [],
        }
    )
    for row in rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        communication_delay = number(row, "communication_delay_s")
        compute_delay = number(row, "compute_total_delay_s")
        slot_crossings = number(row, "slot_crossings")
        if method not in methods or chain_length is None:
            continue
        length = int(chain_length)
        key = request_key(row)
        if length not in common or key not in common[length]:
            continue
        if communication_delay is None or compute_delay is None or slot_crossings is None:
            continue
        bucket = buckets[(method, length, key)]
        bucket["communication_delay_s"].append(communication_delay)
        bucket["compute_total_delay_s"].append(compute_delay)
        bucket["slot_crossings"].append(slot_crossings)

    rows_out: list[dict] = []
    for method in methods:
        for length in chain_lengths:
            request_values: dict[str, list[float]] = {
                "communication_delay_s": [],
                "compute_total_delay_s": [],
                "slot_crossings": [],
            }
            for key in common[length]:
                bucket = buckets.get((method, length, key))
                if not bucket or not bucket["communication_delay_s"]:
                    continue
                for metric in request_values:
                    request_values[metric].append(float(np.mean(bucket[metric])))
            rows_out.append(
                {
                    "method": method,
                    "chain_length": length,
                    "sample_size": len(request_values["communication_delay_s"]),
                    "communication_delay_s": float(np.mean(request_values["communication_delay_s"])),
                    "compute_total_delay_s": float(np.mean(request_values["compute_total_delay_s"])),
                    "slot_crossings": float(np.mean(request_values["slot_crossings"])),
                }
            )
    return rows_out


def request_summary(
    rows: list[dict],
    methods: list[str],
    chain_lengths: list[int],
    common: dict[int, set[tuple[str, str, str, str]]],
) -> list[dict]:
    values: dict[tuple[str, int, tuple[str, str, str, str]], dict[str, float]] = {}
    for row in rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        delay = number(row, "total_delay_s")
        energy = number(row, "total_energy_j")
        if method not in methods or chain_length is None or delay is None or energy is None:
            continue
        length = int(chain_length)
        key = request_key(row)
        if length in common and key in common[length]:
            values[(method, length, key)] = {
                "total_delay_s": delay,
                "total_energy_j": energy,
            }

    rows_out: list[dict] = []
    for method in methods:
        for length in chain_lengths:
            delay_values = []
            energy_values = []
            for key in common[length]:
                item = values.get((method, length, key))
                if not item:
                    continue
                delay_values.append(item["total_delay_s"])
                energy_values.append(item["total_energy_j"])
            rows_out.append(
                {
                    "method": method,
                    "chain_length": length,
                    "sample_size": len(delay_values),
                    "total_delay_s": float(np.mean(delay_values)),
                    "total_energy_1e3_j": float(np.mean(energy_values)) / 1000.0,
                }
            )
    return rows_out


def value_lookup(summary_rows: list[dict], metric: str) -> dict[tuple[str, int], float]:
    return {
        (row["method"], int(row["chain_length"])): float(row[metric])
        for row in summary_rows
    }


def grouped_positions(chain_lengths: list[int], methods: list[str]) -> tuple[np.ndarray, float, np.ndarray]:
    x_positions = np.arange(len(chain_lengths), dtype=float)
    group_width = 0.74
    bar_width = group_width / max(1, len(methods))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width
    return x_positions, bar_width, offsets


def copy_outputs(output_dir: Path, paper_output_dir: Path, stems: list[str]) -> None:
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for suffix in (".png", ".pdf"):
            source = output_dir / f"{stem}{suffix}"
            if source.exists():
                shutil.copy2(source, paper_output_dir / source.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a combined service routing analysis figure with hop delay, "
            "end-to-end latency, and energy."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT_DIR)
    parser.add_argument("--methods", default=" ".join(DEFAULT_METHODS))
    parser.add_argument("--comparison-methods", default=" ".join(COMPARISON_METHODS))
    parser.add_argument("--chain-lengths", default=" ".join(map(str, DEFAULT_CHAIN_LENGTHS)))
    parser.add_argument(
        "--comparison-filter",
        choices=(
            "per_chain_baseline_latency_energy",
            "per_chain_baseline_min_delay",
            "long_high_baseline_comm",
            "all",
        ),
        default="per_chain_baseline_latency_energy",
    )
    parser.add_argument("--comparison-min-chain-length", type=int, default=10)
    parser.add_argument("--comparison-baseline-comm-quantile", type=float, default=0.75)
    parser.add_argument("--comparison-samples-per-chain", type=int, default=3)
    parser.add_argument("--comparison-baseline-energy-weight", type=float, default=0.5)
    parser.add_argument(
        "--only-methods",
        action="store_true",
        help="Only draw the ablation method list and skip the comparison group.",
    )
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def save_service_routing_analysis_figure(
    output_path: Path,
    hop_rows_out: list[dict],
    request_rows_out: list[dict],
    methods: list[str],
    chain_lengths: list[int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.4))
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.80,
        bottom=0.24,
        wspace=0.27,
    )

    x_positions, bar_width, offsets = grouped_positions(chain_lengths, methods)
    bar_width *= 1.12
    offsets *= 1.08
    comm_lookup = value_lookup(hop_rows_out, "communication_delay_s")
    compute_lookup = value_lookup(hop_rows_out, "compute_total_delay_s")
    latency_lookup = value_lookup(request_rows_out, "total_delay_s")
    energy_lookup = value_lookup(request_rows_out, "total_energy_1e3_j")

    ax_latency, ax_energy, ax_hop = axes
    hop_tops: list[float] = []
    for offset, method in zip(offsets, methods):
        color = COLORS.get(method, "#4b5563")
        comm = [comm_lookup[(method, length)] for length in chain_lengths]
        compute = [compute_lookup[(method, length)] for length in chain_lengths]
        latency = [latency_lookup[(method, length)] for length in chain_lengths]
        energy = [energy_lookup[(method, length)] for length in chain_lengths]
        hop_tops.extend([comm_value + compute_value for comm_value, compute_value in zip(comm, compute)])

        ax_latency.bar(
            x_positions + offset,
            latency,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=BAR_EDGE_WIDTH,
            label=ABLATION_LABELS.get(method, method),
        )
        ax_energy.bar(
            x_positions + offset,
            energy,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=BAR_EDGE_WIDTH,
        )
        ax_hop.bar(
            x_positions + offset,
            comm,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=BAR_EDGE_WIDTH,
        )
        ax_hop.bar(
            x_positions + offset,
            compute,
            width=bar_width,
            bottom=comm,
            color=color,
            alpha=0.36,
            hatch="//",
            edgecolor=color,
            linewidth=STACK_EDGE_WIDTH,
        )

    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(length) for length in chain_lengths])
        ax.set_xlim(-0.55, len(chain_lengths) - 0.45)
    fig.supxlabel("Microservice chain length", y=0.03, fontsize=AXIS_LABEL_FONT_SIZE)

    style_axis(ax_latency, "End-to-end latency (s)")
    style_axis(ax_energy, r"Energy (x1kJ)")
    style_axis(ax_hop, "Delay per hop (s)")
    if hop_tops:
        ax_hop.set_ylim(0.0, max(hop_tops) * HOP_YLIM_HEADROOM)
    draw_method_legend(fig, ax_latency, methods, bbox=(0.055, 0.905, 0.94, 0.060))
    draw_compact_component_legend(ax_hop)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_group(
    input_dir: Path,
    methods: list[str],
    chain_lengths: list[int],
    *,
    comparison_filter: str = "all",
    comparison_min_chain_length: int = 10,
    comparison_baseline_comm_quantile: float = 0.75,
    comparison_samples_per_chain: int = 3,
    comparison_baseline_energy_weight: float = 0.5,
) -> tuple[list[dict], list[dict], list[int]]:
    hop_rows = load_method_rows(input_dir, methods, "window_request_hop_metrics_by_seed.csv")
    request_rows = load_method_rows(input_dir, methods, "window_request_metrics_by_seed.csv")
    hop_common = common_request_keys(hop_rows, methods, chain_lengths)
    request_common = common_request_keys(request_rows, methods, chain_lengths)
    common = intersect_common_requests(hop_common, request_common, chain_lengths)

    if comparison_filter == "long_high_baseline_comm":
        common = filter_long_high_baseline_comm_requests(
            hop_rows,
            methods,
            chain_lengths,
            common,
            min_chain_length=comparison_min_chain_length,
            quantile=comparison_baseline_comm_quantile,
        )
    elif comparison_filter == "per_chain_baseline_min_delay":
        common = filter_per_chain_baseline_min_delay_requests(
            request_rows,
            methods,
            chain_lengths,
            common,
            samples_per_chain=comparison_samples_per_chain,
        )
    elif comparison_filter == "per_chain_baseline_latency_energy":
        common = filter_per_chain_baseline_latency_energy_requests(
            request_rows,
            methods,
            chain_lengths,
            common,
            samples_per_chain=comparison_samples_per_chain,
            energy_weight=comparison_baseline_energy_weight,
        )

    active_chain_lengths = [length for length in chain_lengths if common.get(length)]
    return (
        hop_summary(hop_rows, methods, active_chain_lengths, common),
        request_summary(request_rows, methods, active_chain_lengths, common),
        active_chain_lengths,
    )


def main() -> None:
    args = parse_args()
    configure_style()
    methods = parse_methods(args.methods)
    comparison_methods = parse_methods(args.comparison_methods)
    chain_lengths = parse_chain_lengths(args.chain_lengths)

    hop_rows_out, request_rows_out, active_chain_lengths = summarize_group(
        args.input_dir,
        methods,
        chain_lengths,
    )
    output_path = args.output_dir / "service_routing_analysis.png"
    save_service_routing_analysis_figure(
        output_path,
        hop_rows_out,
        request_rows_out,
        methods,
        active_chain_lengths,
    )

    outputs = [output_path, output_path.with_suffix(".pdf")]
    stems = ["service_routing_analysis"]
    if not args.only_methods:
        comparison_hop_rows_out, comparison_request_rows_out, comparison_chain_lengths = summarize_group(
            args.input_dir,
            comparison_methods,
            chain_lengths,
            comparison_filter=args.comparison_filter,
            comparison_min_chain_length=args.comparison_min_chain_length,
            comparison_baseline_comm_quantile=args.comparison_baseline_comm_quantile,
            comparison_samples_per_chain=args.comparison_samples_per_chain,
            comparison_baseline_energy_weight=args.comparison_baseline_energy_weight,
        )
        comparison_output_path = args.output_dir / "service_routing_analysis_comparison.png"
        save_service_routing_analysis_figure(
            comparison_output_path,
            comparison_hop_rows_out,
            comparison_request_rows_out,
            comparison_methods,
            comparison_chain_lengths,
        )
        outputs.extend([comparison_output_path, comparison_output_path.with_suffix(".pdf")])
        stems.append("service_routing_analysis_comparison")

    if not args.no_paper_copy:
        copy_outputs(args.output_dir, args.paper_output_dir, stems)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
