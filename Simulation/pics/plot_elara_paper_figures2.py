from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..ablation_names import ABLATION_LABELS, canonical_ablation_name


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
DEFAULT_CHAIN_INPUT_DIR = (
    ROOT_DIR / "Simulation" / "test_outputs" / "chain_length_ablation_experiments"
)
DEFAULT_CHAIN_FALLBACK_DIR = (
    ROOT_DIR / "Simulation" / "arxiv_test_outputs" / "chain_length_ablation_experiments"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "elara_paper"

COMPARISON_ABLATIONS = ("ELARA", "SECO", "SC-NFV", "SP-Routing")
ABLATION_ABLATIONS = ("ELARA", "ELARA-NB", "ELARA-NR", "ELARA-SH")

FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 11
TICK_LABEL_FONT_SIZE = 9
LEGEND_FONT_SIZE = 9
TITLE_FONT_SIZE = 12
AXIS_LINE_WIDTH = 1.0
LINE_WIDTH = 1.8
MARKER_SIZE = 2.8
GRID_LINE_WIDTH = 0.55

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
    "SECO": "#0f766e",
    "SP-Routing": "#0891b2",
    "SC-NFV": "#6b7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw cross-slot comparison and ablation insight figures with a "
            "2/3-length x-axis."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--chain-input-dir", type=Path, default=DEFAULT_CHAIN_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--comparison-ablations",
        default=" ".join(COMPARISON_ABLATIONS),
        help="Space/comma separated methods for comparison cross-slot plots.",
    )
    parser.add_argument(
        "--ablation-ablations",
        default=" ".join(ABLATION_ABLATIONS),
        help="Space/comma separated methods for ablation cross-slot plots.",
    )
    parser.add_argument("--slot-window", type=int, default=10)
    parser.add_argument("--max-slot", type=int, default=600)
    parser.add_argument(
        "--x-axis-fraction",
        type=float,
        default=2.0 / 3.0,
        help="Fraction of --max-slot kept on the x-axis. Default keeps 400/600 slots.",
    )
    parser.add_argument("--format", choices=("png",), default="png")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_ablation_list(value: str) -> list[str]:
    names: list[str] = []
    seen = set()
    for item in value.replace(",", " ").split():
        name = canonical_ablation_name(item)
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "ablation" in row:
            row["ablation"] = canonical_ablation_name(row["ablation"])
    return rows


def load_request_rows(input_dir: Path, ablations: list[str]) -> list[dict]:
    rows: list[dict] = []
    for ablation in ablations:
        variant_rows = read_rows(input_dir / ablation / "request_metrics_by_seed.csv")
        if not variant_rows:
            for seed_dir in sorted((input_dir / ablation).glob("seed_*")):
                variant_rows.extend(read_rows(seed_dir / "request_metrics.csv"))
        for row in variant_rows:
            row["ablation"] = canonical_ablation_name(row.get("ablation") or ablation)
            rows.append(row)

    if not rows:
        rows = read_rows(input_dir / "all_ablation_request_metrics.csv")
    return [row for row in rows if row.get("ablation") in ablations]


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def effective_max_slot(max_slot: int, slot_window: int, x_axis_fraction: float) -> int:
    if slot_window <= 0:
        raise SystemExit("--slot-window must be positive.")
    if max_slot <= 0:
        raise SystemExit("--max-slot must be positive.")
    if not 0.0 < x_axis_fraction <= 1.0:
        raise SystemExit("--x-axis-fraction must be in the interval (0, 1].")
    raw_max = int(math.floor(max_slot * x_axis_fraction))
    reduced = (raw_max // slot_window) * slot_window
    return max(slot_window, reduced)


def window_means(
    rows: list[dict],
    ablations: list[str],
    metric: str,
    slot_window: int,
    max_slot: int,
) -> tuple[list[int], dict[str, list[float]]]:
    window_count = max_slot // slot_window
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        ablation = row.get("ablation")
        epoch = number(row, "epoch")
        value = number(row, metric)
        if ablation not in ablations or epoch is None or value is None:
            continue
        slot = int(epoch)
        if slot < 1 or slot > max_slot:
            continue
        window_index = (slot - 1) // slot_window + 1
        buckets[(ablation, window_index)].append(value)

    windows = list(range(1, window_count + 1))
    series: dict[str, list[float]] = {}
    for ablation in ablations:
        values: list[float] = []
        for window_index in windows:
            bucket = buckets.get((ablation, window_index), [])
            values.append(float(np.mean(bucket)) if bucket else math.nan)
        series[ablation] = values
    return windows, series


def style_axis(ax, ylabel: str, max_slot: int, slot_window: int) -> None:
    ax.set_xlabel(f"{slot_window}-slot window index, slots 1-{max_slot}")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def style_scalability_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Microservice chain length")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def plot_metric_on_axis(
    ax,
    rows: list[dict],
    ablations: list[str],
    metric: str,
    ylabel: str,
    panel_label: str,
    slot_window: int,
    max_slot: int,
    show_legend: bool,
) -> None:
    windows, series = window_means(rows, ablations, metric, slot_window, max_slot)
    for ablation in ablations:
        ax.plot(
            windows,
            series[ablation],
            color=COLORS.get(ablation, "#4b5563"),
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=ABLATION_LABELS.get(ablation, ablation),
        )
    ax.set_xlim(1, len(windows))
    tick_step = 5 if len(windows) > 12 else 1
    ax.set_xticks(list(range(1, len(windows) + 1, tick_step)))
    ax.text(
        0.015,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    style_axis(ax, ylabel, max_slot, slot_window)
    if show_legend:
        ax.legend(frameon=False, ncol=2, loc="best")


def save_single_plot(
    output_path: Path,
    rows: list[dict],
    ablations: list[str],
    metric: str,
    ylabel: str,
    panel_label: str,
    slot_window: int,
    max_slot: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    plot_metric_on_axis(
        ax,
        rows,
        ablations,
        metric,
        ylabel,
        panel_label,
        slot_window,
        max_slot,
        show_legend=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_plot(
    output_path: Path,
    rows: list[dict],
    ablations: list[str],
    group_name: str,
    slot_window: int,
    max_slot: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)
    plot_metric_on_axis(
        axes[0],
        rows,
        ablations,
        "total_delay_s",
        "Mean end-to-end latency (s)",
        f"({group_name}-a) Latency",
        slot_window,
        max_slot,
        show_legend=False,
    )
    plot_metric_on_axis(
        axes[1],
        rows,
        ablations,
        "total_energy_j",
        "Mean end-to-end energy (J)",
        f"({group_name}-b) Energy",
        slot_window,
        max_slot,
        show_legend=False,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(ablations),
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_chain_summary_rows(chain_input_dir: Path) -> list[dict]:
    candidates = [
        chain_input_dir / "chain_length_ablation_metric_summary.csv",
        DEFAULT_CHAIN_FALLBACK_DIR / "chain_length_ablation_metric_summary.csv",
    ]
    for path in candidates:
        rows = read_rows(path)
        if rows:
            return rows
    return []


def chain_metric_series(
    rows: list[dict],
    ablations: list[str],
    metric: str,
) -> tuple[list[int], dict[str, list[float]]]:
    chain_lengths = sorted(
        {
            int(value)
            for row in rows
            if (value := number(row, "chain_length_filter")) is not None
        }
    )
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        ablation = row.get("ablation")
        chain_length = number(row, "chain_length_filter")
        value = number(row, metric)
        if ablation not in ablations or chain_length is None or value is None:
            continue
        buckets[(ablation, int(chain_length))].append(value)

    series: dict[str, list[float]] = {}
    for ablation in ablations:
        values = []
        for chain_length in chain_lengths:
            bucket = buckets.get((ablation, chain_length), [])
            values.append(float(np.mean(bucket)) if bucket else math.nan)
        series[ablation] = values
    return chain_lengths, series


def plot_chain_metric_on_axis(
    ax,
    rows: list[dict],
    ablations: list[str],
    metric: str,
    ylabel: str,
    panel_label: str,
    show_legend: bool,
) -> None:
    chain_lengths, series = chain_metric_series(rows, ablations, metric)
    for ablation in ablations:
        ax.plot(
            chain_lengths,
            series[ablation],
            color=COLORS.get(ablation, "#4b5563"),
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE + 1.0,
            label=ABLATION_LABELS.get(ablation, ablation),
        )
    ax.set_xticks(chain_lengths)
    ax.text(
        0.015,
        0.97,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    style_scalability_axis(ax, ylabel)
    if show_legend:
        ax.legend(frameon=False, ncol=2, loc="best")


def save_chain_scalability_plot(
    output_path: Path,
    rows: list[dict],
    ablations: list[str],
    group_name: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), constrained_layout=True)
    plot_chain_metric_on_axis(
        axes[0],
        rows,
        ablations,
        "mean_average_end_to_end_delay_s",
        "Mean latency (s)",
        f"({group_name}-a) Average latency",
        show_legend=False,
    )
    plot_chain_metric_on_axis(
        axes[1],
        rows,
        ablations,
        "mean_p95_end_to_end_delay_s",
        "P95 latency (s)",
        f"({group_name}-b) Tail latency",
        show_legend=False,
    )
    plot_chain_metric_on_axis(
        axes[2],
        rows,
        ablations,
        "mean_average_energy_j",
        "Mean energy (J)",
        f"({group_name}-c) Energy",
        show_legend=False,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(ablations),
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_slot = effective_max_slot(args.max_slot, args.slot_window, args.x_axis_fraction)
    comparison_ablations = parse_ablation_list(args.comparison_ablations)
    ablation_ablations = parse_ablation_list(args.ablation_ablations)
    comparison_rows = load_request_rows(args.input_dir, comparison_ablations)
    ablation_rows = load_request_rows(args.input_dir, ablation_ablations)

    outputs = [
        (
            args.output_dir / f"cross_slot_comparison_latency.{args.format}",
            comparison_rows,
            comparison_ablations,
            "total_delay_s",
            "Mean end-to-end latency (s)",
            "(a) Comparison latency",
        ),
        (
            args.output_dir / f"cross_slot_comparison_energy.{args.format}",
            comparison_rows,
            comparison_ablations,
            "total_energy_j",
            "Mean end-to-end energy (J)",
            "(b) Comparison energy",
        ),
        (
            args.output_dir / f"cross_slot_ablation_latency.{args.format}",
            ablation_rows,
            ablation_ablations,
            "total_delay_s",
            "Mean end-to-end latency (s)",
            "(a) Ablation latency",
        ),
        (
            args.output_dir / f"cross_slot_ablation_energy.{args.format}",
            ablation_rows,
            ablation_ablations,
            "total_energy_j",
            "Mean end-to-end energy (J)",
            "(b) Ablation energy",
        ),
    ]
    for output_path, rows, ablations, metric, ylabel, panel_label in outputs:
        save_single_plot(
            output_path,
            rows,
            ablations,
            metric,
            ylabel,
            panel_label,
            args.slot_window,
            max_slot,
        )

    comparison_insight = args.output_dir / f"elara_cs_comparison_insight.{args.format}"
    ablation_insight = args.output_dir / f"elara_cs_ablation_insight.{args.format}"
    save_combined_plot(
        comparison_insight,
        comparison_rows,
        comparison_ablations,
        "Comparison",
        args.slot_window,
        max_slot,
    )
    save_combined_plot(
        ablation_insight,
        ablation_rows,
        ablation_ablations,
        "Ablation",
        args.slot_window,
        max_slot,
    )

    manifest_paths = [item[0] for item in outputs] + [comparison_insight, ablation_insight]
    chain_rows = load_chain_summary_rows(args.chain_input_dir)
    if chain_rows:
        chain_comparison = args.output_dir / f"chain_length_comparison_scalability.{args.format}"
        chain_ablation = args.output_dir / f"chain_length_ablation_scalability.{args.format}"
        save_chain_scalability_plot(
            chain_comparison,
            chain_rows,
            comparison_ablations,
            "Comparison",
        )
        save_chain_scalability_plot(
            chain_ablation,
            chain_rows,
            ablation_ablations,
            "Ablation",
        )
        manifest_paths.extend([chain_comparison, chain_ablation])
    else:
        print(
            "No chain-length summary found; skipped scalability plots. "
            f"Checked {args.chain_input_dir} and {DEFAULT_CHAIN_FALLBACK_DIR}."
        )

    manifest = args.output_dir / "plot_elara_paper_figures2_manifest.txt"
    manifest.write_text(
        "\n".join(str(path.resolve()) for path in manifest_paths) + "\n",
        encoding="utf-8",
    )
    for path in manifest_paths:
        print(path)


if __name__ == "__main__":
    main()
