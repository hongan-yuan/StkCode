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

from ..ablation_names import (
    ABLATION_LABELS,
    COMPARISON_GROUP_ABLATIONS,
    canonical_ablation_names,
    canonicalize_ablation_row,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "bandit_redeployment_replay_experiments"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"
DEFAULT_ABLATIONS = " ".join(COMPARISON_GROUP_ABLATIONS)

FONT_FAMILY = "Times New Roman"
LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 11
TICK_LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE = 12
AXIS_LINE_WIDTH = 1.0
BAR_EDGE_LINE_WIDTH = 0.8
LINE_WIDTH = 2.0

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
    "Fair-NFV": "#cc4c4c",
    "SECO": "#0f766e",
    "SP-Routing": "#0891b2",
    "SC-NFV": "#6b7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot before/after metrics for bandit redeployment replay experiments."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablations", default=DEFAULT_ABLATIONS)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": TICK_LABEL_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [canonicalize_ablation_row(row) for row in csv.DictReader(handle)]


def load_rows(input_dir: Path, ablations: list[str]) -> list[dict]:
    rows = read_rows(input_dir / "all_redeployment_summary_metrics.csv")
    if not rows:
        rows = []
        for ablation in ablations:
            variant_rows = read_rows(input_dir / ablation / "redeployment_summary_by_seed.csv")
            for row in variant_rows:
                row["ablation"] = row.get("ablation") or ablation
                rows.append(canonicalize_ablation_row(row))
    return [row for row in rows if row.get("ablation") in ablations]


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None, "None", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def grouped_values(rows: list[dict], ablations: list[str], column: str) -> dict[str, list[float]]:
    grouped = {ablation: [] for ablation in ablations}
    for row in rows:
        ablation = row.get("ablation")
        value = number(row, column)
        if ablation in grouped and value is not None:
            grouped[ablation].append(value)
    return grouped


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def labels(ablations: list[str]) -> list[str]:
    return [ABLATION_LABELS.get(ablation, ablation) for ablation in ablations]


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def plot_before_after_bars(
    ax,
    rows: list[dict],
    ablations: list[str],
    before_column: str,
    after_column: str,
    title: str,
    ylabel: str,
) -> None:
    before = grouped_values(rows, ablations, before_column)
    after = grouped_values(rows, ablations, after_column)
    x = np.arange(len(ablations))
    width = 0.36
    before_means = [mean(before[ablation]) for ablation in ablations]
    after_means = [mean(after[ablation]) for ablation in ablations]
    before_err = [stderr(before[ablation]) for ablation in ablations]
    after_err = [stderr(after[ablation]) for ablation in ablations]
    colors = [COLORS.get(ablation, "#4b5563") for ablation in ablations]
    ax.bar(
        x - width / 2,
        before_means,
        width,
        yerr=before_err,
        color=colors,
        alpha=0.45,
        edgecolor="#222222",
        linewidth=BAR_EDGE_LINE_WIDTH,
        capsize=3,
        label="Before",
    )
    ax.bar(
        x + width / 2,
        after_means,
        width,
        yerr=after_err,
        color=colors,
        alpha=0.95,
        edgecolor="#222222",
        linewidth=BAR_EDGE_LINE_WIDTH,
        capsize=3,
        label="After",
    )
    ax.set_xticks(x, labels(ablations), rotation=22, ha="right")
    style_axes(ax, title, ylabel)
    ax.legend(frameon=False)


def plot_reduction_bars(
    ax,
    rows: list[dict],
    ablations: list[str],
    ratio_column: str,
    title: str,
    ylabel: str,
) -> None:
    grouped = grouped_values(rows, ablations, ratio_column)
    x = np.arange(len(ablations))
    values = [100.0 * mean(grouped[ablation]) for ablation in ablations]
    errors = [100.0 * stderr(grouped[ablation]) for ablation in ablations]
    colors = [COLORS.get(ablation, "#4b5563") for ablation in ablations]
    ax.axhline(0.0, color="#222222", linewidth=AXIS_LINE_WIDTH)
    ax.bar(
        x,
        values,
        yerr=errors,
        color=colors,
        edgecolor="#222222",
        linewidth=BAR_EDGE_LINE_WIDTH,
        capsize=3,
    )
    ax.set_xticks(x, labels(ablations), rotation=22, ha="right")
    style_axes(ax, title, ylabel)


def plot_action_counts(
    ax,
    rows: list[dict],
    ablations: list[str],
) -> None:
    grouped = grouped_values(rows, ablations, "redeployment_action_count")
    x = np.arange(len(ablations))
    values = [mean(grouped[ablation]) for ablation in ablations]
    colors = [COLORS.get(ablation, "#4b5563") for ablation in ablations]
    ax.plot(
        x,
        values,
        color="#111827",
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=4,
    )
    ax.bar(
        x,
        values,
        color=colors,
        alpha=0.35,
        edgecolor="#222222",
        linewidth=BAR_EDGE_LINE_WIDTH,
    )
    ax.set_xticks(x, labels(ablations), rotation=22, ha="right")
    style_axes(ax, "Bandit redeployment actions", "Mean action count")


def write_metric_summary(output_path: Path, rows: list[dict], ablations: list[str]) -> None:
    fields = [
        "before_average_end_to_end_delay_s",
        "after_average_end_to_end_delay_s",
        "delay_reduction_ratio",
        "before_average_energy_j",
        "after_average_energy_j",
        "energy_reduction_ratio",
        "redeployment_action_count",
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("ablation", "")].append(row)
    output_rows = []
    for ablation in ablations:
        row = {"ablation": ablation, "label": ABLATION_LABELS.get(ablation, ablation)}
        for field in fields:
            values = [value for item in grouped[ablation] if (value := number(item, field)) is not None]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_stderr"] = stderr(values)
        output_rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(output_rows[0]) if output_rows else ["ablation"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    args = parse_args()
    configure_style()
    ablations = canonical_ablation_names(args.ablations)
    rows = load_rows(args.input_dir, ablations)
    if not rows:
        raise SystemExit(f"No redeployment summary rows found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), constrained_layout=True)
    plot_before_after_bars(
        axes[0, 0],
        rows,
        ablations,
        "before_average_end_to_end_delay_s",
        "after_average_end_to_end_delay_s",
        "Before/after end-to-end delay",
        "Mean delay (s)",
    )
    plot_before_after_bars(
        axes[0, 1],
        rows,
        ablations,
        "before_average_energy_j",
        "after_average_energy_j",
        "Before/after energy",
        "Mean energy (J)",
    )
    plot_reduction_bars(
        axes[1, 0],
        rows,
        ablations,
        "delay_reduction_ratio",
        "Delay reduction after replay",
        "Reduction (%)",
    )
    plot_reduction_bars(
        axes[1, 1],
        rows,
        ablations,
        "energy_reduction_ratio",
        "Energy reduction after replay",
        "Reduction (%)",
    )
    output_path = args.output_dir / f"redeployment_replay_metrics.{args.format}"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    action_path = args.output_dir / f"redeployment_action_counts.{args.format}"
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    plot_action_counts(ax, rows, ablations)
    fig.savefig(action_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = args.output_dir / "redeployment_plot_summary.csv"
    write_metric_summary(summary_path, rows, ablations)
    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(
        f"{output_path.resolve()}\n{action_path.resolve()}\n{summary_path.resolve()}\n",
        encoding="utf-8",
    )
    print(output_path)
    print(action_path)
    print(summary_path)


if __name__ == "__main__":
    main()
