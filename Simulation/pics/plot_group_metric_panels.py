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
    ABLATION_GROUP_ABLATIONS,
    ABLATION_LABELS,
    canonical_ablation_names,
    canonicalize_ablation_row,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"
DEFAULT_ABLATIONS = " ".join(ABLATION_GROUP_ABLATIONS)

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
    "Fair-NFV": "#cc4c4c",
    "SP-Routing": "#0891b2",
    "SC-NFV": "#6b7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot latency_metrics.png and energy_metrics.png for one experiment group."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablations", default=DEFAULT_ABLATIONS)
    parser.add_argument("--format", choices=("png",), default="png")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [canonicalize_ablation_row(row) for row in csv.DictReader(handle)]


def rows_for_file(input_dir: Path, ablation: str, filename: str) -> list[dict]:
    rows = read_rows(input_dir / ablation / filename)
    if rows:
        return rows
    rows = []
    for seed_dir in sorted((input_dir / ablation).glob("seed_*")):
        rows.extend(read_rows(seed_dir / filename.replace("_by_seed", "")))
    return rows


def load_request_rows(input_dir: Path, ablations: list[str]) -> list[dict]:
    rows = []
    for ablation in ablations:
        for row in rows_for_file(input_dir, ablation, "request_metrics_by_seed.csv"):
            row["ablation"] = row.get("ablation") or ablation
            canonicalize_ablation_row(row)
            rows.append(row)
    if not rows:
        rows = read_rows(input_dir / "all_ablation_request_metrics.csv")
    return [row for row in rows if row.get("ablation") in ablations]


def load_hop_rows(input_dir: Path, ablations: list[str]) -> list[dict]:
    rows = []
    for ablation in ablations:
        for row in rows_for_file(input_dir, ablation, "request_hop_metrics_by_seed.csv"):
            row["ablation"] = row.get("ablation") or ablation
            canonicalize_ablation_row(row)
            rows.append(row)
    if not rows:
        rows = read_rows(input_dir / "all_ablation_request_hop_metrics.csv")
    return [row for row in rows if row.get("ablation") in ablations]


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def values_by_ablation(rows: list[dict], ablations: list[str], column: str) -> dict[str, list[float]]:
    grouped = {ablation: [] for ablation in ablations}
    for row in rows:
        ablation = row.get("ablation")
        value = number(row, column)
        if ablation in grouped and value is not None:
            grouped[ablation].append(value)
    return grouped


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def p95(values: list[float]) -> float:
    return float(np.percentile(values, 95)) if values else math.nan


def labels(ablations: list[str]) -> list[str]:
    return [ABLATION_LABELS.get(ablation, ablation) for ablation in ablations]


def style_axes(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_bar(ax, grouped: dict[str, list[float]], ablations: list[str], reducer, title: str, ylabel: str) -> None:
    x = np.arange(len(ablations))
    heights = [reducer(grouped[ablation]) for ablation in ablations]
    colors = [COLORS.get(ablation, "#4b5563") for ablation in ablations]
    ax.bar(x, heights, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x, labels(ablations), rotation=20, ha="right")
    style_axes(ax, title, ylabel)


def plot_violin(ax, grouped: dict[str, list[float]], ablations: list[str], title: str, ylabel: str) -> None:
    data = [grouped[ablation] for ablation in ablations if grouped[ablation]]
    positions = [index + 1 for index, ablation in enumerate(ablations) if grouped[ablation]]
    if not data:
        ax.text(0.5, 0.5, "No request-level data", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax, title, ylabel)
        return
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.72)
    for body in parts["bodies"]:
        body.set_facecolor("#8ecae6")
        body.set_edgecolor("#24536b")
        body.set_alpha(0.72)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#263238")
            parts[key].set_linewidth(1.0)
    ax.set_xticks(np.arange(1, len(ablations) + 1), labels(ablations), rotation=20, ha="right")
    style_axes(ax, title, ylabel)


def hop_means(
    hop_rows: list[dict],
    ablations: list[str],
    compute_column: str,
    communication_column: str,
) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    buckets: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {"compute": [], "communication": []})
    for row in hop_rows:
        ablation = row.get("ablation")
        hop = number(row, "hop_index")
        compute_value = number(row, compute_column)
        communication_value = number(row, communication_column)
        if ablation not in ablations or hop is None:
            continue
        key = (ablation, int(hop))
        if compute_value is not None:
            buckets[key]["compute"].append(compute_value)
        if communication_value is not None:
            buckets[key]["communication"].append(communication_value)
    hop_indexes = sorted({hop for _, hop in buckets})
    series = {}
    for ablation in ablations:
        series[ablation] = {"compute": [], "communication": []}
        for hop in hop_indexes:
            values = buckets.get((ablation, hop), {"compute": [], "communication": []})
            series[ablation]["compute"].append(mean(values["compute"]))
            series[ablation]["communication"].append(mean(values["communication"]))
    return hop_indexes, series


def plot_hop_panel(
    ax,
    hop_rows: list[dict],
    ablations: list[str],
    compute_column: str,
    communication_column: str,
    title: str,
    ylabel: str,
) -> None:
    hop_indexes, series = hop_means(hop_rows, ablations, compute_column, communication_column)
    if not hop_indexes:
        ax.text(
            0.5,
            0.5,
            "No hop-level data\nRegenerate tests to write request_hop_metrics_by_seed.csv",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        style_axes(ax, title, ylabel)
        return
    for ablation in ablations:
        color = COLORS.get(ablation, "#4b5563")
        label = ABLATION_LABELS.get(ablation, ablation)
        ax.plot(hop_indexes, series[ablation]["compute"], color=color, linewidth=1.8, marker="o", label=f"{label} compute")
        ax.plot(hop_indexes, series[ablation]["communication"], color=color, linewidth=1.8, linestyle="--", marker="s", label=f"{label} communication")
    ax.set_xlabel("Microservice hop index")
    ax.set_xticks(hop_indexes)
    style_axes(ax, title, ylabel)
    ax.legend(fontsize=7, ncol=2, frameon=False)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def plot_metric_panel(
    output_path: Path,
    request_rows: list[dict],
    hop_rows: list[dict],
    ablations: list[str],
    metric_column: str,
    p95_title: str,
    mean_title: str,
    violin_title: str,
    hop_title: str,
    value_ylabel: str,
    compute_column: str,
    communication_column: str,
    hop_ylabel: str,
) -> None:
    grouped = values_by_ablation(request_rows, ablations, metric_column)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    plot_bar(axes[0, 0], grouped, ablations, mean, mean_title, value_ylabel)
    plot_bar(axes[0, 1], grouped, ablations, p95, p95_title, value_ylabel)
    plot_violin(axes[1, 0], grouped, ablations, violin_title, value_ylabel)
    plot_hop_panel(axes[1, 1], hop_rows, ablations, compute_column, communication_column, hop_title, hop_ylabel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    ablations = canonical_ablation_names(args.ablations)
    request_rows = load_request_rows(args.input_dir, ablations)
    hop_rows = load_hop_rows(args.input_dir, ablations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latency_path = args.output_dir / "latency_metrics.png"
    energy_path = args.output_dir / "energy_metrics.png"
    plot_metric_panel(
        latency_path,
        request_rows,
        hop_rows,
        ablations,
        "total_delay_s",
        "End-to-end latency P95",
        "Mean end-to-end latency",
        "End-to-end latency distribution",
        "Mean compute and communication latency per hop",
        "Latency (s)",
        "compute_total_delay_s",
        "communication_delay_s",
        "Latency (s)",
    )
    plot_metric_panel(
        energy_path,
        request_rows,
        hop_rows,
        ablations,
        "total_energy_j",
        "End-to-end energy P95",
        "Mean end-to-end energy",
        "End-to-end energy distribution",
        "Mean compute and communication energy per hop",
        "Energy (J)",
        "compute_energy_j",
        "communication_energy_j",
        "Energy (J)",
    )
    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(f"{latency_path.resolve()}\n{energy_path.resolve()}\n", encoding="utf-8")
    print(latency_path)
    print(energy_path)


if __name__ == "__main__":
    main()
