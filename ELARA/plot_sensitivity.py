from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Plot ELARA weight and routing-path sensitivity separately."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("ELARA/outputs/sensitivity"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)
    args.formats = tuple(
        dict.fromkeys(item.strip().lower() for item in args.formats.split(",") if item.strip())
    )
    if not args.formats or set(args.formats) - {"png", "pdf", "svg"}:
        parser.error("--formats must contain png, pdf, and/or svg")
    return args


def resolve_summary(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    direct = path / "sensitivity_summary.csv"
    if direct.is_file():
        return direct
    candidates = sorted(path.glob("*/sensitivity_summary.csv"))
    if not candidates:
        raise ValueError(f"no sensitivity_summary.csv found below {path}")
    return candidates[-1]


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "seed", "delay_weight", "energy_weight", "route_max_paths",
        "request_count", "success_rate", "mean_return", "mean_latency_s",
        "mean_energy_j", "mean_route_slot_crossings", "mean_route_phase_count",
        "mean_route_augmentation_count",
    }
    for row in rows:
        for key in numeric:
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = math.nan
    return rows


def mean_ci(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    mean = float(np.mean(values))
    error = (
        float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1 else 0.0
    )
    return mean, error


def grouped(rows, category):
    result = defaultdict(list)
    for row in rows:
        if row["category"] == category:
            result[row["condition"]].append(row)
    return result


def save(fig, directory, stem, formats, dpi):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_metric_panels(groups, labels, metrics, title, directory, formats, dpi):
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.2, 3.35), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, (metric, panel_title, ylabel, multiplier) in zip(axes, metrics):
        means, errors = [], []
        for label in labels:
            mean, error = mean_ci([row[metric] * multiplier for row in groups[label]])
            means.append(mean)
            errors.append(error)
        bars = ax.bar(
            x, means, yerr=errors, capsize=3, width=0.65,
            color="#4c78a8", edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(x, labels)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.45)
        for bar, value in zip(bars, means):
            if not math.isfinite(value):
                continue
            ax.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=7.5,
            )
    fig.suptitle(title, fontsize=11.5)
    return save(fig, directory, "performance_sensitivity", formats, dpi)


def plot_weight_tradeoff(groups, labels, directory, formats, dpi):
    fig, ax = plt.subplots(figsize=(5.6, 4.15), constrained_layout=True)
    for label in labels:
        latency, latency_ci = mean_ci([row["mean_latency_s"] for row in groups[label]])
        energy, energy_ci = mean_ci([row["mean_energy_j"] for row in groups[label]])
        ax.errorbar(
            latency, energy, xerr=latency_ci, yerr=energy_ci,
            marker="o", markersize=6, capsize=3, label=label,
        )
        ax.annotate(label, (latency, energy), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean latency (s)")
    ax.set_ylabel("Mean energy (J)")
    ax.set_title("Latency and Energy Tradeoff")
    ax.grid(True, alpha=0.25, linewidth=0.45)
    return save(fig, directory, "latency_energy_tradeoff", formats, dpi)


def plot_route_overhead(groups, labels, directory, formats, dpi):
    paths = [int(label) for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.35), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0], "mean_route_augmentation_count", "Used augmentations", "Mean per request"),
        (axes[1], "mean_route_slot_crossings", "Cross-slot routing", "Mean crossings"),
    ):
        means, errors = zip(
            *(mean_ci([row[metric] for row in groups[label]]) for label in labels)
        )
        ax.errorbar(paths, means, yerr=errors, marker="o", capsize=3, linewidth=1.5)
        ax.set_xticks(paths)
        ax.set_xlabel("Maximum augmenting paths")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.45)
    fig.suptitle("Data Routing Complexity and Robustness", fontsize=11.5)
    return save(fig, directory, "routing_overhead", formats, dpi)


def write_statistics(rows, output_dir):
    path = output_dir / "sensitivity_statistics.csv"
    fields = (
        "category", "condition", "seed_count",
        "mean_latency_s", "mean_latency_s_ci95",
        "mean_energy_j", "mean_energy_j_ci95",
        "success_rate", "success_rate_ci95",
        "mean_route_augmentation_count", "mean_route_augmentation_count_ci95",
    )
    records = []
    for category in ("latency_energy_weights", "routing_max_paths"):
        for condition, samples in grouped(rows, category).items():
            record = {"category": category, "condition": condition, "seed_count": len(samples)}
            for metric in (
                "mean_latency_s", "mean_energy_j", "success_rate",
                "mean_route_augmentation_count",
            ):
                mean, ci = mean_ci([row[metric] for row in samples])
                record[metric] = mean
                record[f"{metric}_ci95"] = ci
            records.append(record)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = resolve_summary(args.input)
    rows = load_rows(summary)
    output_dir = (args.output_dir or summary.parent / "sensitivity_plots").resolve()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.5,
        }
    )
    generated = []
    weight_groups = grouped(rows, "latency_energy_weights")
    weight_labels = sorted(
        weight_groups,
        key=lambda label: weight_groups[label][0]["delay_weight"],
    )
    route_groups = grouped(rows, "routing_max_paths")
    route_labels = sorted(
        route_groups,
        key=lambda label: route_groups[label][0]["route_max_paths"],
    )
    if weight_groups:
        labels = [
            f"{weight_groups[label][0]['delay_weight']:.2f}:"
            f"{weight_groups[label][0]['energy_weight']:.2f}"
            for label in weight_labels
        ]
        display_groups = {
            display: weight_groups[key] for display, key in zip(labels, weight_labels)
        }
        generated.extend(
            plot_metric_panels(
                display_groups, labels,
                (
                    ("mean_latency_s", "Mean latency", "Seconds", 1.0),
                    ("mean_energy_j", "Mean energy", "Joules", 1.0),
                    ("success_rate", "Success rate", "Percent", 100.0),
                ),
                "Latency and Energy Weight Sensitivity",
                output_dir / "latency_energy_weights", args.formats, args.dpi,
            )
        )
        generated.extend(
            plot_weight_tradeoff(
                display_groups, labels, output_dir / "latency_energy_weights",
                args.formats, args.dpi,
            )
        )
    if route_groups:
        display_labels = [label.split("_")[-1] for label in route_labels]
        display_groups = {
            display: route_groups[key] for display, key in zip(display_labels, route_labels)
        }
        generated.extend(
            plot_metric_panels(
                display_groups, display_labels,
                (
                    ("mean_latency_s", "Mean latency", "Seconds", 1.0),
                    ("mean_energy_j", "Mean energy", "Joules", 1.0),
                    ("success_rate", "Success rate", "Percent", 100.0),
                ),
                "Maximum Augmenting Path Sensitivity",
                output_dir / "routing_max_paths", args.formats, args.dpi,
            )
        )
        generated.extend(
            plot_route_overhead(
                display_groups, display_labels,
                output_dir / "routing_max_paths", args.formats, args.dpi,
            )
        )
    statistics = write_statistics(rows, output_dir)
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "summary": str(summary),
                "generated": [str(path) for path in generated],
                "statistics": str(statistics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"sensitivity summary: {summary}")
    print(f"plots: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
