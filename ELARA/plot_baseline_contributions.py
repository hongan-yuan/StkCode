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


ABLATION_METHODS = ("ELARA", "ELARA-NB", "ELARA-NR", "ELARA-SH")
COMPARISON_METHODS = ("ELARA", "SECO", "SP-Routing", "SC-NFV")
ALL_METHODS = tuple(dict.fromkeys((*ABLATION_METHODS, *COMPARISON_METHODS)))
METHOD_COLORS = {
    "ELARA": "#1f77b4",
    "ELARA-NB": "#ff7f0e",
    "ELARA-NR": "#2ca02c",
    "ELARA-SH": "#9467bd",
    "SECO": "#ff7f0e",
    "SP-Routing": "#2ca02c",
    "SC-NFV": "#9467bd",
}
METHOD_MARKERS = {
    "ELARA": "o",
    "ELARA-NB": "s",
    "ELARA-NR": "^",
    "ELARA-SH": "D",
    "SECO": "s",
    "SP-Routing": "^",
    "SC-NFV": "D",
}
REQUIRED_FILES = (
    "all_ablation_cycle_metrics.csv",
    "all_ablation_request_metrics.csv",
    "all_ablation_request_hop_metrics.csv",
    "all_ablation_slot_metrics.csv",
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ELARA baseline-test data and plot ablation and comparison "
            "experiments in separate directories."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("ELARA/outputs/baseline-tests"),
        help="one completed baseline run or the parent baseline-tests directory",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)
    if args.rolling_window < 1:
        parser.error("--rolling-window must be at least 1")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    args.formats = tuple(
        dict.fromkeys(item.strip().lower() for item in args.formats.split(",") if item.strip())
    )
    if not args.formats or set(args.formats) - {"png", "pdf", "svg"}:
        parser.error("--formats must contain one or more of: png,pdf,svg")
    return args


def _is_run_root(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def resolve_run_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if _is_run_root(path):
        return path
    if not path.is_dir():
        raise ValueError(f"input directory does not exist: {path}")
    candidates = sorted(
        (candidate for candidate in path.iterdir() if _is_run_root(candidate)),
        key=lambda candidate: candidate.name,
    )
    if not candidates:
        raise ValueError(f"no completed baseline-test run found below {path}")
    return candidates[-1]


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def finite(value) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean_ci95(values) -> tuple[float, float]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if len(array) == 0:
        return math.nan, math.nan
    mean = float(np.mean(array))
    ci = float(1.96 * np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, ci


def load_cycle_metrics(run_root: Path):
    result = defaultdict(dict)
    for row in read_rows(run_root / "all_ablation_cycle_metrics.csv"):
        method = row.get("ablation", "")
        seed = int(float(row["seed"]))
        result[method][seed] = {
            key: finite(value)
            for key, value in row.items()
            if key not in {"ablation", "seed"}
        }
    missing = [method for method in ALL_METHODS if method not in result]
    if missing:
        raise ValueError(f"cycle metrics are missing methods: {', '.join(missing)}")
    return result


def load_request_data(run_root: Path):
    chain = defaultdict(lambda: defaultdict(list))
    distributions = defaultdict(lambda: defaultdict(list))
    for row in read_rows(run_root / "all_ablation_request_metrics.csv"):
        method = row.get("ablation", "")
        if method not in ALL_METHODS or not truthy(row.get("feasible")):
            continue
        seed = int(float(row["seed"]))
        chain_length = int(float(row["chain_length"]))
        delay = finite(row.get("total_delay_s"))
        energy = finite(row.get("total_energy_j"))
        if delay is not None:
            chain[(method, seed, chain_length)]["delay"].append(delay)
            distributions[method]["delay"].append(delay)
        if energy is not None:
            chain[(method, seed, chain_length)]["energy"].append(energy)
            distributions[method]["energy"].append(energy)
    return chain, distributions


def load_slot_data(run_root: Path):
    values = defaultdict(lambda: defaultdict(dict))
    for row in read_rows(run_root / "all_ablation_slot_metrics.csv"):
        method = row.get("ablation", "")
        if method not in ALL_METHODS:
            continue
        seed = int(float(row["seed"]))
        slot = int(float(row["slot_mod"]))
        for metric, column in (
            ("delay", "average_end_to_end_delay_s"),
            ("energy", "average_energy_j"),
        ):
            number = finite(row.get(column))
            if number is not None:
                values[(method, seed)][metric][slot] = number
    return values


def load_decomposition(run_root: Path):
    totals = defaultdict(lambda: np.zeros(4, dtype=float))
    request_ids = defaultdict(set)
    for row in read_rows(run_root / "all_ablation_request_hop_metrics.csv"):
        method = row.get("ablation", "")
        if method not in ALL_METHODS:
            continue
        seed = int(float(row["seed"]))
        key = (method, seed)
        request_ids[key].add(row.get("request_id", ""))
        columns = (
            "communication_delay_s",
            "compute_total_delay_s",
            "communication_energy_j",
            "compute_energy_j",
        )
        for index, column in enumerate(columns):
            number = finite(row.get(column))
            if number is not None:
                totals[key][index] += number
    result = {}
    for key, components in totals.items():
        count = max(1, len(request_ids[key]))
        result[key] = components / count
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.2,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, directory: Path, stem: str, formats, dpi: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def cycle_values(cycle, method: str, metric: str):
    return [metrics.get(metric) for metrics in cycle[method].values()]


def _bar_panel(ax, cycle, methods, metric, title, ylabel, percent=False, log_scale=False):
    means, errors = [], []
    for method in methods:
        values = cycle_values(cycle, method, metric)
        if percent:
            values = [100.0 * value if value is not None else None for value in values]
        mean, error = mean_ci95(values)
        means.append(mean)
        errors.append(error)
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=3,
        width=0.68,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="black",
        linewidth=0.55,
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x, methods, rotation=18 if any(len(item) > 8 for item in methods) else 0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.45, alpha=0.25)
    for bar, value in zip(bars, means):
        label = f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def plot_overall(cycle, methods, title_prefix, directory, formats, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), constrained_layout=True)
    panels = (
        ("average_end_to_end_delay_s", "Mean latency", "Seconds"),
        ("p95_end_to_end_delay_s", "P95 latency", "Seconds"),
        ("average_energy_j", "Mean energy", "Joules"),
    )
    for ax, (metric, title, ylabel) in zip(axes, panels):
        _bar_panel(ax, cycle, methods, metric, title, ylabel)
    fig.suptitle(f"{title_prefix}: Overall Performance", fontsize=11.5)
    return save_figure(fig, directory, "overall_performance", formats, dpi)


def plot_routing_reliability(cycle, methods, title_prefix, directory, formats, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), constrained_layout=True)
    panels = (
        ("average_communication_delay_s", "Communication latency", "Seconds", False),
        ("average_slot_crossings", "Cross-slot routing", "Mean slot crossings", False),
        ("failure_count", "Failed requests", "Requests per cycle", True),
    )
    for ax, (metric, title, ylabel, log_scale) in zip(axes, panels):
        _bar_panel(ax, cycle, methods, metric, title, ylabel, log_scale=log_scale)
    fig.suptitle(f"{title_prefix}: Routing and Reliability", fontsize=11.5)
    return save_figure(fig, directory, "routing_reliability", formats, dpi)


def plot_relative_contribution(cycle, methods, title_prefix, directory, formats, dpi):
    baselines = methods[1:]
    metrics = (
        ("average_end_to_end_delay_s", "Mean latency"),
        ("p95_end_to_end_delay_s", "P95 latency"),
        ("average_energy_j", "Mean energy"),
    )
    reference = {
        metric: mean_ci95(cycle_values(cycle, "ELARA", metric))[0]
        for metric, _ in metrics
    }
    x = np.arange(len(baselines))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.8, 3.65), constrained_layout=True)
    for index, (metric, label) in enumerate(metrics):
        reductions = []
        for method in baselines:
            baseline = mean_ci95(cycle_values(cycle, method, metric))[0]
            reductions.append(improvement(reference[metric], baseline))
        positions = x + (index - 1) * width
        bars = ax.bar(positions, reductions, width=width, label=label)
        for bar, value in zip(bars, reductions):
            ax.annotate(
                f"{value:+.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if value >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.5,
            )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, baselines)
    ax.set_ylabel("Reduction achieved by ELARA (%)")
    ax.set_title(f"{title_prefix}: Relative Contribution")
    ax.grid(axis="y", linewidth=0.45, alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return save_figure(fig, directory, "relative_contribution", formats, dpi)


def chain_seed_means(chain, method, chain_length, metric):
    values = []
    seeds = sorted({key[1] for key in chain if key[0] == method and key[2] == chain_length})
    for seed in seeds:
        samples = chain[(method, seed, chain_length)][metric]
        if samples:
            values.append(float(np.mean(samples)))
    return values


def plot_chain_length(chain, methods, title_prefix, directory, formats, dpi):
    chain_lengths = (5, 10, 15)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), constrained_layout=True)
    for ax, metric, ylabel, title in (
        (axes[0], "delay", "Seconds", "End-to-end latency"),
        (axes[1], "energy", "Joules", "Energy consumption"),
    ):
        for method in methods:
            means, errors = [], []
            for chain_length in chain_lengths:
                mean, error = mean_ci95(chain_seed_means(chain, method, chain_length, metric))
                means.append(mean)
                errors.append(error)
            ax.errorbar(
                chain_lengths,
                means,
                yerr=errors,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.6,
                markersize=4.5,
                capsize=2.5,
                label=method,
            )
        ax.set_xticks(chain_lengths)
        ax.set_xlabel("Service chain length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.45, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(f"{title_prefix}: Chain-Length Sensitivity", fontsize=11.5)
    return save_figure(fig, directory, "chain_length_sensitivity", formats, dpi)


def nan_moving_average(values: np.ndarray, window: int):
    result = np.full(len(values), np.nan, dtype=float)
    for index in range(len(values)):
        segment = values[max(0, index + 1 - window) : index + 1]
        finite_values = segment[np.isfinite(segment)]
        if len(finite_values):
            result[index] = float(np.mean(finite_values))
    return result


def temporal_matrix(slot_data, method, metric, window):
    keys = sorted(key for key in slot_data if key[0] == method)
    max_slot = max(
        (max(slot_data[key][metric], default=-1) for key in keys), default=-1
    )
    matrix = np.full((len(keys), max_slot + 1), np.nan, dtype=float)
    for row_index, key in enumerate(keys):
        for slot, value in slot_data[key][metric].items():
            matrix[row_index, slot] = value
        matrix[row_index] = nan_moving_average(matrix[row_index], window)
    return matrix


def plot_temporal(slot_data, methods, title_prefix, directory, formats, dpi, window):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    for ax, metric, ylabel, title in (
        (axes[0], "delay", "Seconds", "Latency over the constellation cycle"),
        (axes[1], "energy", "Joules", "Energy over the constellation cycle"),
    ):
        for method in methods:
            matrix = temporal_matrix(slot_data, method, metric, window)
            mean = np.nanmean(matrix, axis=0)
            counts = np.sum(np.isfinite(matrix), axis=0)
            std = np.nanstd(matrix, axis=0, ddof=1)
            ci = np.nan_to_num(1.96 * std / np.sqrt(np.maximum(counts, 1)))
            x = np.arange(len(mean))
            ax.plot(x, mean, color=METHOD_COLORS[method], linewidth=1.35, label=method)
            ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[method], alpha=0.08)
        ax.set_xlabel("Time slot")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.45, alpha=0.22)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(f"{title_prefix}: Temporal Robustness ({window}-slot average)", fontsize=11.5)
    return save_figure(fig, directory, "temporal_robustness", formats, dpi)


def plot_tail_distributions(distributions, methods, directory, formats, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    for ax, metric, xlabel, title in (
        (axes[0], "delay", "End-to-end latency (s)", "Latency tail"),
        (axes[1], "energy", "Energy consumption (J)", "Energy tail"),
    ):
        combined = []
        for method in methods:
            values = np.sort(np.asarray(distributions[method][metric], dtype=float))
            combined.extend(values.tolist())
            if not len(values):
                continue
            indices = np.unique(np.linspace(0, len(values) - 1, min(2500, len(values))).astype(int))
            survival = 1.0 - (indices + 1) / (len(values) + 1)
            ax.plot(
                values[indices], survival, color=METHOD_COLORS[method],
                linewidth=1.5, label=method,
            )
        if combined:
            ax.set_xlim(0, float(np.percentile(combined, 99.5)))
        ax.set_yscale("log")
        ax.set_ylim(1.0e-3, 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Tail probability")
        ax.set_title(title)
        ax.grid(True, which="both", linewidth=0.45, alpha=0.24)
    axes[0].legend(frameon=False)
    fig.suptitle("Comparison: Request-Level Tail Distributions", fontsize=11.5)
    return save_figure(fig, directory, "tail_distributions", formats, dpi)


def plot_decomposition(decomposition, methods, directory, formats, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65), constrained_layout=True)
    for ax, first, second, ylabel, title in (
        (axes[0], 0, 1, "Seconds", "Latency decomposition"),
        (axes[1], 2, 3, "Joules", "Energy decomposition"),
    ):
        communication, computation, total_errors = [], [], []
        for method in methods:
            rows = [components for (name, _), components in decomposition.items() if name == method]
            communication.append(float(np.mean([row[first] for row in rows])))
            computation.append(float(np.mean([row[second] for row in rows])))
            _, error = mean_ci95([row[first] + row[second] for row in rows])
            total_errors.append(error)
        x = np.arange(len(methods))
        ax.bar(x, communication, width=0.68, color="#4c78a8", label="Communication")
        ax.bar(x, computation, width=0.68, bottom=communication, color="#f2a541", label="Computation")
        totals = np.asarray(communication) + np.asarray(computation)
        ax.errorbar(x, totals, yerr=total_errors, fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax.set_xticks(x, methods, rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.45, alpha=0.24)
    axes[0].legend(frameon=False)
    fig.suptitle("Comparison: Communication and Computation Costs", fontsize=11.5)
    return save_figure(fig, directory, "cost_decomposition", formats, dpi)


def improvement(reference: float, baseline: float) -> float:
    return 100.0 * (baseline - reference) / baseline if baseline else math.nan


def write_analysis(run_root, output_dir, cycle) -> tuple[Path, Path, dict]:
    metric_map = {
        "mean_latency_s": "average_end_to_end_delay_s",
        "p95_latency_s": "p95_end_to_end_delay_s",
        "mean_energy_j": "average_energy_j",
        "communication_delay_s": "average_communication_delay_s",
        "slot_crossings": "average_slot_crossings",
        "failure_count": "failure_count",
        "completion_rate": "task_completion_rate",
    }
    rows = []
    means = {}
    for method in ALL_METHODS:
        means[method] = {}
        row = {"method": method, "seed_count": len(cycle[method])}
        for output_name, source_name in metric_map.items():
            mean, ci = mean_ci95(cycle_values(cycle, method, source_name))
            row[output_name] = mean
            row[f"{output_name}_ci95"] = ci
            means[method][output_name] = mean
        rows.append(row)
    csv_path = output_dir / "contribution_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latency_reference = means["ELARA"]["mean_latency_s"]
    p95_reference = means["ELARA"]["p95_latency_s"]
    energy_reference = means["ELARA"]["mean_energy_j"]
    analysis = {
        "run_root": str(run_root),
        "methods": list(ALL_METHODS),
        "seeds": sorted(cycle["ELARA"]),
        "ablation": {},
        "comparison": {},
    }
    for category, methods in (
        ("ablation", ABLATION_METHODS[1:]),
        ("comparison", COMPARISON_METHODS[1:]),
    ):
        for method in methods:
            analysis[category][method] = {
                "mean_latency_reduction_percent": improvement(
                    latency_reference, means[method]["mean_latency_s"]
                ),
                "p95_latency_reduction_percent": improvement(
                    p95_reference, means[method]["p95_latency_s"]
                ),
                "energy_reduction_percent": improvement(
                    energy_reference, means[method]["mean_energy_j"]
                ),
                "completion_rate_percentage_points": 100.0 * (
                    means["ELARA"]["completion_rate"] - means[method]["completion_rate"]
                ),
            }
    json_path = output_dir / "contribution_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return csv_path, json_path, analysis


def write_markdown_summary(output_dir: Path, analysis) -> Path:
    path = output_dir / "contribution_analysis.md"
    lines = [
        "# ELARA Baseline Contribution Analysis",
        "",
        f"Seeds: {', '.join(map(str, analysis['seeds']))}",
        "",
        "## Ablation experiments",
        "",
    ]
    for method, values in analysis["ablation"].items():
        lines.append(
            f"- Versus {method}, ELARA changes mean latency by "
            f"{values['mean_latency_reduction_percent']:+.2f}%, P95 latency by "
            f"{values['p95_latency_reduction_percent']:+.2f}%, and energy by "
            f"{values['energy_reduction_percent']:+.2f}%."
        )
    lines.extend(("", "## Comparison experiments", ""))
    for method, values in analysis["comparison"].items():
        lines.append(
            f"- Versus {method}, ELARA changes mean latency by "
            f"{values['mean_latency_reduction_percent']:+.2f}%, P95 latency by "
            f"{values['p95_latency_reduction_percent']:+.2f}%, and energy by "
            f"{values['energy_reduction_percent']:+.2f}%."
        )
    lines.extend(
        (
            "",
            "Positive reduction values favor ELARA. Negative energy reduction values indicate an energy tradeoff.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_root = resolve_run_root(args.input)
        cycle = load_cycle_metrics(run_root)
        chain, distributions = load_request_data(run_root)
        slot_data = load_slot_data(run_root)
        decomposition = load_decomposition(run_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = (args.output_dir or run_root / "contribution_plots").expanduser().resolve()
    ablation_dir = output_dir / "ablation"
    comparison_dir = output_dir / "comparison"
    _style()
    generated = []
    for methods, title, directory in (
        (ABLATION_METHODS, "Ablation", ablation_dir),
        (COMPARISON_METHODS, "Comparison", comparison_dir),
    ):
        generated.extend(plot_overall(cycle, methods, title, directory, args.formats, args.dpi))
        generated.extend(
            plot_routing_reliability(cycle, methods, title, directory, args.formats, args.dpi)
        )
        generated.extend(
            plot_relative_contribution(cycle, methods, title, directory, args.formats, args.dpi)
        )
        generated.extend(plot_chain_length(chain, methods, title, directory, args.formats, args.dpi))
        generated.extend(
            plot_temporal(
                slot_data, methods, title, directory, args.formats, args.dpi, args.rolling_window
            )
        )
    generated.extend(
        plot_tail_distributions(
            distributions, COMPARISON_METHODS, comparison_dir, args.formats, args.dpi
        )
    )
    generated.extend(
        plot_decomposition(
            decomposition, COMPARISON_METHODS, comparison_dir, args.formats, args.dpi
        )
    )
    csv_path, json_path, analysis = write_analysis(run_root, output_dir, cycle)
    markdown_path = write_markdown_summary(output_dir, analysis)
    manifest = {
        "run_root": str(run_root),
        "ablation_methods": list(ABLATION_METHODS),
        "comparison_methods": list(COMPARISON_METHODS),
        "rolling_window": args.rolling_window,
        "generated": [str(path) for path in generated],
        "analysis_files": [str(csv_path), str(json_path), str(markdown_path)],
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Analyzed baseline run: {run_root}")
    print(f"Ablation plots: {ablation_dir}")
    print(f"Comparison plots: {comparison_dir}")
    print(f"Analysis summary: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
