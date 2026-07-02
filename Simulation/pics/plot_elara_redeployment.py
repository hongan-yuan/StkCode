from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ..ablation_names import ABLATION_LABELS, canonical_ablation_name
except ImportError:  # pragma: no cover - allows direct script execution
    from Simulation.ablation_names import ABLATION_LABELS, canonical_ablation_name


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
DEFAULT_REDEPLOY_DIR = (
    ROOT_DIR / "Simulation" / "test_outputs" / "bandit_redeployment_replay_experiments"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "elara_paper"
DEFAULT_PAPER_FIG_DIR = ROOT_DIR / "MyPaper" / "elara_exp"

REDEPLOY_METHODS = ("ELARA", "ELARA-NB")

FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20
AXIS_LINE_WIDTH = 1.0
LINE_WIDTH = 2.8
MARKER_SIZE = 4.0
GRID_LINE_WIDTH = 0.55

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
    "SECO": "#0f766e",
    "SP-Routing": "#0891b2",
    "SC-NFV": "#6b7280",
    "Latency": "#2f6fbb",
    "Energy": "#d9822b",
    "add": "#2f9e44",
    "move": "#2f6fbb",
    "remove": "#d9822b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw ELARA experiment-section figures for INFOCOM manuscript."
    )
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--redeploy-dir", type=Path, default=DEFAULT_REDEPLOY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument(
        "--no-copy-to-paper",
        action="store_true",
        help="Do not copy generated figures into MyPaper/elara_exp.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_method_list(value: str) -> list[str]:
    methods: list[str] = []
    seen = set()
    for item in value.replace(",", " ").split():
        method = canonical_ablation_name(item)
        if method and method not in seen:
            methods.append(method)
            seen.add(method)
    return methods


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "ablation" in row:
            row["ablation"] = canonical_ablation_name(row["ablation"])
    return rows


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def load_slot_rows(ablation_dir: Path, methods: list[str]) -> list[dict]:
    rows = read_rows(ablation_dir / "all_ablation_slot_metrics.csv")
    if rows:
        return [row for row in rows if row.get("ablation") in methods]

    collected: list[dict] = []
    for method in methods:
        collected.extend(read_rows(ablation_dir / method / "slot_metrics_by_seed.csv"))
    return [row for row in collected if row.get("ablation") in methods]


def load_hop_rows(ablation_dir: Path, methods: list[str]) -> list[dict]:
    rows = read_rows(ablation_dir / "all_ablation_request_hop_metrics.csv")
    if rows:
        return [row for row in rows if row.get("ablation") in methods]

    collected: list[dict] = []
    for method in methods:
        collected.extend(read_rows(ablation_dir / method / "request_hop_metrics_by_seed.csv"))
    return [row for row in collected if row.get("ablation") in methods]


def window_mean_series(
    rows: list[dict],
    methods: list[str],
    metric: str,
    slot_window: int,
    max_slot: int,
) -> tuple[list[int], dict[str, list[float]]]:
    window_count = max(1, max_slot // slot_window)
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        method = row.get("ablation")
        epoch = number(row, "epoch")
        value = number(row, metric)
        if method not in methods or epoch is None or value is None:
            continue
        slot = int(epoch)
        if slot < 1 or slot > max_slot:
            continue
        window_index = (slot - 1) // slot_window + 1
        buckets[(method, window_index)].append(value)

    windows = list(range(1, window_count + 1))
    series: dict[str, list[float]] = {}
    for method in methods:
        values: list[float] = []
        for window_index in windows:
            bucket = buckets.get((method, window_index), [])
            values.append(float(np.mean(bucket)) if bucket else math.nan)
        series[method] = values
    return windows, series


def plot_window_metric(
    ax,
    rows: list[dict],
    methods: list[str],
    metric: str,
    ylabel: str,
    panel_label: str,
    slot_window: int,
    max_slot: int,
) -> None:
    windows, series = window_mean_series(rows, methods, metric, slot_window, max_slot)
    for method in methods:
        ax.plot(
            windows,
            series[method],
            color=COLORS.get(method, "#4b5563"),
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markevery=max(1, len(windows) // 12),
            label=ABLATION_LABELS.get(method, method),
        )
    ax.set_xlabel(f"{slot_window}-slot window index")
    ax.set_xlim(1, len(windows))
    tick_step = 10 if len(windows) > 40 else 5
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
    style_axis(ax, ylabel)


def slot_crossing_distribution(
    rows: list[dict],
    methods: list[str],
) -> dict[str, list[float]]:
    counts: dict[str, Counter[str]] = {method: Counter() for method in methods}
    for row in rows:
        method = row.get("ablation")
        value = number(row, "slot_crossings")
        if method not in methods or value is None:
            continue
        crossing = int(round(value))
        if crossing <= 0:
            bucket = "0"
        elif crossing == 1:
            bucket = "1"
        else:
            bucket = ">=2"
        counts[method][bucket] += 1

    distribution: dict[str, list[float]] = {}
    for method in methods:
        total = sum(counts[method].values())
        distribution[method] = [
            100.0 * counts[method][bucket] / total if total else 0.0
            for bucket in ("0", "1", ">=2")
        ]
    return distribution


def plot_slot_crossing_distribution(
    ax,
    rows: list[dict],
    methods: list[str],
) -> None:
    available_methods = [
        method
        for method in methods
        if any(row.get("ablation") == method for row in rows)
    ]
    distribution = slot_crossing_distribution(rows, available_methods)
    categories = ["0", "1", ">=2"]
    x = np.arange(len(categories))
    width = min(0.26, 0.78 / max(1, len(available_methods)))
    offsets = (
        np.arange(len(available_methods)) - (len(available_methods) - 1) / 2.0
    ) * width
    for offset, method in zip(offsets, available_methods):
        ax.bar(
            x + offset,
            distribution[method],
            width=width,
            color=COLORS.get(method, "#4b5563"),
            label=ABLATION_LABELS.get(method, method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Slot crossings per service stage")
    ax.text(
        0.015,
        0.97,
        "(c) Crossing distribution",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )
    style_axis(ax, "Fraction of stages (%)")


def save_cross_slot_routing_figure(
    output_path: Path,
    slot_rows: list[dict],
    hop_rows: list[dict],
    methods: list[str],
    slot_window: int,
    max_slot: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    plot_window_metric(
        axes[0],
        slot_rows,
        methods,
        "average_communication_delay_s",
        "Communication delay (s)",
        "(a) Communication delay",
        slot_window,
        max_slot,
    )
    plot_window_metric(
        axes[1],
        slot_rows,
        methods,
        "average_slot_crossings",
        "Average slot crossings",
        "(b) Slot crossings",
        slot_window,
        max_slot,
    )
    plot_slot_crossing_distribution(axes[2], hop_rows, methods)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(methods),
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_redeployment_window_rows(redeploy_dir: Path) -> list[dict]:
    rows = read_rows(redeploy_dir / "all_redeployment_window_metrics.csv")
    if rows:
        return [row for row in rows if row.get("ablation") in REDEPLOY_METHODS]
    collected: list[dict] = []
    for method in REDEPLOY_METHODS:
        collected.extend(read_rows(redeploy_dir / method / "redeployment_window_metrics_by_seed.csv"))
    return [row for row in collected if row.get("ablation") in REDEPLOY_METHODS]


def redeployment_reduction_series(
    rows: list[dict],
) -> tuple[list[int], dict[str, list[float]]]:
    grouped: dict[tuple[int, int, str], dict] = {}
    for row in rows:
        method = row.get("ablation")
        seed = number(row, "seed")
        window = number(row, "window_index")
        if method not in REDEPLOY_METHODS or seed is None or window is None:
            continue
        grouped[(int(seed), int(window), method)] = row

    reductions: dict[tuple[int, str], list[float]] = defaultdict(list)
    windows = sorted({key[1] for key in grouped})
    for seed, window, method in list(grouped):
        if method != "ELARA":
            continue
        elara = grouped.get((seed, window, "ELARA"))
        no_bandit = grouped.get((seed, window, "ELARA-NB"))
        if not elara or not no_bandit:
            continue
        for metric, label in (
            ("average_end_to_end_delay_s", "Latency"),
            ("average_energy_j", "Energy"),
        ):
            base = number(no_bandit, metric)
            value = number(elara, metric)
            if base is None or value is None or abs(base) < 1.0e-12:
                continue
            reductions[(window, label)].append(100.0 * (base - value) / base)

    series = {"Latency": [], "Energy": []}
    for window in windows:
        for label in ("Latency", "Energy"):
            bucket = reductions.get((window, label), [])
            series[label].append(float(np.mean(bucket)) if bucket else math.nan)
    return windows, series


def smooth_nan_series(values: list[float], window: int = 5) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values
    half_window = window // 2
    array = np.array(values, dtype=float)
    smoothed: list[float] = []
    for index, value in enumerate(array):
        start = max(0, index - half_window)
        end = min(len(array), index + half_window + 1)
        local = array[start:end]
        finite = local[np.isfinite(local)]
        smoothed.append(float(np.mean(finite)) if len(finite) else float(value))
    return smoothed


def plot_redeployment_reductions(ax, rows: list[dict]) -> None:
    windows, series = redeployment_reduction_series(rows)
    for label in ("Latency", "Energy"):
        values = smooth_nan_series(series[label], window=7)
        ax.plot(
            windows,
            values,
            color=COLORS[label],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markevery=max(1, len(windows) // 12),
            label=f"{label} reduction",
        )
    ax.axhline(0.0, color="#111827", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Deployment window index")
    ax.set_xlim(1, max(windows) if windows else 1)
    style_axis(ax, "Reduction over ELARA-NB (%)")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="best")


def load_migration_action_rows(redeploy_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((redeploy_dir / "ELARA").glob("seed_*/migration_actions.csv")):
        rows.extend(read_rows(path))
    return rows


def plot_redeployment_action_distribution(ax, rows: list[dict]) -> None:
    counts = Counter(row.get("action", "") for row in rows if row.get("action"))
    actions = [action for action in ("add", "move", "remove") if counts[action] > 0]
    if not actions:
        actions = ["add", "move", "remove"]
    values = [counts[action] for action in actions]
    bars = ax.bar(
        actions,
        values,
        color=[COLORS.get(action, "#4b5563") for action in actions],
        width=0.58,
    )
    total = sum(values)
    for bar, value in zip(bars, values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_FONT_SIZE,
        )
    if values:
        ax.set_ylim(0.0, max(values) * 1.18)
    ax.set_xlabel("Redeployment action")
    style_axis(ax, "Number of actions")


def save_redeployment_adaptivity_figure(
    output_path: Path,
    window_rows: list[dict],
    action_rows: list[dict],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    plot_redeployment_reductions(axes[0], window_rows)
    plot_redeployment_action_distribution(axes[1], action_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [output_path, pdf_path]


def copy_to_paper(paths: list[Path], paper_fig_dir: Path) -> None:
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, paper_fig_dir / path.name)


def main() -> None:
    args = parse_args()
    configure_style()
    window_rows = load_redeployment_window_rows(args.redeploy_dir)
    action_rows = load_migration_action_rows(args.redeploy_dir)

    redeploy_path = args.output_dir / "redeployment_adaptivity.png"
    outputs = save_redeployment_adaptivity_figure(redeploy_path, window_rows, action_rows)
    if not args.no_copy_to_paper:
        copy_to_paper(outputs, args.paper_fig_dir)

    manifest = args.output_dir / "plot_elara_experiment_section_figures_manifest.txt"
    manifest.write_text(
        "\n".join(str(path.resolve()) for path in outputs) + "\n",
        encoding="utf-8",
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
