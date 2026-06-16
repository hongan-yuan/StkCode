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

from ..ablation_names import ABLATION_LABELS, canonical_ablation_names, canonicalize_ablation_row


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT_DIR / "Simulation" / "test_outputs" / "bandit_redeployment_replay_experiments"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"
DEFAULT_ABLATIONS = "ELARA ELARA-NB"

FONT_FAMILY = "Times New Roman"
LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 11
TICK_LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE = 12
AXIS_LINE_WIDTH = 1.0
BAR_EDGE_LINE_WIDTH = 0.8
LINE_WIDTH = 2.2
MARKER_SIZE = 4.5

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 10-slot window metrics for the repeated service redeployment "
            "experiment."
        )
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


def load_window_rows(input_dir: Path, ablations: list[str]) -> list[dict]:
    rows = read_rows(input_dir / "all_redeployment_window_metrics.csv")
    if not rows:
        rows = []
        for ablation in ablations:
            for row in read_rows(input_dir / ablation / "redeployment_window_metrics_by_seed.csv"):
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


def grouped_window_values(
    rows: list[dict],
    ablations: list[str],
    metric: str,
) -> tuple[list[int], dict[str, dict[int, list[float]]]]:
    grouped: dict[str, dict[int, list[float]]] = {
        ablation: defaultdict(list) for ablation in ablations
    }
    windows = set()
    for row in rows:
        ablation = row.get("ablation")
        window = number(row, "window_index")
        value = number(row, metric)
        if ablation not in grouped or window is None or value is None:
            continue
        window_index = int(window)
        windows.add(window_index)
        grouped[ablation][window_index].append(value)
    return sorted(windows), grouped


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def label_for(ablation: str) -> str:
    return ABLATION_LABELS.get(ablation, ablation)


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_xlabel("10-slot window index")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def window_ticks(windows: list[int]) -> list[int]:
    if len(windows) <= 12:
        return windows
    step = 10 if max(windows) >= 50 else 5
    ticks = [windows[0]]
    ticks.extend(window for window in windows if window % step == 0)
    if windows[-1] not in ticks:
        ticks.append(windows[-1])
    return sorted(dict.fromkeys(ticks))


def plot_metric_curve(
    ax,
    rows: list[dict],
    ablations: list[str],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    windows, grouped = grouped_window_values(rows, ablations, metric)
    for ablation in ablations:
        means = [mean(grouped[ablation].get(window, [])) for window in windows]
        errors = [stderr(grouped[ablation].get(window, [])) for window in windows]
        ax.errorbar(
            windows,
            means,
            yerr=errors,
            color=COLORS.get(ablation, "#4b5563"),
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            capsize=3,
            label=label_for(ablation),
        )
    ax.set_xticks(window_ticks(windows))
    style_axes(ax, title, ylabel)
    ax.legend(frameon=False, loc="best")


def plot_action_curve(
    output_path: Path,
    rows: list[dict],
    ablations: list[str],
) -> None:
    windows, grouped = grouped_window_values(rows, ablations, "redeployment_action_count")
    fig, ax = plt.subplots(figsize=(12.8, 4.8), constrained_layout=True)
    width = 0.34
    x = np.arange(len(windows))
    offset_start = -width * (len(ablations) - 1) / 2.0
    for index, ablation in enumerate(ablations):
        means = [mean(grouped[ablation].get(window, [])) for window in windows]
        ax.bar(
            x + offset_start + index * width,
            means,
            width,
            color=COLORS.get(ablation, "#4b5563"),
            edgecolor="#222222",
            linewidth=BAR_EDGE_LINE_WIDTH,
            label=label_for(ablation),
        )
    tick_windows = window_ticks(windows)
    tick_positions = [windows.index(window) for window in tick_windows]
    ax.set_xticks(tick_positions, [str(window) for window in tick_windows])
    style_axes(ax, "Redeployment actions after each window", "Mean action count")
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_metric_summary(output_path: Path, rows: list[dict], ablations: list[str]) -> None:
    metrics = (
        "average_end_to_end_delay_s",
        "average_energy_j",
        "redeployment_action_count",
    )
    windows = sorted(
        {
            int(window)
            for row in rows
            if (window := number(row, "window_index")) is not None
        }
    )
    output_rows = []
    for ablation in ablations:
        ablation_rows = [row for row in rows if row.get("ablation") == ablation]
        for window in windows:
            row = {
                "ablation": ablation,
                "label": label_for(ablation),
                "window_index": window,
            }
            window_rows = [
                item
                for item in ablation_rows
                if int(number(item, "window_index") or -1) == window
            ]
            for metric in metrics:
                values = [value for item in window_rows if (value := number(item, metric)) is not None]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_stderr"] = stderr(values)
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
    rows = load_window_rows(args.input_dir, ablations)
    if not rows:
        raise SystemExit(f"No redeployment window rows found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    window_count = len({int(number(row, "window_index") or 0) for row in rows})
    figure_width = 13.2 if window_count > 20 else 9.6
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 8.2), constrained_layout=True)
    plot_metric_curve(
        axes[0],
        rows,
        ablations,
        "average_end_to_end_delay_s",
        "Mean end-to-end delay across repeated 10-slot windows",
        "Mean delay (s)",
    )
    plot_metric_curve(
        axes[1],
        rows,
        ablations,
        "average_energy_j",
        "Mean energy across repeated 10-slot windows",
        "Mean energy (J)",
    )
    metric_path = args.output_dir / f"redeployment_window_metrics.{args.format}"
    fig.savefig(metric_path, bbox_inches="tight")
    plt.close(fig)

    action_path = args.output_dir / f"redeployment_action_counts.{args.format}"
    plot_action_curve(action_path, rows, ablations)

    summary_path = args.output_dir / "redeployment_plot_summary.csv"
    write_metric_summary(summary_path, rows, ablations)
    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(
        f"{metric_path.resolve()}\n{action_path.resolve()}\n{summary_path.resolve()}\n",
        encoding="utf-8",
    )
    print(metric_path)
    print(action_path)
    print(summary_path)


if __name__ == "__main__":
    main()
