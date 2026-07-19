from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..ablation_names import ABLATION_LABELS, canonical_ablation_name


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_ROOT = ROOT_DIR / "Simulation" / "multi_seed_runs"
DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "elara_paper"

TRAIN_REWARD_COLUMN = "average_reward_per_request"
TRAIN_LOSS_COLUMN = "ppo_loss"
LATENCY_COLUMN = "mean_average_end_to_end_delay_s"
ENERGY_COLUMN = "mean_average_energy_j"

COMPARISON_ABLATIONS = ("ELARA", "SECO", "SC-NFV", "SP-Routing")
ABLATION_ABLATIONS = ("ELARA", "ELARA-NB", "ELARA-NR", "ELARA-SH")

FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20
TITLE_FONT_SIZE = 13
AXIS_LINE_WIDTH = 1.1
BAR_EDGE_LINE_WIDTH = 0.8
LINE_WIDTH = 2.2
SHADE_ALPHA = 0.18
BAR_VALUE_FONT_SIZE = 16

REWARD_COLOR = "#1f77b4"
LOSS_COLOR = "#d95f02"
LATENCY_COLOR = "#4c78a8"
ENERGY_COLOR = "#f58518"
ENERGY_SCALE = 1000.0
SCALED_ENERGY_AXIS_LABEL = "Mean execution energy (x1kJ)"


@dataclass
class RunData:
    seed: str
    rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact ELARA training, comparison, and ablation figures."
    )
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional space/comma separated training seeds to load.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving-average window for training reward and loss curves.",
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


def parse_list(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def seed_from_dir(path: Path) -> str:
    return path.name[5:] if path.name.startswith("seed_") else path.name


def discover_runs(train_root: Path, seeds: list[str]) -> list[RunData]:
    run_dirs = (
        [train_root / f"seed_{seed}" for seed in seeds]
        if seeds
        else sorted(path for path in train_root.glob("seed_*") if path.is_dir())
    )
    runs: list[RunData] = []
    for run_dir in run_dirs:
        rows = read_csv_rows(run_dir / "training_metrics.csv")
        if rows:
            runs.append(RunData(seed=seed_from_dir(run_dir), rows=rows))
    if not runs:
        raise SystemExit(f"No training_metrics.csv files found under {train_root}")
    return runs


def moving_average(values: list[float], window: int) -> list[float]:
    window = max(1, int(window))
    averaged: list[float] = []
    running: list[float] = []
    for value in values:
        running.append(value)
        if len(running) > window:
            running.pop(0)
        averaged.append(float(np.mean(running)))
    return averaged


def run_series(run: RunData, column: str, window: int) -> dict[int, float]:
    epochs: list[int] = []
    values: list[float] = []
    for row in run.rows:
        epoch_value = number(row, "epoch")
        metric_value = number(row, column)
        if epoch_value is None or metric_value is None:
            continue
        epochs.append(int(epoch_value))
        values.append(metric_value)
    smoothed = moving_average(values, window)
    return dict(zip(epochs, smoothed))


def aggregate_series(
    runs: list[RunData],
    column: str,
    window: int,
) -> tuple[list[int], list[float], list[float]]:
    per_run = [run_series(run, column, window) for run in runs]
    epochs = sorted(set().union(*(series.keys() for series in per_run)))
    kept_epochs: list[int] = []
    means: list[float] = []
    stds: list[float] = []
    for epoch in epochs:
        values = [series[epoch] for series in per_run if epoch in series]
        if not values:
            continue
        kept_epochs.append(epoch)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return kept_epochs, means, stds


def load_summary_rows(experiment_dir: Path) -> list[dict]:
    summary_rows = read_csv_rows(experiment_dir / "ablation_metric_summary.csv")
    if summary_rows:
        for row in summary_rows:
            row["ablation"] = canonical_ablation_name(row.get("ablation", ""))
        return summary_rows

    cycle_rows = read_csv_rows(experiment_dir / "all_ablation_cycle_metrics.csv")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"delay": [], "energy": []}
    )
    for row in cycle_rows:
        ablation = canonical_ablation_name(row.get("ablation", ""))
        delay = number(row, "average_end_to_end_delay_s")
        energy = number(row, "average_energy_j")
        if delay is not None:
            grouped[ablation]["delay"].append(delay)
        if energy is not None:
            grouped[ablation]["energy"].append(energy)

    rows: list[dict] = []
    for ablation, values in grouped.items():
        rows.append(
            {
                "ablation": ablation,
                LATENCY_COLUMN: str(float(np.mean(values["delay"]))),
                ENERGY_COLUMN: str(float(np.mean(values["energy"]))),
            }
        )
    return rows


def metrics_for_ablations(
    rows: list[dict],
    ablations: tuple[str, ...],
) -> tuple[list[str], list[float], list[float]]:
    by_name = {canonical_ablation_name(row.get("ablation", "")): row for row in rows}
    labels: list[str] = []
    delays: list[float] = []
    energies: list[float] = []
    missing: list[str] = []
    for ablation in ablations:
        row = by_name.get(ablation)
        delay = number(row or {}, LATENCY_COLUMN)
        energy = number(row or {}, ENERGY_COLUMN)
        if row is None or delay is None or energy is None:
            missing.append(ablation)
            continue
        labels.append(ablation)
        delays.append(delay)
        energies.append(energy)
    if missing:
        print(f"Warning: missing metric rows for: {', '.join(missing)}")
    if not labels:
        raise SystemExit("No usable latency/energy metrics found.")
    return labels, delays, energies


def set_axis_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def annotate_bars(ax, bars, fmt: str, color: str) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.018
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=BAR_VALUE_FONT_SIZE,
            color=color,
            rotation=0,
        )


def save_png_and_pdf(fig, output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_training_figure(
    output_path: Path,
    runs: list[RunData],
    window: int,
) -> tuple[Path, Path]:
    reward_epochs, reward_mean, reward_std = aggregate_series(
        runs, TRAIN_REWARD_COLUMN, window
    )
    loss_epochs, loss_mean, loss_std = aggregate_series(runs, TRAIN_LOSS_COLUMN, window)
    if not reward_epochs or not loss_epochs:
        raise SystemExit("Training reward/loss data is incomplete.")

    reward_mean_arr = np.asarray(reward_mean)
    reward_std_arr = np.asarray(reward_std)
    loss_mean_arr = np.asarray(loss_mean)
    loss_std_arr = np.asarray(loss_std)

    fig, ax_reward = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    ax_loss = ax_reward.twinx()

    reward_line = ax_reward.plot(
        reward_epochs,
        reward_mean,
        color=REWARD_COLOR,
        linewidth=LINE_WIDTH,
        label="Reward mean",
    )[0]
    ax_reward.fill_between(
        reward_epochs,
        reward_mean_arr - reward_std_arr,
        reward_mean_arr + reward_std_arr,
        color=REWARD_COLOR,
        alpha=SHADE_ALPHA,
        label="Reward mean +/- std",
    )

    loss_line = ax_loss.plot(
        loss_epochs,
        loss_mean,
        color=LOSS_COLOR,
        linewidth=LINE_WIDTH,
        linestyle="-",
        label="PPO loss mean",
    )[0]
    ax_loss.fill_between(
        loss_epochs,
        loss_mean_arr - loss_std_arr,
        loss_mean_arr + loss_std_arr,
        color=LOSS_COLOR,
        alpha=SHADE_ALPHA,
        label="PPO loss mean +/- std",
    )

    ax_reward.set_xlabel("Training epoch")
    ax_reward.set_ylabel("Average reward per request")
    ax_loss.set_ylabel("PPO training loss")
    ax_reward.tick_params(axis="y")
    ax_loss.tick_params(axis="y")
    ax_reward.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    set_axis_spines(ax_reward)
    set_axis_spines(ax_loss)

    handles = [
        reward_line,
        Patch(facecolor=REWARD_COLOR, alpha=SHADE_ALPHA, label="Reward mean +/- std"),
        loss_line,
        Patch(facecolor=LOSS_COLOR, alpha=SHADE_ALPHA, label="PPO loss mean +/- std"),
    ]
    ax_reward.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
        ncol=1,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.6,
        borderaxespad=0.2,
    )
    output_paths = save_png_and_pdf(fig, output_path)
    plt.close(fig)
    return output_paths


def plot_dual_axis_bars(
    output_path: Path,
    labels: list[str],
    delays: list[float],
    energies: list[float],
) -> tuple[Path, Path]:
    x = np.arange(len(labels))
    width = 0.34
    scaled_energies = [energy / ENERGY_SCALE for energy in energies]
    fig, ax_delay = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax_energy = ax_delay.twinx()

    delay_bars = ax_delay.bar(
        x - width / 2,
        delays,
        width=width,
        color=LATENCY_COLOR,
        edgecolor="#1f2937",
        linewidth=BAR_EDGE_LINE_WIDTH,
        label="Mean execution latency",
        zorder=3,
    )
    energy_bars = ax_energy.bar(
        x + width / 2,
        scaled_energies,
        width=width,
        color=ENERGY_COLOR,
        edgecolor="#1f2937",
        linewidth=BAR_EDGE_LINE_WIDTH,
        label="Mean execution energy",
        zorder=3,
    )

    ax_delay.set_xticks(x)
    display_labels = [ABLATION_LABELS.get(label, label) for label in labels]
    if any(label.startswith("ELARA-") for label in labels):
        display_labels = [label.replace(" ", "\n") for label in display_labels]
    ax_delay.set_xticklabels(display_labels)
    ax_delay.set_ylabel("Mean end-to-end latency (s)")
    ax_energy.set_ylabel(SCALED_ENERGY_AXIS_LABEL)
    ax_delay.tick_params(axis="y")
    ax_energy.tick_params(axis="y")
    ax_delay.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax_delay.set_ylim(0, max(delays) * 1.18)
    ax_energy.set_ylim(0, max(scaled_energies) * 1.18)

    annotate_bars(ax_delay, delay_bars, "{:.2f}", LATENCY_COLOR)
    annotate_bars(ax_energy, energy_bars, "{:.2f}", ENERGY_COLOR)
    set_axis_spines(ax_delay)
    set_axis_spines(ax_energy)

    handles = [delay_bars[0], energy_bars[0]]
    labels_for_legend = ["Mean end-to-end latency", "Mean execution energy"]
    ax_delay.legend(
        handles,
        labels_for_legend,
        loc="upper left",
        # bbox_to_anchor=(0.5, 1.01),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )

    output_paths = save_png_and_pdf(fig, output_path)
    plt.close(fig)
    return output_paths


def write_metric_snapshot(
    output_dir: Path,
    comparison: tuple[list[str], list[float], list[float]],
    ablation: tuple[list[str], list[float], list[float]],
) -> None:
    rows = []
    for group_name, group_data in (
        ("comparison", comparison),
        ("ablation", ablation),
    ):
        labels, delays, energies = group_data
        for label, delay, energy in zip(labels, delays, energies):
            rows.append(
                {
                    "figure_group": group_name,
                    "ablation": label,
                    "mean_end_to_end_latency_s": f"{delay:.10g}",
                    "mean_end_to_end_energy_j": f"{energy:.10g}",
                }
            )
    path = output_dir / "elara_paper_metric_snapshot.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure_group",
                "ablation",
                "mean_end_to_end_latency_s",
                "mean_end_to_end_energy_j",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.train_root, parse_list(args.seeds))
    summary_rows = load_summary_rows(args.experiment_dir)
    comparison_metrics = metrics_for_ablations(summary_rows, COMPARISON_ABLATIONS)
    ablation_metrics = metrics_for_ablations(summary_rows, ABLATION_ABLATIONS)

    train_path = args.output_dir / "elara_train.png"
    comparison_path = args.output_dir / "elara_comparison.png"
    ablation_path = args.output_dir / "elara_ablation.png"

    generated_paths = []
    generated_paths.extend(plot_training_figure(train_path, runs, args.window))
    generated_paths.extend(plot_dual_axis_bars(comparison_path, *comparison_metrics))
    generated_paths.extend(plot_dual_axis_bars(ablation_path, *ablation_metrics))
    write_metric_snapshot(args.output_dir, comparison_metrics, ablation_metrics)

    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(
        "\n".join(
            str(path.resolve())
            for path in generated_paths
        )
        + "\n",
        encoding="utf-8",
    )
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
