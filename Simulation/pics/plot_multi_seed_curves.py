from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on local plotting env
    plt = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT_DIR / "Simulation" / "multi_seed_runs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "multi_seed"


@dataclass
class RunData:
    seed: str
    run_dir: Path
    metrics: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot single-seed or multi-seed aggregate PPO/Bandit training curves."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing seed_<seed>/training_metrics.csv subdirectories.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Explicit run directory. Can be provided multiple times.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional space/comma separated seed ids to load from --input-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where figures will be written. PNG is used when "
            "matplotlib is installed; otherwise SVG fallback files are written."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("aggregate", "single", "both"),
        default="both",
        help="Plot aggregate curves, single-seed curves, or both.",
    )
    parser.add_argument(
        "--single-seed",
        default="",
        help="When --mode includes single, plot only this seed id.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window for reward, loss, success, and bandit curves.",
    )
    parser.add_argument(
        "--reward-column",
        default="average_reward_per_request",
        help=(
            "Reward column to plot. Use "
            "average_chain_length_normalized_reward_per_request to inspect the "
            "chain-length-normalized curve."
        ),
    )
    return parser.parse_args()


def parse_seed_list(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def seed_from_run_dir(path: Path) -> str:
    name = path.name
    return name[5:] if name.startswith("seed_") else name


def discover_run_dirs(input_root: Path, seeds: list[str]) -> list[Path]:
    if seeds:
        return [input_root / f"seed_{seed}" for seed in seeds]
    return sorted(
        path for path in input_root.glob("seed_*")
        if (path / "training_metrics.csv").exists()
    )


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_runs(args: argparse.Namespace) -> list[RunData]:
    run_dirs = args.run_dir or discover_run_dirs(
        args.input_root, parse_seed_list(args.seeds)
    )
    runs = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "training_metrics.csv"
        metrics = read_csv_rows(metrics_path)
        if not metrics:
            print(f"Skipping {run_dir}: missing or empty training_metrics.csv")
            continue
        runs.append(RunData(seed_from_run_dir(run_dir), run_dir, metrics))
    if not runs:
        raise SystemExit("No usable training runs found.")
    return runs


def numeric_value(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_epoch(row: dict) -> int | None:
    value = numeric_value(row, "epoch")
    return int(value) if value is not None else None


def moving_average(values: list[float | None], window: int) -> list[float | None]:
    window = max(1, int(window))
    averaged: list[float | None] = []
    finite_window: list[float] = []
    for value in values:
        if value is not None:
            finite_window.append(value)
        if len(finite_window) > window:
            finite_window.pop(0)
        averaged.append(
            sum(finite_window) / len(finite_window) if finite_window else None
        )
    return averaged


def series_by_epoch(
    run: RunData,
    column: str,
    window: int,
) -> dict[int, float]:
    epochs: list[int] = []
    raw_values: list[float | None] = []
    for row in run.metrics:
        epoch = row_epoch(row)
        if epoch is None:
            continue
        epochs.append(epoch)
        raw_values.append(numeric_value(row, column))
    averaged = moving_average(raw_values, window)
    return {
        epoch: value
        for epoch, value in zip(epochs, averaged)
        if value is not None
    }


def aggregate_series(
    runs: list[RunData],
    column: str,
    window: int,
) -> tuple[list[int], list[float], list[float]]:
    per_run = [series_by_epoch(run, column, window) for run in runs]
    epochs = sorted(set().union(*(series.keys() for series in per_run)))
    means: list[float] = []
    stds: list[float] = []
    kept_epochs: list[int] = []
    for epoch in epochs:
        values = [series[epoch] for series in per_run if epoch in series]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if len(values) > 1 else 0.0
        )
        kept_epochs.append(epoch)
        means.append(mean)
        stds.append(math.sqrt(variance))
    return kept_epochs, means, stds


def single_series(
    run: RunData,
    column: str,
    window: int,
) -> tuple[list[int], list[float]]:
    series = series_by_epoch(run, column, window)
    epochs = sorted(series)
    return epochs, [series[epoch] for epoch in epochs]


def save_line_plot(
    path: Path,
    title: str,
    ylabel: str,
    aggregate: tuple[list[int], list[float], list[float]] | None = None,
    singles: list[tuple[str, list[int], list[float]]] | None = None,
) -> None:
    if plt is None:
        save_line_plot_svg(path.with_suffix(".svg"), title, ylabel, aggregate, singles)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    if aggregate is not None:
        epochs, means, stds = aggregate
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        ax.plot(epochs, means, color="#2563eb", linewidth=2.2, label="Mean")
        ax.fill_between(
            epochs, lower, upper,
            color="#93c5fd", alpha=0.35, label="Mean +/- std"
        )
    for label, epochs, values in singles or []:
        ax.plot(epochs, values, linewidth=1.6, alpha=0.85, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def scale_points(
    xs: list[float],
    ys: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    margin: int,
) -> list[tuple[float, float]]:
    x_span = max(1.0e-9, x_max - x_min)
    y_span = max(1.0e-9, y_max - y_min)
    return [
        (
            margin + (x - x_min) / x_span * (width - 2 * margin),
            height - margin - (y - y_min) / y_span * (height - 2 * margin),
        )
        for x, y in zip(xs, ys)
    ]


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def save_line_plot_svg(
    path: Path,
    title: str,
    ylabel: str,
    aggregate: tuple[list[int], list[float], list[float]] | None = None,
    singles: list[tuple[str, list[int], list[float]]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 1100, 560, 70
    all_x: list[float] = []
    all_y: list[float] = []
    if aggregate is not None:
        epochs, means, stds = aggregate
        all_x.extend(epochs)
        all_y.extend(mean - std for mean, std in zip(means, stds))
        all_y.extend(mean + std for mean, std in zip(means, stds))
    for _, epochs, values in singles or []:
        all_x.extend(epochs)
        all_y.extend(values)
    if not all_x or not all_y:
        all_x, all_y = [0, 1], [0.0, 1.0]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_padding = max(1.0e-9, (y_max - y_min) * 0.08)
    y_min -= y_padding
    y_max += y_padding

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        f'font-family="Arial" font-size="24">{escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
        f'font-family="Arial" font-size="15">Epoch</text>',
        f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 22 {height / 2:.1f})" '
        f'font-family="Arial" font-size="15">{escape(ylabel)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        f'stroke="#334155" stroke-width="1"/>',
    ]
    for i in range(6):
        x = margin + i * (width - 2 * margin) / 5
        y = margin + i * (height - 2 * margin) / 5
        x_label = x_min + i * (x_max - x_min) / 5
        y_label = y_max - i * (y_max - y_min) / 5
        body.append(
            f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" '
            f'y2="{height - margin}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        body.append(
            f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" '
            f'y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{x:.2f}" y="{height - margin + 22}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{x_label:.0f}</text>'
        )
        body.append(
            f'<text x="{margin - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{y_label:.2f}</text>'
        )
    if aggregate is not None:
        epochs, means, stds = aggregate
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        lower_points = scale_points(
            [float(x) for x in epochs], lower, x_min, x_max, y_min, y_max,
            width, height, margin
        )
        upper_points = scale_points(
            [float(x) for x in epochs], upper, x_min, x_max, y_min, y_max,
            width, height, margin
        )
        band = upper_points + list(reversed(lower_points))
        mean_points = scale_points(
            [float(x) for x in epochs], means, x_min, x_max, y_min, y_max,
            width, height, margin
        )
        body.append(
            f'<polygon points="{polyline(band)}" fill="#93c5fd" opacity="0.35"/>'
        )
        body.append(
            f'<polyline points="{polyline(mean_points)}" fill="none" '
            f'stroke="#2563eb" stroke-width="3"/>'
        )
    colors = ["#dc2626", "#059669", "#7c3aed", "#ea580c"]
    for index, (label, epochs, values) in enumerate(singles or []):
        points = scale_points(
            [float(x) for x in epochs], values, x_min, x_max, y_min, y_max,
            width, height, margin
        )
        body.append(
            f'<polyline points="{polyline(points)}" fill="none" '
            f'stroke="{colors[index % len(colors)]}" stroke-width="2"/>'
        )
        body.append(
            f'<text x="{width - margin}" y="{margin + 18 * index}" '
            f'text-anchor="end" font-family="Arial" font-size="13" '
            f'fill="{colors[index % len(colors)]}">{escape(label)}</text>'
        )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def read_action_counts(run: RunData) -> Counter:
    actions_path = run.run_dir / "bandit_actions.csv"
    counts: Counter[str] = Counter()
    if actions_path.exists():
        for row in read_csv_rows(actions_path):
            action = row.get("action", "")
            if action:
                counts[action] += 1
        if counts:
            return counts

    if not run.metrics:
        return counts
    last_row = run.metrics[-1]
    for action in ("add", "move", "remove"):
        value = numeric_value(last_row, f"bandit_total_applied_{action}_count")
        if value is not None:
            counts[action] = int(value)
    return counts


def aggregate_action_counts(runs: list[RunData]) -> tuple[list[str], list[float], list[float]]:
    per_run = [read_action_counts(run) for run in runs]
    actions = sorted(set().union(*(counts.keys() for counts in per_run)))
    means: list[float] = []
    stds: list[float] = []
    for action in actions:
        values = [float(counts.get(action, 0)) for counts in per_run]
        mean = sum(values) / len(values) if values else 0.0
        variance = (
            sum((value - mean) ** 2 for value in values) / len(values)
            if len(values) > 1 else 0.0
        )
        means.append(mean)
        stds.append(math.sqrt(variance))
    return actions, means, stds


def save_action_distribution(path: Path, runs: list[RunData], title: str) -> None:
    if plt is None:
        save_action_distribution_svg(path.with_suffix(".svg"), runs, title)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    actions, means, stds = aggregate_action_counts(runs)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.bar(actions, means, yerr=stds, color="#10b981", alpha=0.82, capsize=5)
    ax.set_title(title)
    ax.set_xlabel("Bandit action")
    ax.set_ylabel("Applied action count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_action_distribution_svg(path: Path, runs: list[RunData], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions, means, stds = aggregate_action_counts(runs)
    width, height, margin = 820, 500, 70
    max_y = max([mean + std for mean, std in zip(means, stds)] + [1.0])
    bar_area = width - 2 * margin
    bar_width = bar_area / max(1, len(actions)) * 0.55
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        f'font-family="Arial" font-size="24">{escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        f'stroke="#334155" stroke-width="1"/>',
    ]
    for index, action in enumerate(actions):
        center = margin + (index + 0.5) * bar_area / max(1, len(actions))
        mean = means[index]
        std = stds[index]
        bar_height = mean / max_y * (height - 2 * margin)
        x = center - bar_width / 2
        y = height - margin - bar_height
        err_top = height - margin - (mean + std) / max_y * (height - 2 * margin)
        err_bottom = height - margin - max(0.0, mean - std) / max_y * (height - 2 * margin)
        body.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="#10b981" opacity="0.82"/>'
        )
        body.append(
            f'<line x1="{center:.2f}" y1="{err_top:.2f}" x2="{center:.2f}" '
            f'y2="{err_bottom:.2f}" stroke="#064e3b" stroke-width="2"/>'
        )
        body.append(
            f'<text x="{center:.2f}" y="{height - margin + 22}" '
            f'text-anchor="middle" font-family="Arial" font-size="13">'
            f'{escape(action)}</text>'
        )
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
        f'font-family="Arial" font-size="15">Bandit action</text>'
    )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def plot_aggregate(runs: list[RunData], args: argparse.Namespace) -> None:
    specs = [
        (
            args.reward_column,
            "ppo_average_reward_per_request.png",
            "PPO-Agent Average Reward Per Request",
            "Reward / request",
        ),
        (
            "ppo_loss",
            "ppo_training_loss.png",
            "PPO-Agent Training Loss",
            "Loss",
        ),
        (
            "success_rate",
            "slot_success_rate.png",
            "Per-Slot Request Success Rate",
            "Success rate",
        ),
        (
            "bandit_average_arm_reward",
            "bandit_learning_curve.png",
            "Bandit Learning Curve",
            "Average arm reward",
        ),
    ]
    for column, filename, title, ylabel in specs:
        save_line_plot(
            args.output_dir / filename,
            title,
            ylabel,
            aggregate=aggregate_series(runs, column, args.window),
        )
    save_action_distribution(
        args.output_dir / "bandit_action_distribution.png",
        runs,
        "Bandit Action Distribution",
    )


def plot_single(runs: list[RunData], args: argparse.Namespace) -> None:
    selected = [
        run for run in runs
        if not args.single_seed or run.seed == args.single_seed
    ]
    if not selected:
        raise SystemExit(f"No run matched --single-seed={args.single_seed}")

    specs = [
        (
            args.reward_column,
            "ppo_average_reward_per_request",
            "PPO-Agent Average Reward Per Request",
            "Reward / request",
        ),
        ("ppo_loss", "ppo_training_loss", "PPO-Agent Training Loss", "Loss"),
        ("success_rate", "slot_success_rate", "Per-Slot Request Success Rate", "Success rate"),
        (
            "bandit_average_arm_reward",
            "bandit_learning_curve",
            "Bandit Learning Curve",
            "Average arm reward",
        ),
    ]
    for run in selected:
        seed_dir = args.output_dir / f"seed_{run.seed}"
        for column, stem, title, ylabel in specs:
            epochs, values = single_series(run, column, args.window)
            save_line_plot(
                seed_dir / f"{stem}.png",
                f"{title} (seed {run.seed})",
                ylabel,
                singles=[(f"seed {run.seed}", epochs, values)],
            )
        save_action_distribution(
            seed_dir / "bandit_action_distribution.png",
            [run],
            f"Bandit Action Distribution (seed {run.seed})",
        )


def main() -> None:
    args = parse_args()
    args.window = max(1, int(args.window))
    runs = load_runs(args)
    if args.mode in ("aggregate", "both"):
        plot_aggregate(runs, args)
    if args.mode in ("single", "both"):
        plot_single(runs, args)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
