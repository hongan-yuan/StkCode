from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunCurves:
    label: str
    source: Path
    episodes: np.ndarray
    rewards: np.ndarray
    updates: np.ndarray
    total_loss: np.ndarray
    policy_loss: np.ndarray
    value_loss: np.ndarray
    entropy: np.ndarray


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Plot ELARA PPO reward and loss curves from training_metrics.csv."
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "training_metrics.csv, one training task directory, or a parallel "
            "training root containing multiple task directories"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reward-window", type=int, default=25)
    parser.add_argument("--loss-window", type=int, default=1)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="draw each individual seed behind the aggregate curve",
    )
    args = parser.parse_args(argv)
    if args.reward_window < 1 or args.loss_window < 1:
        parser.error("smoothing windows must be at least 1")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    args.formats = tuple(
        dict.fromkeys(item.strip().lower() for item in args.formats.split(",") if item.strip())
    )
    unsupported = set(args.formats) - {"png", "pdf", "svg"}
    if unsupported or not args.formats:
        parser.error("--formats must contain one or more of: png,pdf,svg")
    return args


def discover_metrics(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "training_metrics.csv":
            raise ValueError(f"expected training_metrics.csv, got {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"input does not exist: {path}")
    direct = path / "training_metrics.csv"
    if direct.is_file():
        return [direct]
    metrics = sorted(path.rglob("training_metrics.csv"))
    if not metrics:
        raise ValueError(f"no training_metrics.csv found below {path}")
    return metrics


def _finite(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _run_metadata(metrics_path: Path) -> tuple[str, float, float]:
    config_path = metrics_path.parent / "config.json"
    config = {}
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    seed = config.get("seed")
    if seed is None:
        match = re.search(r"seed[-_](\d+)", str(metrics_path.parent), re.IGNORECASE)
        seed = match.group(1) if match else metrics_path.parent.name
    label = f"seed {seed}"
    return (
        label,
        float(config.get("ppo_value_coef", 0.5)),
        float(config.get("ppo_entropy_coef", 0.01)),
    )


def load_run(metrics_path: Path) -> RunCurves:
    label, value_coef, entropy_coef = _run_metadata(metrics_path)
    episodes: list[float] = []
    rewards: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"episode", "return", "policy_loss", "value_loss", "entropy"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            episode = _finite(row.get("episode"))
            reward = _finite(row.get("return"))
            if episode is not None and reward is not None:
                episodes.append(episode)
                rewards.append(reward)
            policy = _finite(row.get("policy_loss"))
            value = _finite(row.get("value_loss"))
            entropy = _finite(row.get("entropy"))
            if policy is not None and value is not None and entropy is not None:
                policy_losses.append(policy)
                value_losses.append(value)
                entropies.append(entropy)
    if not rewards:
        raise ValueError(f"{metrics_path} contains no finite reward values")
    policy_array = np.asarray(policy_losses, dtype=float)
    value_array = np.asarray(value_losses, dtype=float)
    entropy_array = np.asarray(entropies, dtype=float)
    total_loss = policy_array + value_coef * value_array - entropy_coef * entropy_array
    return RunCurves(
        label=label,
        source=metrics_path,
        episodes=np.asarray(episodes, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        updates=np.arange(1, len(policy_array) + 1, dtype=float),
        total_loss=total_loss,
        policy_loss=policy_array,
        value_loss=value_array,
        entropy=entropy_array,
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.astype(float, copy=True)
    window = min(int(window), len(values))
    cumulative = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        count = index + 1 - start
        result[index] = (cumulative[index + 1] - cumulative[start]) / count
    return result


def aggregate_series(
    runs: list[RunCurves], attribute: str, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = [
        moving_average(getattr(run, attribute), window)
        for run in runs
        if len(getattr(run, attribute)) > 0
    ]
    if not series:
        return tuple(np.asarray([], dtype=float) for _ in range(4))
    common_length = min(len(values) for values in series)
    matrix = np.full((len(series), common_length), np.nan, dtype=float)
    for index, values in enumerate(series):
        matrix[index, :] = values[:common_length]
    counts = np.sum(np.isfinite(matrix), axis=0)
    means = np.nanmean(matrix, axis=0)
    std = np.zeros(common_length, dtype=float)
    for index in range(common_length):
        finite = matrix[:, index][np.isfinite(matrix[:, index])]
        std[index] = np.std(finite, ddof=1) if len(finite) > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(np.maximum(counts, 1))
    x = np.arange(common_length, dtype=float)
    return x, means, means - ci95, means + ci95


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def _plot_series(
    ax,
    runs: list[RunCurves],
    attribute: str,
    window: int,
    xlabel: str,
    ylabel: str,
    title: str,
    show_runs: bool,
    color: str,
) -> None:
    if show_runs:
        for run in runs:
            values = moving_average(getattr(run, attribute), window)
            x = run.episodes[: len(values)] if attribute == "rewards" else run.updates[: len(values)]
            ax.plot(x, values, linewidth=0.75, alpha=0.22, color=color)
    x, mean, lower, upper = aggregate_series(runs, attribute, window)
    if attribute != "rewards":
        x = x + 1
    ax.plot(x, mean, linewidth=1.8, color=color, label="Mean")
    if len(runs) > 1:
        ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0, label="95% CI")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.25)
    if len(runs) > 1:
        ax.legend(frameon=False, loc="best")


def _save(fig, output_dir: Path, stem: str, formats: tuple[str, ...], dpi: int) -> list[Path]:
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_curves(
    runs: list[RunCurves],
    output_dir: Path,
    reward_window: int,
    loss_window: int,
    formats: tuple[str, ...],
    dpi: int,
    show_runs: bool,
) -> list[Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    _plot_series(
        ax, runs, "rewards", reward_window, "Episode", "Episode return",
        f"PPO reward (moving average window = {reward_window})", show_runs, "#1f77b4",
    )
    generated.extend(_save(fig, output_dir, "ppo_reward_curve", formats, dpi))

    loss_panels = (
        ("total_loss", "Total PPO loss", "#d62728"),
        ("policy_loss", "Policy loss", "#ff7f0e"),
        ("value_loss", "Value loss", "#2ca02c"),
        ("entropy", "Policy entropy", "#9467bd"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for ax, (attribute, title, color) in zip(axes.flat, loss_panels):
        _plot_series(
            ax, runs, attribute, loss_window, "PPO update", title,
            title, show_runs, color,
        )
    generated.extend(_save(fig, output_dir, "ppo_loss_curves", formats, dpi))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    _plot_series(
        axes[0], runs, "rewards", reward_window, "Episode", "Episode return",
        "Reward", show_runs, "#1f77b4",
    )
    _plot_series(
        axes[1], runs, "total_loss", loss_window, "PPO update", "Total PPO loss",
        "Loss", show_runs, "#d62728",
    )
    generated.extend(_save(fig, output_dir, "ppo_reward_loss_curves", formats, dpi))
    return generated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metrics_paths = discover_metrics(args.input)
        runs = [load_run(path) for path in metrics_paths]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not any(len(run.total_loss) for run in runs):
        raise SystemExit(
            "No PPO loss records were found. The selected training run did not "
            "reach a PPO update, or its metrics file is incomplete."
        )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (args.input.parent if args.input.is_file() else args.input) / "plots"
    output_dir = output_dir.expanduser().resolve()
    generated = plot_curves(
        runs,
        output_dir,
        args.reward_window,
        args.loss_window,
        args.formats,
        args.dpi,
        args.show_runs,
    )
    summary = {
        "run_count": len(runs),
        "runs": [
            {
                "label": run.label,
                "source": str(run.source),
                "episode_count": len(run.rewards),
                "ppo_update_count": len(run.total_loss),
            }
            for run in runs
        ],
        "reward_window": args.reward_window,
        "loss_window": args.loss_window,
        "generated": [str(path) for path in generated],
    }
    (output_dir / "ppo_curve_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Loaded {len(runs)} training run(s).")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
