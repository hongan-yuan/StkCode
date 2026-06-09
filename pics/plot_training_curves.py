from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "outputs"


def default_input_dir() -> Path:
    candidates = [
        ROOT_DIR / "Simulation" / "dyn_train_data_260605",
    ]
    for candidate in candidates:
        if (candidate / "training_metrics.csv").exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PPO reward and Bandit learning curves from training CSV logs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir(),
        help="Directory containing training_metrics.csv and Bandit CSV logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG figures will be saved.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving-average window for smoothed training curves.",
    )
    parser.add_argument(
        "--extra-diagnostics",
        action="store_true",
        help="Also draw request/failure auxiliary diagnostics beyond the core five figures.",
    )
    return parser.parse_args()


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        print(
            "matplotlib is unavailable; falling back to dependency-free SVG output. "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return None


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, window)
    smoothed = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1.0e-12:
        return low - 0.5, high + 0.5
    return low, high


def svg_polyline(
    xs: list[int],
    ys: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    margin: int,
) -> str:
    points = []
    for x_value, y_value in zip(xs, ys):
        x_norm = (x_value - x_min) / max(1.0e-12, x_max - x_min)
        y_norm = (y_value - y_min) / max(1.0e-12, y_max - y_min)
        x_px = margin + x_norm * (width - 2 * margin)
        y_px = height - margin - y_norm * (height - 2 * margin)
        points.append(f"{x_px:.1f},{y_px:.1f}")
    return " ".join(points)


def save_svg_line_chart(
    path: Path,
    title: str,
    epochs: list[int],
    series: list[tuple[str, list[float], str]],
    y_label: str,
) -> Path:
    width, height, margin = 980, 520, 64
    x_min, x_max = min_max([float(epoch) for epoch in epochs])
    all_values = [value for _, values, _ in series for value in values]
    y_min, y_max = min_max(all_values)
    lines = []
    legend = []
    for idx, (label, values, color) in enumerate(series):
        points = svg_polyline(epochs, values, x_min, x_max, y_min, y_max, width, height, margin)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />'
        )
        legend_y = 58 + idx * 22
        legend.append(
            f'<line x1="760" y1="{legend_y}" x2="800" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3" />'
            f'<text x="812" y="{legend_y + 5}" font-size="14">{label}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="24" font-family="Arial" font-weight="700">{title}</text>
<text x="{margin}" y="{height - 18}" font-size="14" font-family="Arial">Epoch</text>
<text x="18" y="{margin}" font-size="14" font-family="Arial" transform="rotate(-90 18,{margin})">{y_label}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111827" stroke-width="1.5"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111827" stroke-width="1.5"/>
<text x="{margin}" y="{height - margin + 22}" font-size="12" font-family="Arial">{x_min:.0f}</text>
<text x="{width - margin - 28}" y="{height - margin + 22}" font-size="12" font-family="Arial">{x_max:.0f}</text>
<text x="22" y="{height - margin + 4}" font-size="12" font-family="Arial">{y_min:.3g}</text>
<text x="22" y="{margin + 4}" font-size="12" font-family="Arial">{y_max:.3g}</text>
{''.join(lines)}
{''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")
    return path


def save_svg_bar_chart(path: Path, title: str, counts: Counter) -> Path | None:
    if not counts:
        return None
    width, height, margin = 820, 480, 64
    labels = list(counts)
    values = [counts[label] for label in labels]
    _, y_max = min_max([float(value) for value in values])
    bar_width = (width - 2 * margin) / max(1, len(labels)) * 0.64
    chunks = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        group_x = margin + idx * (width - 2 * margin) / max(1, len(labels))
        bar_height = value / y_max * (height - 2 * margin)
        x = group_x + bar_width * 0.28
        y = height - margin - bar_height
        chunks.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#2563eb"/>'
            f'<text x="{x:.1f}" y="{height - margin + 22}" font-size="13" '
            f'font-family="Arial">{label}</text>'
            f'<text x="{x:.1f}" y="{y - 8:.1f}" font-size="13" '
            f'font-family="Arial">{value}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="24" font-family="Arial" font-weight="700">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111827" stroke-width="1.5"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111827" stroke-width="1.5"/>
{''.join(chunks)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")
    return path


def save_svg_slot_load_diagnostics(rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    arrivals = [to_float(row.get("arrival_count")) for row in rows]
    total_rewards = [to_float(row.get("total_reward")) for row in rows]
    rewards_per_request = [to_float(row.get("average_reward_per_request")) for row in rows]

    path = output_dir / "slot_load_diagnostics.svg"
    width, panel_height, margin = 980, 300, 64
    height = panel_height * 3 + 76
    panels = [
        (
            "Requests",
            [
                ("Arrival count", arrivals, "#0f766e"),
                (f"Arrival MA ({window})", moving_average(arrivals, window), "#14b8a6"),
            ],
        ),
        (
            "Total reward",
            [
                ("Raw total reward", total_rewards, "#94a3b8"),
                (f"Total reward MA ({window})", moving_average(total_rewards, window), "#475569"),
            ],
        ),
        (
            "Reward / request",
            [
                ("Raw reward / request", rewards_per_request, "#60a5fa"),
                (
                    f"Reward / request MA ({window})",
                    moving_average(rewards_per_request, window),
                    "#2563eb",
                ),
            ],
        ),
    ]
    x_min, x_max = min_max([float(epoch) for epoch in epochs])
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">Slot Load Diagnostics</text>',
    ]
    plot_width = width - 2 * margin
    plot_height = panel_height - 52
    for panel_index, (label, series) in enumerate(panels):
        y_offset = 54 + panel_index * panel_height
        values = [value for _, value_list, _ in series for value in value_list]
        y_min, y_max = min_max(values)
        chunks.append(
            f'<line x1="{margin}" y1="{y_offset + plot_height}" x2="{width - margin}" y2="{y_offset + plot_height}" stroke="#111827" stroke-width="1.2"/>'
            f'<line x1="{margin}" y1="{y_offset}" x2="{margin}" y2="{y_offset + plot_height}" stroke="#111827" stroke-width="1.2"/>'
            f'<text x="18" y="{y_offset + plot_height / 2:.1f}" font-size="14" font-family="Arial" transform="rotate(-90 18,{y_offset + plot_height / 2:.1f})">{label}</text>'
        )
        for tick in range(5):
            y_value = y_min + (y_max - y_min) * tick / 4
            y_norm = (y_value - y_min) / max(1.0e-12, y_max - y_min)
            y_px = y_offset + plot_height - y_norm * plot_height
            chunks.append(
                f'<line x1="{margin}" y1="{y_px:.1f}" x2="{width - margin}" y2="{y_px:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
        for series_label, values, color in series:
            points = []
            for epoch, value in zip(epochs, values):
                x_norm = (epoch - x_min) / max(1.0e-12, x_max - x_min)
                y_norm = (value - y_min) / max(1.0e-12, y_max - y_min)
                points.append(
                    f"{margin + x_norm * plot_width:.1f},{y_offset + plot_height - y_norm * plot_height:.1f}"
                )
            width_px = "3" if "MA" in series_label else "1.3"
            opacity = "1.0" if "MA" in series_label else "0.55"
            chunks.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{width_px}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        if panel_index == 0:
            for idx, (series_label, _, color) in enumerate(series):
                legend_y = y_offset + 16 + idx * 22
                chunks.append(
                    f'<line x1="{width - 270}" y1="{legend_y}" x2="{width - 230}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
                    f'<text x="{width - 220}" y="{legend_y + 5}" font-size="14" font-family="Arial">{series_label}</text>'
                )
    chunks.append(f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="16" font-family="Arial">Epoch</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def save_svg_ppo_training_diagnostics(rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    success_rates = [to_float(row.get("success_rate")) for row in rows]
    losses = [to_float(row.get("ppo_loss")) for row in rows]

    path = output_dir / "ppo_training_diagnostics.svg"
    width, panel_height, margin = 980, 300, 64
    height = panel_height * 2 + 76
    panels = [
        (
            "Success rate",
            [("Success rate", success_rates, "#059669")],
            -0.05,
            1.05,
        ),
        (
            "Loss",
            [
                ("Raw PPO loss", losses, "#fecaca"),
                (f"PPO loss MA ({window})", moving_average(losses, window), "#dc2626"),
            ],
            None,
            None,
        ),
    ]
    x_min, x_max = min_max([float(epoch) for epoch in epochs])
    plot_width = width - 2 * margin
    plot_height = panel_height - 52
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">PPO-Agent Training Diagnostics</text>',
    ]
    for panel_index, (label, series, fixed_min, fixed_max) in enumerate(panels):
        y_offset = 54 + panel_index * panel_height
        values = [value for _, value_list, _ in series for value in value_list]
        y_min, y_max = (fixed_min, fixed_max) if fixed_min is not None else min_max(values)
        chunks.append(
            f'<line x1="{margin}" y1="{y_offset + plot_height}" x2="{width - margin}" y2="{y_offset + plot_height}" stroke="#111827" stroke-width="1.2"/>'
            f'<line x1="{margin}" y1="{y_offset}" x2="{margin}" y2="{y_offset + plot_height}" stroke="#111827" stroke-width="1.2"/>'
            f'<text x="18" y="{y_offset + plot_height / 2:.1f}" font-size="14" font-family="Arial" transform="rotate(-90 18,{y_offset + plot_height / 2:.1f})">{label}</text>'
        )
        for series_label, values, color in series:
            points = []
            for epoch, value in zip(epochs, values):
                x_norm = (epoch - x_min) / max(1.0e-12, x_max - x_min)
                y_norm = (value - y_min) / max(1.0e-12, y_max - y_min)
                points.append(
                    f"{margin + x_norm * plot_width:.1f},{y_offset + plot_height - y_norm * plot_height:.1f}"
                )
            width_px = "3" if "MA" in series_label else "1.3"
            opacity = "1.0" if "MA" in series_label or len(series) == 1 else "0.55"
            chunks.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{width_px}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for idx, (series_label, _, color) in enumerate(series):
            legend_y = y_offset + 20 + idx * 22
            chunks.append(
                f'<line x1="{width - 260}" y1="{legend_y}" x2="{width - 220}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
                f'<text x="{width - 210}" y="{legend_y + 5}" font-size="14" font-family="Arial">{series_label}</text>'
            )
    chunks.append(f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="16" font-family="Arial">Epoch</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def save_svg_fallback(rows: list[dict], action_rows: list[dict], output_dir: Path, window: int) -> list[Path]:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    average_rewards_per_request = [
        to_float(row.get("average_reward_per_request")) for row in rows
    ]
    avg_arm_reward = [to_float(row.get("bandit_average_arm_reward")) for row in rows]
    positive_arms = [to_float(row.get("bandit_positive_arm_count")) for row in rows]
    known_arms = [to_float(row.get("bandit_known_arm_count")) for row in rows]

    paths = [
        save_svg_slot_load_diagnostics(rows, output_dir, window),
        save_svg_ppo_training_diagnostics(rows, output_dir, window),
        save_svg_line_chart(
            output_dir / "ppo_average_reward_per_request.svg",
            "PPO-Agent Average Reward Per Request",
            epochs,
            [
                ("Raw reward / request", average_rewards_per_request, "#9ca3af"),
                (
                    f"Moving average ({window})",
                    moving_average(average_rewards_per_request, window),
                    "#2563eb",
                ),
            ],
            "Reward / request",
        ),
        save_svg_line_chart(
            output_dir / "bandit_learning_curve.svg",
            "Bandit Strategy Learning Quality",
            epochs,
            [
                ("Mean arm reward", moving_average(avg_arm_reward, window), "#7c3aed"),
                ("Positive arms", positive_arms, "#f97316"),
                ("Known arms", known_arms, "#0f766e"),
            ],
            "Metric value",
        ),
    ]

    counts = Counter(row.get("action", "unknown") or "unknown" for row in action_rows)
    bar_path = save_svg_bar_chart(
        output_dir / "bandit_action_distribution.svg",
        "Bandit Migration Action Distribution",
        counts,
    )
    if bar_path is not None:
        paths.append(bar_path)
    return paths


def save_svg_request_failure_charts(request_rows: list[dict], output_dir: Path) -> list[Path]:
    failed_rows = [
        row for row in request_rows if str(row.get("feasible", "")).lower() == "false"
    ]
    if not failed_rows:
        return []
    by_template = Counter(row.get("template_id", "unknown") or "unknown" for row in failed_rows)
    by_reason = Counter(row.get("failure_reason", "unknown") or "unknown" for row in failed_rows)
    paths = []
    template_path = save_svg_bar_chart(
        output_dir / "failed_requests_by_template.svg",
        "Failed Requests by Template",
        by_template,
    )
    reason_path = save_svg_bar_chart(
        output_dir / "failed_requests_by_reason.svg",
        "Failed Requests by Reason",
        by_reason,
    )
    for path in (template_path, reason_path):
        if path is not None:
            paths.append(path)
    return paths


def save_ppo_reward_curve(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    rewards = [to_float(row.get("total_reward")) for row in rows]
    smoothed = moving_average(rewards, window)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(epochs, rewards, color="#9ca3af", linewidth=1.0, alpha=0.55, label="Raw reward")
    ax.plot(
        epochs,
        smoothed,
        color="#2563eb",
        linewidth=2.2,
        label=f"Moving average ({window})",
    )
    ax.set_title("PPO-Agent Training Reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total episode reward")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "ppo_reward_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_ppo_average_reward_per_request(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    values = [to_float(row.get("average_reward_per_request")) for row in rows]
    smoothed = moving_average(values, window)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(
        epochs,
        values,
        color="#9ca3af",
        linewidth=1.0,
        alpha=0.55,
        label="Raw reward / request",
    )
    ax.plot(
        epochs,
        smoothed,
        color="#2563eb",
        linewidth=2.2,
        label=f"Moving average ({window})",
    )
    ax.set_title("PPO-Agent Average Reward Per Request")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward / request")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "ppo_average_reward_per_request.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_ppo_diagnostics(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    success_rates = [to_float(row.get("success_rate")) for row in rows]
    losses = [to_float(row.get("ppo_loss")) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        epochs,
        moving_average(success_rates, window),
        color="#059669",
        linewidth=2.0,
        label="Success rate",
    )
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axes[0].legend()

    axes[1].plot(
        epochs,
        losses,
        color="#fecaca",
        linewidth=1.0,
        alpha=0.65,
        label="Raw PPO loss",
    )
    axes[1].plot(
        epochs,
        moving_average(losses, window),
        color="#dc2626",
        linewidth=2.0,
        label=f"PPO loss MA ({window})",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axes[1].legend()

    fig.suptitle("PPO-Agent Training Diagnostics")
    fig.tight_layout()
    path = output_dir / "ppo_training_diagnostics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_bandit_learning_curve(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    avg_arm_reward = [to_float(row.get("bandit_average_arm_reward")) for row in rows]
    positive_arms = [to_float(row.get("bandit_positive_arm_count")) for row in rows]
    known_arms = [to_float(row.get("bandit_known_arm_count")) for row in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(
        epochs,
        moving_average(avg_arm_reward, window),
        color="#7c3aed",
        linewidth=2.2,
        label="Mean arm reward",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean arm reward")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        positive_arms,
        color="#f97316",
        linewidth=1.8,
        label="Positive arms",
    )
    ax2.plot(
        epochs,
        known_arms,
        color="#0f766e",
        linewidth=1.8,
        label="Known arms",
    )
    ax2.set_ylabel("Arm count")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.suptitle("Bandit Strategy Learning Quality")
    fig.tight_layout()
    path = output_dir / "bandit_learning_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_bandit_action_distribution(plt, rows: list[dict], output_dir: Path) -> Path | None:
    if not rows:
        return None
    counts = Counter(row.get("action", "unknown") or "unknown" for row in rows)
    labels = list(counts)
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=["#2563eb", "#059669", "#f97316", "#64748b"][: len(labels)])
    ax.set_title("Bandit Migration Action Distribution")
    ax.set_xlabel("Action type")
    ax.set_ylabel("Action count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    path = output_dir / "bandit_action_distribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_slot_load_diagnostics(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    arrivals = [to_float(row.get("arrival_count")) for row in rows]
    total_rewards = [to_float(row.get("total_reward")) for row in rows]
    rewards_per_request = [to_float(row.get("average_reward_per_request")) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.4), sharex=True)
    axes[0].plot(epochs, arrivals, color="#0f766e", linewidth=1.2, label="Arrival count")
    axes[0].plot(
        epochs,
        moving_average(arrivals, window),
        color="#14b8a6",
        linewidth=2.3,
        label=f"Arrival MA ({window})",
    )
    axes[0].set_ylabel("Requests")
    axes[0].legend(loc="upper right")

    axes[1].plot(epochs, total_rewards, color="#94a3b8", linewidth=1.0, alpha=0.75)
    axes[1].plot(
        epochs,
        moving_average(total_rewards, window),
        color="#475569",
        linewidth=2.3,
    )
    axes[1].set_ylabel("Total reward")

    axes[2].plot(epochs, rewards_per_request, color="#60a5fa", linewidth=1.0, alpha=0.75)
    axes[2].plot(
        epochs,
        moving_average(rewards_per_request, window),
        color="#2563eb",
        linewidth=2.3,
    )
    axes[2].set_ylabel("Reward / request")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    fig.suptitle("Slot Load Diagnostics")
    fig.tight_layout()
    path = output_dir / "slot_load_diagnostics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_request_load_failure_curve(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    request_counts = [to_float(row.get("request_count", row.get("processed_request_count"))) for row in rows]
    feasible_counts = [to_float(row.get("feasible_count")) for row in rows]
    failed_counts = [max(0.0, request - feasible) for request, feasible in zip(request_counts, feasible_counts)]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(epochs, moving_average(request_counts, window), color="#2563eb", linewidth=2.0, label="Requests")
    ax.plot(epochs, moving_average(feasible_counts, window), color="#059669", linewidth=2.0, label="Feasible")
    ax.plot(epochs, moving_average(failed_counts, window), color="#dc2626", linewidth=2.0, label="Failed")
    ax.set_title("Request Load and Failures")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Request count")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "request_load_failure_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_failure_awareness_curve(plt, rows: list[dict], output_dir: Path, window: int) -> Path:
    epochs = [int(to_float(row.get("epoch"))) for row in rows]
    observations = [to_float(row.get("bandit_total_failed_replica_observations")) for row in rows]
    known_failed = [to_float(row.get("bandit_known_failed_replica_count")) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(epochs, observations, color="#dc2626", linewidth=2.0, label="Failed replica observations")
    ax.plot(epochs, known_failed, color="#f97316", linewidth=2.0, label="Known failed replicas")
    ax.set_title("Failure-Aware Bandit Signal")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "failure_awareness_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_request_failure_distribution(plt, rows: list[dict], output_dir: Path) -> list[Path]:
    failed_rows = [row for row in rows if str(row.get("feasible", "")).lower() == "false"]
    if not failed_rows:
        return []

    outputs = []
    charts = [
        (
            "failed_requests_by_template.png",
            "Failed Requests by Template",
            Counter(row.get("template_id", "unknown") or "unknown" for row in failed_rows),
            "Template ID",
        ),
        (
            "failed_requests_by_reason.png",
            "Failed Requests by Reason",
            Counter(row.get("failure_reason", "unknown") or "unknown" for row in failed_rows),
            "Failure reason",
        ),
    ]
    for filename, title, counts, xlabel in charts:
        labels = list(counts)
        values = [counts[label] for label in labels]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.8))
        ax.bar(labels, values, color="#dc2626")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Failed request count")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        if len(labels) > 4:
            ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    plt = import_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_rows = read_csv_rows(args.input_dir / "training_metrics.csv")
    if not training_rows:
        raise FileNotFoundError(
            f"No training rows found in {args.input_dir / 'training_metrics.csv'}"
        )

    bandit_action_rows = read_csv_rows(args.input_dir / "bandit_actions.csv")
    request_rows = read_csv_rows(args.input_dir / "request_metrics.csv")
    if plt is None:
        saved_paths = save_svg_fallback(
            training_rows, bandit_action_rows, args.output_dir, args.window
        )
        if args.extra_diagnostics:
            saved_paths.extend(save_svg_request_failure_charts(request_rows, args.output_dir))
    else:
        saved_paths = [
            save_slot_load_diagnostics(plt, training_rows, args.output_dir, args.window),
            save_ppo_diagnostics(plt, training_rows, args.output_dir, args.window),
            save_ppo_average_reward_per_request(
                plt, training_rows, args.output_dir, args.window
            ),
            save_bandit_learning_curve(plt, training_rows, args.output_dir, args.window),
        ]
        action_path = save_bandit_action_distribution(
            plt, bandit_action_rows, args.output_dir
        )
        if action_path is not None:
            saved_paths.append(action_path)
        if args.extra_diagnostics:
            saved_paths.extend(
                [
                    save_ppo_reward_curve(plt, training_rows, args.output_dir, args.window),
                    save_request_load_failure_curve(
                        plt, training_rows, args.output_dir, args.window
                    ),
                    save_failure_awareness_curve(
                        plt, training_rows, args.output_dir, args.window
                    ),
                ]
            )
            saved_paths.extend(save_request_failure_distribution(plt, request_rows, args.output_dir))

    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
