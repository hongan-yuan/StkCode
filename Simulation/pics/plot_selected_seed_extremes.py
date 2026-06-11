from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on local plotting env
    plt = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"

ABLATION_LABELS = {
    "full": "Full",
    "no_bandit": "No Bandit",
    "shortest_hop_routing": "Shortest-Hop",
    "nearest_replica": "Nearest Replica",
    "service_pressure": "Service Pressure",
    "sc_nfv": "SC-NFV",
    "fairness_nfv_greedy": "Fairness-NFV",
}

METRICS = {
    "task_completion_rate": ("Task completion rate", "Task completion rate"),
    "average_end_to_end_delay_s": ("Average end-to-end delay", "Delay (s)"),
    "average_energy_j": ("Average energy", "Energy (J)"),
    "p95_end_to_end_delay_s": ("P95 end-to-end delay", "Delay (s)"),
    "average_communication_delay_s": ("Average communication delay", "Delay (s)"),
    "average_slot_crossings": ("Average slot crossings", "Slot crossings"),
    "deadline_acceptance_rate": ("Deadline acceptance rate", "Acceptance rate"),
    "delay_margin_jain_fairness": ("Delay-margin fairness", "Jain fairness"),
}

HIGHER_BETTER = (
    "task_completion_rate",
    "deadline_acceptance_rate",
    "delay_margin_jain_fairness",
)
LOWER_BETTER = (
    "average_end_to_end_delay_s",
    "average_energy_j",
    "p95_end_to_end_delay_s",
    "average_communication_delay_s",
    "average_slot_crossings",
)
SELECTION_METRICS = HIGHER_BETTER + LOWER_BETTER

DEFAULT_ABLATIONS = (
    "full",
    "no_bandit",
    "shortest_hop_routing",
    "nearest_replica",
    "service_pressure",
    "sc_nfv",
    "fairness_nfv_greedy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the target method's best seed and each baseline's worst "
            "seed, then render a separate comparison figure."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<input-dir>/selected_seed_plots."
        ),
    )
    parser.add_argument(
        "--target-ablation",
        default="full",
        help="Ablation whose best seed is selected as our method.",
    )
    parser.add_argument(
        "--ablations",
        default=" ".join(DEFAULT_ABLATIONS),
        help="Space/comma separated ablation names and plotting order.",
    )
    parser.add_argument(
        "--metrics",
        default=" ".join(METRICS),
        help="Space/comma separated metric names to plot.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "png", "svg", "both"),
        default="auto",
        help="Figure format. auto uses PNG when matplotlib is available, SVG otherwise.",
    )
    parser.add_argument(
        "--score-metrics",
        default=" ".join(SELECTION_METRICS),
        help=(
            "Space/comma separated metrics used for seed selection. "
            "Direction is inferred from the built-in higher/lower-better sets."
        ),
    )
    return parser.parse_args()


def parse_names(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def numeric(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None, "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_cycle_rows(input_dir: Path) -> list[dict]:
    merged = read_csv_rows(input_dir / "all_ablation_cycle_metrics.csv")
    if merged:
        return merged

    rows: list[dict] = []
    for path in sorted(input_dir.glob("*/cycle_metrics_by_seed.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def grouped_by_ablation(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        ablation = str(row.get("ablation", ""))
        if ablation:
            grouped[ablation].append(row)
    return grouped


def metric_ranges(rows: list[dict], metrics: list[str]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for metric in metrics:
        values = [value for row in rows if (value := numeric(row, metric)) is not None]
        if values:
            ranges[metric] = (min(values), max(values))
    return ranges


def normalized_metric_score(
    row: dict,
    metric: str,
    ranges: dict[str, tuple[float, float]],
) -> float | None:
    value = numeric(row, metric)
    if value is None or metric not in ranges:
        return None
    low, high = ranges[metric]
    if math.isclose(low, high):
        return 1.0
    if metric in HIGHER_BETTER:
        return (value - low) / (high - low)
    if metric in LOWER_BETTER:
        return (high - value) / (high - low)
    return (value - low) / (high - low)


def selection_score(
    row: dict,
    ranges: dict[str, tuple[float, float]],
    metrics: list[str],
) -> float:
    scores = [
        score
        for metric in metrics
        if (score := normalized_metric_score(row, metric, ranges)) is not None
    ]
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


def stable_seed_value(row: dict) -> int:
    try:
        return int(float(row.get("seed", 0)))
    except (TypeError, ValueError):
        return 0


def select_extreme_rows(
    rows: list[dict],
    ablations: list[str],
    target_ablation: str,
    score_metrics: list[str],
) -> list[dict]:
    grouped = grouped_by_ablation(rows)
    selected = []
    for ablation in ablations:
        candidates = grouped.get(ablation, [])
        if not candidates:
            continue
        ranges = metric_ranges(candidates, score_metrics)
        scored = []
        for row in candidates:
            enriched = dict(row)
            score = selection_score(row, ranges, score_metrics)
            enriched["selection_score"] = f"{score:.12g}" if math.isfinite(score) else ""
            enriched["selection_role"] = (
                "target_best" if ablation == target_ablation else "baseline_worst"
            )
            scored.append((score, enriched))

        if ablation == target_ablation:
            chosen = max(
                scored,
                key=lambda item: (
                    item[0],
                    numeric(item[1], "task_completion_rate") or -math.inf,
                    -(numeric(item[1], "average_end_to_end_delay_s") or math.inf),
                    -(numeric(item[1], "p95_end_to_end_delay_s") or math.inf),
                    -stable_seed_value(item[1]),
                ),
            )[1]
        else:
            chosen = min(
                scored,
                key=lambda item: (
                    item[0],
                    numeric(item[1], "task_completion_rate") or math.inf,
                    -(numeric(item[1], "average_end_to_end_delay_s") or -math.inf),
                    stable_seed_value(item[1]),
                ),
            )[1]
        selected.append(chosen)
    return selected


def selected_key(row: dict) -> tuple[str, str]:
    return str(row.get("ablation", "")), str(row.get("seed", ""))


def filter_selected_rows(rows: list[dict], selected: list[dict]) -> list[dict]:
    keys = {selected_key(row) for row in selected}
    return [row for row in rows if selected_key(row) in keys]


def label_for_row(row: dict) -> str:
    ablation = str(row.get("ablation", ""))
    seed = str(row.get("seed", ""))
    label = ABLATION_LABELS.get(ablation, ablation)
    return f"{label}\nseed {seed}"


def configure_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, output_dir: Path, stem: str, fmt: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "auto":
        formats = ["png"]
    else:
        formats = ["png", "svg"] if fmt == "both" else [fmt]
    paths = []
    for item in formats:
        path = output_dir / f"{stem}.{item}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        paths.append(str(path))
    return paths


def svg_text(x: float, y: float, text: str, **attrs) -> str:
    attr_text = " ".join(
        f'{("class" if key == "class_" else key.replace("_", "-"))}="{escape(str(value))}"'
        for key, value in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attr_text}>{escape(str(text))}</text>'


def svg_line(x1: float, y1: float, x2: float, y2: float, **attrs) -> str:
    attr_text = " ".join(
        f'{("class" if key == "class_" else key.replace("_", "-"))}="{escape(str(value))}"'
        for key, value in attrs.items()
    )
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attr_text}/>'


def svg_rect(x: float, y: float, width: float, height: float, **attrs) -> str:
    attr_text = " ".join(
        f'{("class" if key == "class_" else key.replace("_", "-"))}="{escape(str(value))}"'
        for key, value in attrs.items()
    )
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" {attr_text}/>'


def svg_polyline(points: list[tuple[float, float]], **attrs) -> str:
    if not points:
        return ""
    attr_text = " ".join(
        f'{("class" if key == "class_" else key.replace("_", "-"))}="{escape(str(value))}"'
        for key, value in attrs.items()
    )
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{point_text}" {attr_text}/>'


def write_svg(path: Path, width: int, height: int, body: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Times New Roman,DejaVu Serif,serif;} .small{font-size:10px;} .tick{font-size:9px;} .title{font-size:14px;font-weight:bold;} .panel{font-size:12px;font-weight:bold;}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return str(path)


def metric_axis_max(values: list[float], metric: str) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 1.0
    high = max(finite)
    if metric in HIGHER_BETTER and high <= 1.0:
        return 1.0
    return high * 1.12 if high > 0.0 else 1.0


def plot_metric_panels_svg(
    selected: list[dict],
    metrics: list[str],
    output_dir: Path,
) -> list[str]:
    cols = 4
    panel_w = 320
    panel_h = 255
    rows = math.ceil(len(metrics) / cols)
    width = cols * panel_w
    height = rows * panel_h + 45
    labels = [label_for_row(row).replace("\n", " ") for row in selected]
    body = [svg_text(width / 2, 25, "Selected seed comparison: target best vs baseline worst", text_anchor="middle", class_="title")]
    colors = ["#2f77b4"] + ["#8cc7e8"] * max(0, len(selected) - 1)
    for metric_index, metric in enumerate(metrics):
        row_i, col_i = divmod(metric_index, cols)
        ox = col_i * panel_w + 45
        oy = row_i * panel_h + 58
        plot_w = panel_w - 75
        plot_h = 145
        title, ylabel = METRICS.get(metric, (metric, metric))
        values = [numeric(row, metric) for row in selected]
        y_max = metric_axis_max([value for value in values if value is not None], metric)
        body.append(svg_text(ox + plot_w / 2, oy - 20, title, text_anchor="middle", class_="panel"))
        body.append(svg_line(ox, oy + plot_h, ox + plot_w, oy + plot_h, stroke="#333", stroke_width="1"))
        body.append(svg_line(ox, oy, ox, oy + plot_h, stroke="#333", stroke_width="1"))
        body.append(svg_text(ox - 35, oy + plot_h / 2, ylabel, text_anchor="middle", class_="tick", transform=f"rotate(-90 {ox - 35:.1f} {oy + plot_h / 2:.1f})"))
        body.append(svg_text(ox - 5, oy + 4, f"{y_max:.3g}", text_anchor="end", class_="tick"))
        body.append(svg_text(ox - 5, oy + plot_h, "0", text_anchor="end", class_="tick"))
        gap = 5
        bar_w = max(10, (plot_w - gap * (len(selected) + 1)) / max(1, len(selected)))
        for index, value in enumerate(values):
            if value is None:
                continue
            bar_h = max(0.0, min(plot_h, (value / y_max) * plot_h))
            x = ox + gap + index * (bar_w + gap)
            y = oy + plot_h - bar_h
            body.append(svg_rect(x, y, bar_w, bar_h, fill=colors[index], stroke="#1f3f5b", stroke_width="0.7"))
            body.append(svg_text(x + bar_w / 2, y - 3, f"{value:.3g}", text_anchor="middle", class_="tick"))
            body.append(svg_text(x + bar_w / 2, oy + plot_h + 14, labels[index].split(" seed ")[0], text_anchor="middle", class_="tick", transform=f"rotate(35 {x + bar_w / 2:.1f} {oy + plot_h + 14:.1f})"))
            body.append(svg_text(x + bar_w / 2, oy + plot_h + 28, "s" + labels[index].split(" seed ")[-1], text_anchor="middle", class_="tick", transform=f"rotate(35 {x + bar_w / 2:.1f} {oy + plot_h + 28:.1f})"))
    path = output_dir / "selected_seed_metric_panels.svg"
    return [write_svg(path, width, height, body)]


def plot_metric_panels(
    selected: list[dict],
    metrics: list[str],
    output_dir: Path,
    fmt: str,
) -> list[str]:
    if plt is None:
        return plot_metric_panels_svg(selected, metrics, output_dir)
    configure_style()
    cols = 4
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    x = list(range(len(selected)))
    colors = ["#2f77b4"] + ["#8cc7e8"] * max(0, len(selected) - 1)
    labels = [label_for_row(row) for row in selected]

    for ax, metric in zip(flat_axes, metrics):
        title, ylabel = METRICS.get(metric, (metric, metric))
        values = [numeric(row, metric) for row in selected]
        plot_values = [value if value is not None else 0.0 for value in values]
        ax.bar(x, plot_values, color=colors, edgecolor="#1f3f5b", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            if value is None:
                continue
            ax.text(
                index,
                plot_values[index],
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    for ax in flat_axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Selected seed comparison: target best vs baseline worst", y=1.02)
    return save_figure(fig, output_dir, "selected_seed_metric_panels", fmt)


def plot_selection_scores(
    selected: list[dict],
    output_dir: Path,
    fmt: str,
) -> list[str]:
    if plt is None:
        width = max(760, 120 * len(selected))
        height = 360
        ox, oy, plot_w, plot_h = 70, 55, width - 120, 210
        labels = [label_for_row(row).replace("\n", " ") for row in selected]
        values = [numeric(row, "selection_score") or 0.0 for row in selected]
        colors = ["#2f77b4"] + ["#8cc7e8"] * max(0, len(selected) - 1)
        body = [svg_text(width / 2, 25, "Seed selection score", text_anchor="middle", class_="title")]
        body.append(svg_line(ox, oy + plot_h, ox + plot_w, oy + plot_h, stroke="#333", stroke_width="1"))
        body.append(svg_line(ox, oy, ox, oy + plot_h, stroke="#333", stroke_width="1"))
        body.append(svg_text(ox - 38, oy + plot_h / 2, "Selection utility", text_anchor="middle", class_="small", transform=f"rotate(-90 {ox - 38:.1f} {oy + plot_h / 2:.1f})"))
        gap = 9
        bar_w = max(16, (plot_w - gap * (len(selected) + 1)) / max(1, len(selected)))
        for index, value in enumerate(values):
            bar_h = max(0.0, min(plot_h, value * plot_h))
            x = ox + gap + index * (bar_w + gap)
            y = oy + plot_h - bar_h
            body.append(svg_rect(x, y, bar_w, bar_h, fill=colors[index], stroke="#1f3f5b", stroke_width="0.7"))
            body.append(svg_text(x + bar_w / 2, y - 4, f"{value:.3f}", text_anchor="middle", class_="tick"))
            body.append(svg_text(x + bar_w / 2, oy + plot_h + 18, labels[index], text_anchor="middle", class_="tick", transform=f"rotate(25 {x + bar_w / 2:.1f} {oy + plot_h + 18:.1f})"))
        return [write_svg(output_dir / "selected_seed_scores.svg", width, height, body)]
    configure_style()
    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(selected)), 4.2))
    labels = [label_for_row(row) for row in selected]
    values = [numeric(row, "selection_score") or 0.0 for row in selected]
    colors = ["#2f77b4"] + ["#8cc7e8"] * max(0, len(selected) - 1)
    ax.bar(range(len(selected)), values, color=colors, edgecolor="#1f3f5b", linewidth=0.8)
    ax.set_ylabel("Selection utility")
    ax.set_title("Seed selection score")
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    return save_figure(fig, output_dir, "selected_seed_scores", fmt)


def plot_slot_curves(
    selected: list[dict],
    slot_rows: list[dict],
    output_dir: Path,
    fmt: str,
) -> list[str]:
    if plt is None:
        if not slot_rows:
            return []
        curve_metrics = [
            "task_completion_rate",
            "average_end_to_end_delay_s",
            "average_energy_j",
            "p95_end_to_end_delay_s",
        ]
        width, height = 1100, 680
        panel_w, panel_h = 520, 285
        colors = ["#2f77b4", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in slot_rows:
            grouped[selected_key(row)].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: numeric(row, "epoch") or 0.0)
        body = [svg_text(width / 2, 25, "Selected seed slot curves", text_anchor="middle", class_="title")]
        for metric_index, metric in enumerate(curve_metrics):
            row_i, col_i = divmod(metric_index, 2)
            ox = col_i * panel_w + 70
            oy = row_i * panel_h + 60
            plot_w = panel_w - 115
            plot_h = 190
            title, ylabel = METRICS.get(metric, (metric, metric))
            all_points = []
            for selected_row in selected:
                for item in grouped.get(selected_key(selected_row), []):
                    x = numeric(item, "epoch")
                    y = numeric(item, metric)
                    if x is not None and y is not None:
                        all_points.append((x, y))
            if not all_points:
                continue
            x_min, x_max = min(x for x, _ in all_points), max(x for x, _ in all_points)
            y_values = [y for _, y in all_points]
            y_min = 0.0 if metric in HIGHER_BETTER else min(y_values)
            y_max = metric_axis_max(y_values, metric)
            if math.isclose(x_min, x_max):
                x_max = x_min + 1.0
            if math.isclose(y_min, y_max):
                y_max = y_min + 1.0
            body.append(svg_text(ox + plot_w / 2, oy - 18, title, text_anchor="middle", class_="panel"))
            body.append(svg_line(ox, oy + plot_h, ox + plot_w, oy + plot_h, stroke="#333", stroke_width="1"))
            body.append(svg_line(ox, oy, ox, oy + plot_h, stroke="#333", stroke_width="1"))
            body.append(svg_text(ox - 38, oy + plot_h / 2, ylabel, text_anchor="middle", class_="tick", transform=f"rotate(-90 {ox - 38:.1f} {oy + plot_h / 2:.1f})"))
            for idx, selected_row in enumerate(selected):
                rows = grouped.get(selected_key(selected_row), [])
                points = []
                for item in rows:
                    x = numeric(item, "epoch")
                    y = numeric(item, metric)
                    if x is None or y is None:
                        continue
                    px = ox + (x - x_min) / (x_max - x_min) * plot_w
                    py = oy + plot_h - (y - y_min) / (y_max - y_min) * plot_h
                    points.append((px, py))
                body.append(svg_polyline(points, fill="none", stroke=colors[idx % len(colors)], stroke_width="1.3"))
        legend_y = height - 55
        for idx, row in enumerate(selected):
            x = 70 + (idx % 4) * 250
            y = legend_y + (idx // 4) * 18
            body.append(svg_line(x, y - 4, x + 24, y - 4, stroke=colors[idx % len(colors)], stroke_width="2"))
            body.append(svg_text(x + 30, y, label_for_row(row).replace("\n", " "), class_="tick"))
        return [write_svg(output_dir / "selected_seed_slot_curves.svg", width, height, body)]
    if not slot_rows:
        return []
    configure_style()
    curve_metrics = [
        "task_completion_rate",
        "average_end_to_end_delay_s",
        "average_energy_j",
        "p95_end_to_end_delay_s",
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in slot_rows:
        grouped[selected_key(row)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: numeric(row, "epoch") or 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    flat_axes = axes.flatten()
    for ax, metric in zip(flat_axes, curve_metrics):
        title, ylabel = METRICS.get(metric, (metric, metric))
        for row in selected:
            key = selected_key(row)
            rows = grouped.get(key, [])
            xs = [numeric(item, "epoch") for item in rows]
            ys = [numeric(item, metric) for item in rows]
            points = [
                (x, y)
                for x, y in zip(xs, ys)
                if x is not None and y is not None and math.isfinite(y)
            ]
            if not points:
                continue
            ax.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                linewidth=1.3,
                label=label_for_row(row).replace("\n", " "),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in flat_axes[-2:]:
        ax.set_xlabel("Slot")
    flat_axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Selected seed slot curves", y=1.02)
    return save_figure(fig, output_dir, "selected_seed_slot_curves", fmt)


def read_summary_metadata(input_dir: Path, selected: list[dict]) -> dict[str, dict]:
    metadata = {}
    for row in selected:
        ablation, seed = selected_key(row)
        path = input_dir / ablation / f"seed_{seed}" / "summary.json"
        if not path.exists():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata[f"{ablation}:{seed}"] = {
            "checkpoint_loaded": summary.get("checkpoint_loaded"),
            "checkpoint_load_error": summary.get("checkpoint_load_error"),
            "bandit_loaded_arm_count": summary.get("bandit_loaded_arm_count"),
            "service_routing_strategy": summary.get("service_routing_strategy"),
            "execution_agent": summary.get("execution_agent"),
            "request_count": summary.get("overall_summary", {}).get("request_count"),
            "feasible_count": summary.get("overall_summary", {}).get("feasible_count"),
        }
    return metadata


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "selected_seed_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = parse_names(args.ablations)
    metrics = parse_names(args.metrics)
    score_metrics = parse_names(args.score_metrics)
    cycle_rows = load_cycle_rows(input_dir)
    if not cycle_rows:
        raise SystemExit(f"No cycle metrics found under {input_dir}.")

    cycle_rows = [row for row in cycle_rows if row.get("ablation") in set(ablations)]
    selected = select_extreme_rows(
        cycle_rows,
        ablations,
        args.target_ablation,
        score_metrics,
    )
    selected_keys = {selected_key(row) for row in selected}
    if not selected_keys:
        raise SystemExit("No selected rows were produced.")

    slot_rows = filter_selected_rows(
        read_csv_rows(input_dir / "all_ablation_slot_metrics.csv"),
        selected,
    )
    request_rows = filter_selected_rows(
        read_csv_rows(input_dir / "all_ablation_request_metrics.csv"),
        selected,
    )

    write_csv_rows(output_dir / "selected_cycle_metrics.csv", selected)
    write_csv_rows(output_dir / "selected_slot_metrics.csv", slot_rows)
    write_csv_rows(output_dir / "selected_request_metrics.csv", request_rows)

    figure_paths = []
    figure_paths.extend(plot_metric_panels(selected, metrics, output_dir, args.format))
    figure_paths.extend(plot_selection_scores(selected, output_dir, args.format))
    figure_paths.extend(plot_slot_curves(selected, slot_rows, output_dir, args.format))

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_ablation": args.target_ablation,
        "selection_rule": (
            "target ablation chooses max normalized utility; every other "
            "ablation chooses min normalized utility within that ablation"
        ),
        "higher_better_metrics": list(HIGHER_BETTER),
        "lower_better_metrics": list(LOWER_BETTER),
        "score_metrics": score_metrics,
        "plotted_metrics": metrics,
        "selected": [
            {
                "ablation": row.get("ablation"),
                "seed": row.get("seed"),
                "selection_role": row.get("selection_role"),
                "selection_score": row.get("selection_score"),
            }
            for row in selected
        ],
        "metadata": read_summary_metadata(input_dir, selected),
        "figures": figure_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["selected"], ensure_ascii=False, indent=2))
    print(f"Figures and selected CSVs written to {output_dir}")


if __name__ == "__main__":
    main()
