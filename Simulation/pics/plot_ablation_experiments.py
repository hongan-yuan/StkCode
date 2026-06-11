from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"

FONT_FAMILY = "Times New Roman"
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 11
SUMMARY_PANEL_TITLE_FONT_SIZE = 14
SUMMARY_BAR_LABEL_FONT_SIZE = 10
AXIS_SPINE_LINEWIDTH = 1.2
AXIS_TICK_LINEWIDTH = 1.0
PLOT_LINEWIDTH = 2.1
LEGEND_LINEWIDTH = 2.5
GRID_LINEWIDTH = 0.7
BOX_LINEWIDTH = 1.5
BOX_MEDIAN_LINEWIDTH = 2.0
BAR_EDGE_LINEWIDTH = 0.9
ERROR_BAR_LINEWIDTH = 1.0

ABLATION_LABELS = {
    "full": "Full",
    "no_bandit": "No Bandit",
    "shortest_hop_routing": "Shortest-Hop Routing",
    "nearest_replica": "Nearest Replica",
    "service_pressure": "Service Pressure",
    "sc_nfv": "SC-NFV",
}

METRICS = {
    "task_completion_rate": ("Task completion rate", "Task completion rate"),
    "average_end_to_end_delay_s": ("Average end-to-end delay", "Delay (s)"),
    "average_energy_j": ("Average energy", "Energy (J)"),
    "p95_end_to_end_delay_s": ("P95 end-to-end delay", "Delay (s)"),
    "average_communication_delay_s": ("Average communication delay", "Delay (s)"),
    "average_slot_crossings": ("Average slot crossings", "Slot crossings"),
}


def configure_matplotlib_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXIS_SPINE_LINEWIDTH,
            "xtick.major.width": AXIS_TICK_LINEWIDTH,
            "ytick.major.width": AXIS_TICK_LINEWIDTH,
            "grid.linewidth": GRID_LINEWIDTH,
            "lines.linewidth": PLOT_LINEWIDTH,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontfamily=FONT_FAMILY)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, fontfamily=FONT_FAMILY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, fontfamily=FONT_FAMILY)
    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
        width=AXIS_TICK_LINEWIDTH,
    )
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_SPINE_LINEWIDTH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ablation experiment curves and comparison charts."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ablations",
        default="full no_bandit shortest_hop_routing nearest_replica service_pressure sc_nfv",
        help="Space/comma separated ablation names and plotting order.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Moving-average window over slot index for line plots.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "png", "svg"),
        default="auto",
        help="Use PNG with matplotlib, SVG fallback, or force SVG.",
    )
    return parser.parse_args()


def parse_names(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def numeric_value(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None, "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_int(row: dict, key: str) -> int | None:
    value = numeric_value(row, key)
    return int(value) if value is not None else None


def finite_values(values) -> list[float]:
    result = []
    for value in values:
        if value in ("", None, "None", "null"):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def moving_average(values: list[float | None], window: int) -> list[float | None]:
    window = max(1, int(window))
    averaged = []
    finite_window: list[float] = []
    for value in values:
        if value is not None:
            finite_window.append(value)
        if len(finite_window) > window:
            finite_window.pop(0)
        averaged.append(mean(finite_window) if finite_window else None)
    return averaged


def load_slot_rows(input_dir: Path) -> list[dict]:
    rows = read_csv_rows(input_dir / "all_ablation_slot_metrics.csv")
    if rows:
        return rows
    rows = []
    for variant_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        metrics_path = variant_dir / "slot_metrics_by_seed.csv"
        for row in read_csv_rows(metrics_path):
            row.setdefault("ablation", variant_dir.name)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No ablation slot metrics found under {input_dir}")
    return rows


def load_cycle_rows(input_dir: Path) -> list[dict]:
    rows = read_csv_rows(input_dir / "all_ablation_cycle_metrics.csv")
    if rows:
        return rows
    rows = []
    for variant_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        metrics_path = variant_dir / "cycle_metrics_by_seed.csv"
        for row in read_csv_rows(metrics_path):
            row.setdefault("ablation", variant_dir.name)
            rows.append(row)
    return rows


def aggregate_by_slot(
    rows: list[dict],
    ablations: list[str],
    metric: str,
    window: int,
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = {
        ablation: defaultdict(list) for ablation in ablations
    }
    for row in rows:
        ablation = row.get("ablation", "")
        if ablation not in grouped:
            continue
        slot = row_int(row, "slot_mod")
        value = numeric_value(row, metric)
        if slot is None or value is None:
            continue
        grouped[ablation][slot].append(value)

    result = {}
    for ablation in ablations:
        slots = sorted(grouped[ablation])
        means = [mean(grouped[ablation][slot]) for slot in slots]
        stds = [std(grouped[ablation][slot]) for slot in slots]
        means = moving_average(means, window)
        stds = moving_average(stds, window)
        kept = [
            (slot, avg, spread)
            for slot, avg, spread in zip(slots, means, stds)
            if avg is not None and spread is not None
        ]
        result[ablation] = (
            [item[0] for item in kept],
            [item[1] for item in kept],
            [item[2] for item in kept],
        )
    return result


def values_by_ablation(
    rows: list[dict],
    ablations: list[str],
    metric: str,
) -> dict[str, list[float]]:
    grouped = {ablation: [] for ablation in ablations}
    for row in rows:
        ablation = row.get("ablation", "")
        if ablation not in grouped:
            continue
        value = numeric_value(row, metric)
        if value is not None:
            grouped[ablation].append(value)
    return grouped


def summary_means(
    rows: list[dict],
    ablations: list[str],
    metrics: list[str],
) -> dict[str, dict[str, tuple[float | None, float]]]:
    result = {}
    for ablation in ablations:
        ablation_rows = [row for row in rows if row.get("ablation") == ablation]
        result[ablation] = {}
        for metric in metrics:
            values = finite_values(row.get(metric, "") for row in ablation_rows)
            result[ablation][metric] = (mean(values), std(values))
    return result


def should_use_matplotlib(fmt: str) -> bool:
    return fmt == "png" or (fmt == "auto" and plt is not None)


def output_path(output_dir: Path, stem: str, fmt: str) -> Path:
    suffix = ".png" if should_use_matplotlib(fmt) else ".svg"
    return output_dir / f"{stem}{suffix}"


def plot_slot_curve_matplotlib(path: Path, title: str, ylabel: str, series: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=170)
    for ablation, (slots, means, stds) in series.items():
        if not slots:
            continue
        lower = [avg - spread for avg, spread in zip(means, stds)]
        upper = [avg + spread for avg, spread in zip(means, stds)]
        label = ABLATION_LABELS.get(ablation, ablation)
        line = ax.plot(slots, means, linewidth=PLOT_LINEWIDTH, label=label)[0]
        ax.fill_between(slots, lower, upper, alpha=0.16, color=line.get_color())
    style_axes(ax, title=title, xlabel="Time slot", ylabel=ylabel)
    ax.grid(True, alpha=0.25, linewidth=GRID_LINEWIDTH)
    legend = ax.legend(prop={"family": FONT_FAMILY, "size": LEGEND_FONT_SIZE})
    if legend is not None:
        legend.get_frame().set_linewidth(AXIS_SPINE_LINEWIDTH)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_box_matplotlib(path: Path, title: str, ylabel: str, data: dict, ablations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=170)
    values = [data.get(ablation, []) for ablation in ablations]
    labels = [ABLATION_LABELS.get(ablation, ablation) for ablation in ablations]
    ax.boxplot(
        values,
        labels=labels,
        showmeans=True,
        boxprops={"linewidth": BOX_LINEWIDTH},
        whiskerprops={"linewidth": BOX_LINEWIDTH},
        capprops={"linewidth": BOX_LINEWIDTH},
        medianprops={"linewidth": BOX_MEDIAN_LINEWIDTH},
        meanprops={"markeredgewidth": BOX_LINEWIDTH},
    )
    style_axes(ax, title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=GRID_LINEWIDTH)
    ax.tick_params(axis="x", rotation=15, width=AXIS_TICK_LINEWIDTH)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_summary_bar_matplotlib(path: Path, summaries: dict, metrics: list[str], ablations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8), dpi=170)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        title, ylabel = METRICS[metric]
        values = [summaries[ablation][metric][0] for ablation in ablations]
        errors = [summaries[ablation][metric][1] for ablation in ablations]
        labels = [ABLATION_LABELS.get(ablation, ablation) for ablation in ablations]
        positions = list(range(len(ablations)))
        ax.bar(
            positions,
            [value if value is not None else 0.0 for value in values],
            yerr=errors,
            capsize=4,
            color="#7dd3fc",
            edgecolor="#0f172a",
            linewidth=BAR_EDGE_LINEWIDTH,
            error_kw={
                "elinewidth": ERROR_BAR_LINEWIDTH,
                "capthick": ERROR_BAR_LINEWIDTH,
            },
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            labels,
            rotation=18,
            ha="right",
            fontsize=TICK_FONT_SIZE,
            fontfamily=FONT_FAMILY,
        )
        style_axes(ax, title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=GRID_LINEWIDTH)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if math.isclose(low, high):
        return (out_low + out_high) / 2
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def svg_canvas(title: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{TITLE_FONT_SIZE}">{escape(title)}</text>',
    ]


def plot_slot_curve_svg(path: Path, title: str, ylabel: str, series: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 460
    left, right, top, bottom = 80, 24, 52, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    all_slots = [slot for slots, _, _ in series.values() for slot in slots]
    all_values = [value for _, means, stds in series.values() for value in means + stds]
    if not all_slots:
        all_slots = [0, 1]
    if not all_values:
        all_values = [0.0, 1.0]
    x_min, x_max = min(all_slots), max(all_slots)
    y_min = min(0.0, min(all_values))
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    parts = svg_canvas(title, width, height)
    parts.extend([
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155" stroke-width="{AXIS_SPINE_LINEWIDTH}"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155" stroke-width="{AXIS_SPINE_LINEWIDTH}"/>',
        f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{AXIS_LABEL_FONT_SIZE}">Time slot</text>',
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{AXIS_LABEL_FONT_SIZE}">{escape(ylabel)}</text>',
    ])
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = scale(value, y_min, y_max, height - bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="{GRID_LINEWIDTH}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="{FONT_FAMILY}" font-size="{TICK_FONT_SIZE}">{value:.3g}</text>')
    legend_y = top
    for idx, (ablation, (slots, means, _)) in enumerate(series.items()):
        if not slots:
            continue
        color = colors[idx % len(colors)]
        points = [
            (
                scale(slot, x_min, x_max, left, left + plot_w),
                scale(value, y_min, y_max, height - bottom, top),
            )
            for slot, value in zip(slots, means)
        ]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{PLOT_LINEWIDTH}" points="{point_text}"/>')
        parts.append(f'<line x1="{width - 230}" y1="{legend_y}" x2="{width - 200}" y2="{legend_y}" stroke="{color}" stroke-width="{LEGEND_LINEWIDTH}"/>')
        parts.append(f'<text x="{width - 194}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" font-size="{LEGEND_FONT_SIZE}">{escape(ABLATION_LABELS.get(ablation, ablation))}</text>')
        legend_y += 18
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return math.inf
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def plot_box_svg(path: Path, title: str, ylabel: str, data: dict, ablations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = max(820, 170 * len(ablations)), 460
    left, right, top, bottom = 84, 24, 52, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [value for ablation in ablations for value in data.get(ablation, [])]
    if not all_values:
        all_values = [0.0, 1.0]
    y_min, y_max = min(all_values), max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad
    parts = svg_canvas(title, width, height)
    parts.extend([
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155" stroke-width="{AXIS_SPINE_LINEWIDTH}"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155" stroke-width="{AXIS_SPINE_LINEWIDTH}"/>',
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{AXIS_LABEL_FONT_SIZE}">{escape(ylabel)}</text>',
    ])
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = scale(value, y_min, y_max, height - bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="{GRID_LINEWIDTH}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="{FONT_FAMILY}" font-size="{TICK_FONT_SIZE}">{value:.3g}</text>')
    for index, ablation in enumerate(ablations):
        values = sorted(data.get(ablation, []))
        x = left + (index + 0.5) * plot_w / max(1, len(ablations))
        label = ABLATION_LABELS.get(ablation, ablation)
        parts.append(f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{TICK_FONT_SIZE}">{escape(label)}</text>')
        if not values:
            continue
        q1, med, q3 = quantile(values, 0.25), quantile(values, 0.5), quantile(values, 0.75)
        low, high = values[0], values[-1]
        avg = sum(values) / len(values)
        box_w = min(58, plot_w / max(1, len(ablations)) * 0.45)
        y_low = scale(low, y_min, y_max, height - bottom, top)
        y_high = scale(high, y_min, y_max, height - bottom, top)
        y_q1 = scale(q1, y_min, y_max, height - bottom, top)
        y_q3 = scale(q3, y_min, y_max, height - bottom, top)
        y_med = scale(med, y_min, y_max, height - bottom, top)
        y_avg = scale(avg, y_min, y_max, height - bottom, top)
        parts.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="#475569" stroke-width="{BOX_LINEWIDTH}"/>')
        parts.append(f'<rect x="{x - box_w / 2:.1f}" y="{y_q3:.1f}" width="{box_w:.1f}" height="{max(1.0, y_q1 - y_q3):.1f}" fill="#dbeafe" stroke="#2563eb" stroke-width="{BOX_LINEWIDTH}"/>')
        parts.append(f'<line x1="{x - box_w / 2:.1f}" y1="{y_med:.1f}" x2="{x + box_w / 2:.1f}" y2="{y_med:.1f}" stroke="#1e3a8a" stroke-width="{BOX_MEDIAN_LINEWIDTH}"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y_avg:.1f}" r="3" fill="#dc2626"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_summary_bar_svg(path: Path, summaries: dict, metrics: list[str], ablations: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = max(920, 220 * len(metrics)), 470
    parts = svg_canvas("Ablation Metric Summary", width, height)
    panel_w = (width - 50) / len(metrics)
    colors = ["#7dd3fc", "#fecaca", "#bbf7d0", "#ddd6fe", "#fed7aa", "#bae6fd"]
    for m_index, metric in enumerate(metrics):
        title, ylabel = METRICS[metric]
        x0 = 32 + panel_w * m_index
        y0 = 62
        h = 300
        w = panel_w - 28
        values = [summaries[ablation][metric][0] for ablation in ablations]
        finite = [value for value in values if value is not None]
        y_max = max(finite) if finite else 1.0
        if y_max <= 0:
            y_max = 1.0
        parts.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 - 14}" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{SUMMARY_PANEL_TITLE_FONT_SIZE}">{escape(title)}</text>')
        parts.append(f'<line x1="{x0 + 48:.1f}" y1="{y0 + h:.1f}" x2="{x0 + w:.1f}" y2="{y0 + h:.1f}" stroke="#334155" stroke-width="{AXIS_SPINE_LINEWIDTH}"/>')
        parts.append(f'<line x1="{x0 + 48:.1f}" y1="{y0:.1f}" x2="{x0 + 48:.1f}" y2="{y0 + h:.1f}" stroke="#334155" stroke-width="{AXIS_SPINE_LINEWIDTH}"/>')
        for index, ablation in enumerate(ablations):
            value = values[index]
            if value is None:
                continue
            bar_w = (w - 66) / max(1, len(ablations)) * 0.65
            x = x0 + 58 + index * (w - 66) / max(1, len(ablations))
            bar_h = value / y_max * h
            y = y0 + h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[index % len(colors)]}" stroke="#0f172a" stroke-width="{BAR_EDGE_LINEWIDTH}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + h + 16}" transform="rotate(18 {x + bar_w / 2:.1f} {y0 + h + 16})" text-anchor="start" font-family="{FONT_FAMILY}" font-size="{SUMMARY_BAR_LABEL_FONT_SIZE}">{escape(ABLATION_LABELS.get(ablation, ablation))}</text>')
        parts.append(f'<text x="{x0 + 12:.1f}" y="{y0 + h / 2:.1f}" transform="rotate(-90 {x0 + 12:.1f} {y0 + h / 2:.1f})" text-anchor="middle" font-family="{FONT_FAMILY}" font-size="{TICK_FONT_SIZE}">{escape(ylabel)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_slot_curve(path: Path, title: str, ylabel: str, series: dict, fmt: str) -> Path:
    actual = output_path(path.parent, path.stem, fmt)
    if should_use_matplotlib(fmt):
        plot_slot_curve_matplotlib(actual, title, ylabel, series)
    else:
        plot_slot_curve_svg(actual, title, ylabel, series)
    return actual


def plot_box(path: Path, title: str, ylabel: str, data: dict, ablations: list[str], fmt: str) -> Path:
    actual = output_path(path.parent, path.stem, fmt)
    if should_use_matplotlib(fmt):
        plot_box_matplotlib(actual, title, ylabel, data, ablations)
    else:
        plot_box_svg(actual, title, ylabel, data, ablations)
    return actual


def plot_summary_bar(path: Path, summaries: dict, metrics: list[str], ablations: list[str], fmt: str) -> Path:
    actual = output_path(path.parent, path.stem, fmt)
    if should_use_matplotlib(fmt):
        plot_summary_bar_matplotlib(actual, summaries, metrics, ablations)
    else:
        plot_summary_bar_svg(actual, summaries, metrics, ablations)
    return actual


def main() -> None:
    args = parse_args()
    ablations = parse_names(args.ablations)
    rows = load_slot_rows(args.input_dir)
    cycle_rows = load_cycle_rows(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "svg" if args.format == "png" and plt is None else args.format

    generated: list[Path] = []
    summary_metrics = [
        "task_completion_rate",
        "average_end_to_end_delay_s",
        "average_energy_j",
        "p95_end_to_end_delay_s",
    ]
    summary_rows = cycle_rows if cycle_rows else rows
    summaries = summary_means(summary_rows, ablations, summary_metrics)
    generated.append(
        plot_summary_bar(
            args.output_dir / "ablation_metric_summary",
            summaries,
            summary_metrics,
            ablations,
            fmt,
        )
    )

    for metric in [
        "task_completion_rate",
        "average_end_to_end_delay_s",
        "average_energy_j",
        "p95_end_to_end_delay_s",
        "average_communication_delay_s",
        "average_slot_crossings",
    ]:
        title, ylabel = METRICS[metric]
        series = aggregate_by_slot(rows, ablations, metric, args.window)
        generated.append(
            plot_slot_curve(
                args.output_dir / f"{metric}_by_slot",
                f"{title} by time slot",
                ylabel,
                series,
                fmt,
            )
        )

    for metric in [
        "task_completion_rate",
        "average_end_to_end_delay_s",
        "average_energy_j",
        "p95_end_to_end_delay_s",
    ]:
        title, ylabel = METRICS[metric]
        data = values_by_ablation(rows, ablations, metric)
        generated.append(
            plot_box(
                args.output_dir / f"{metric}_distribution",
                f"{title} distribution by ablation",
                ylabel,
                data,
                ablations,
                fmt,
            )
        )

    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(
        "\n".join(str(path) for path in generated) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} plot files to {args.output_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
