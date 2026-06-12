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

from ..ablation_names import (
    ABLATION_GROUP_ABLATIONS,
    ABLATION_LABELS,
    canonical_ablation_names,
    canonicalize_ablation_row,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "cross_slot_plots"
DEFAULT_ABLATIONS = " ".join(ABLATION_GROUP_ABLATIONS)

COLORS = {
    "ELARA": "#2f6fbb",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
    "Fair-NFV": "#cc4c4c",
    "SP-Routing": "#0891b2",
    "SC-NFV": "#6b7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 10-slot request latency and energy windows. Only slots 1-600 "
            "are included; slots 601-606 are excluded."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablations", default=DEFAULT_ABLATIONS)
    parser.add_argument("--slot-window", type=int, default=10)
    parser.add_argument("--max-slot", type=int, default=600)
    parser.add_argument("--format", choices=("png",), default="png")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [canonicalize_ablation_row(row) for row in csv.DictReader(handle)]


def load_request_rows(input_dir: Path, ablations: list[str]) -> list[dict]:
    rows = []
    for ablation in ablations:
        variant_path = input_dir / ablation / "request_metrics_by_seed.csv"
        variant_rows = read_rows(variant_path)
        if not variant_rows:
            for seed_dir in sorted((input_dir / ablation).glob("seed_*")):
                variant_rows.extend(read_rows(seed_dir / "request_metrics.csv"))
        for row in variant_rows:
            row["ablation"] = row.get("ablation") or ablation
            canonicalize_ablation_row(row)
            rows.append(row)
    if not rows:
        rows = read_rows(input_dir / "all_ablation_request_metrics.csv")
    return [row for row in rows if row.get("ablation") in ablations]


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def window_means(
    rows: list[dict],
    ablations: list[str],
    metric: str,
    slot_window: int,
    max_slot: int,
) -> tuple[list[int], dict[str, list[float]]]:
    window_count = max_slot // slot_window
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        ablation = row.get("ablation")
        epoch = number(row, "epoch")
        value = number(row, metric)
        if ablation not in ablations or epoch is None or value is None:
            continue
        slot = int(epoch)
        if slot < 1 or slot > max_slot:
            continue
        window_index = (slot - 1) // slot_window + 1
        buckets[(ablation, window_index)].append(value)
    windows = list(range(1, window_count + 1))
    series = {}
    for ablation in ablations:
        values = []
        for window_index in windows:
            bucket = buckets.get((ablation, window_index), [])
            values.append(float(np.mean(bucket)) if bucket else math.nan)
        series[ablation] = values
    return windows, series


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def plot_curve(
    output_path: Path,
    rows: list[dict],
    ablations: list[str],
    metric: str,
    ylabel: str,
    title: str,
    slot_window: int,
    max_slot: int,
) -> None:
    windows, series = window_means(rows, ablations, metric, slot_window, max_slot)
    fig, ax = plt.subplots(figsize=(12.6, 5.8), constrained_layout=True)
    for ablation in ablations:
        ax.plot(
            windows,
            series[ablation],
            color=COLORS.get(ablation, "#4b5563"),
            linewidth=1.9,
            marker="o",
            markersize=3.0,
            label=ABLATION_LABELS.get(ablation, ablation),
        )
    ax.set_title(title, fontsize=12, pad=9)
    ax.set_xlabel(f"10-slot window index, slots 1-{max_slot}")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, len(windows))
    ax.set_xticks(list(range(1, len(windows) + 1, 5)))
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    ablations = canonical_ablation_names(args.ablations)
    rows = load_request_rows(args.input_dir, ablations)
    latency_path = args.output_dir / "latency_cross_slot.png"
    energy_path = args.output_dir / "energy_cross_slot.png"
    plot_curve(
        latency_path,
        rows,
        ablations,
        "total_delay_s",
        "Mean end-to-end latency (s)",
        "Mean request latency per 10-slot window",
        args.slot_window,
        args.max_slot,
    )
    plot_curve(
        energy_path,
        rows,
        ablations,
        "total_energy_j",
        "Mean energy (J)",
        "Mean request energy per 10-slot window",
        args.slot_window,
        args.max_slot,
    )
    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(f"{latency_path.resolve()}\n{energy_path.resolve()}\n", encoding="utf-8")
    print(latency_path)
    print(energy_path)


if __name__ == "__main__":
    main()
