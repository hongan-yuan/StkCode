from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..ablation_names import canonical_ablation_name


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Simulation" / "pics" / "elara_paper"
DEFAULT_PAPER_OUTPUT_DIR = ROOT_DIR / "MyPaper" / "elara_exp"

METHODS = ("ELARA", "SECO", "SC-NFV", "SP-Routing", "ELARA-NB", "ELARA-NR", "ELARA-SH")
DEFAULT_CHAIN_LENGTHS = (5, 10, 15)

COLORS = {
    "ELARA": "#2f6fbb",
    "SECO": "#0f766e",
    "SC-NFV": "#6b7280",
    "SP-Routing": "#0891b2",
    "ELARA-NB": "#d9822b",
    "ELARA-NR": "#2f9e44",
    "ELARA-SH": "#8b5cf6",
}

FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
GRID_LINE_WIDTH = 0.55
AXIS_LINE_WIDTH = 1.0
VIOLIN_ALPHA = 0.72
VIOLIN_LINE_WIDTH = 1.0
MEDIAN_LINE_WIDTH = 2.2
BOX_LINE_WIDTH = 1.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw request-level latency and energy violin distributions."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT_DIR)
    parser.add_argument("--methods", default=" ".join(METHODS))
    parser.add_argument("--chain-lengths", default=" ".join(map(str, DEFAULT_CHAIN_LENGTHS)))
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_methods(value: str) -> list[str]:
    methods: list[str] = []
    seen = set()
    for item in value.replace(",", " ").split():
        method = canonical_ablation_name(item)
        if method and method not in seen:
            methods.append(method)
            seen.add(method)
    if not methods:
        raise SystemExit("At least one method must be provided.")
    return methods


def parse_chain_lengths(value: str) -> set[int]:
    lengths: set[int] = set()
    for item in value.replace(",", " ").split():
        lengths.add(int(item))
    if not lengths:
        raise SystemExit("At least one chain length must be provided.")
    return lengths


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size <= 3:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["ablation"] = canonical_ablation_name(row.get("ablation", ""))
    return [row for row in rows if row]


def number(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "null", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def request_key(row: dict) -> tuple[str, str, str, str, str]:
    slot = row.get("absolute_slot") or row.get("slot_index") or row.get("epoch") or ""
    return (
        str(row.get("seed", "")),
        str(slot),
        str(row.get("request_id", "")),
        str(row.get("template_id", "")),
        str(row.get("chain_length", "")),
    )


def load_request_rows(input_dir: Path, methods: list[str], chain_lengths: set[int]) -> dict[str, dict]:
    rows_by_method: dict[str, dict] = {method: {} for method in methods}
    aggregate_path = input_dir / "all_ablation_request_metrics.csv"
    aggregate_rows = read_rows(aggregate_path)
    if aggregate_rows:
        source_rows = [row for row in aggregate_rows if row.get("ablation") in methods]
    else:
        source_rows = []
        for method in methods:
            for filename in ("request_metrics_by_seed.csv", "window_request_metrics_by_seed.csv"):
                path = input_dir / method / filename
                rows = read_rows(path)
                if rows:
                    source_rows.extend(rows)
                    break
            else:
                raise SystemExit(f"No request rows found for {method} under {input_dir / method}.")

    for row in source_rows:
        method = row.get("ablation")
        chain_length = number(row, "chain_length")
        delay = number(row, "total_delay_s")
        energy = number(row, "total_energy_j")
        if method not in methods or chain_length is None or int(chain_length) not in chain_lengths:
            continue
        if delay is None or energy is None:
            continue
        rows_by_method[method][request_key(row)] = {
            "latency": delay,
            "energy_1e3": energy / 1000.0,
        }
    missing_methods = [method for method in methods if not rows_by_method[method]]
    if missing_methods:
        raise SystemExit(f"No usable request rows found for methods: {missing_methods}")
    return rows_by_method


def aligned_metric_values(
    rows_by_method: dict[str, dict],
    methods: list[str],
    metric: str,
) -> tuple[list[np.ndarray], int]:
    method_key_sets = [set(rows_by_method[method]) for method in methods]
    common_keys = set.intersection(*method_key_sets) if method_key_sets else set()
    if not common_keys:
        raise SystemExit("No common request instances exist across the selected methods.")
    ordered_keys = sorted(common_keys)
    values = [
        np.array([rows_by_method[method][key][metric] for key in ordered_keys], dtype=float)
        for method in methods
    ]
    return values, len(ordered_keys)


def style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)


def save_violin_figure(
    output_path: Path,
    values: list[np.ndarray],
    methods: list[str],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(13.6, 5.2))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.96, bottom=0.23)

    positions = np.arange(1, len(methods) + 1)
    parts = ax.violinplot(
        values,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, method in zip(parts["bodies"], methods):
        body.set_facecolor(COLORS.get(method, "#4b5563"))
        body.set_edgecolor("#111827")
        body.set_alpha(VIOLIN_ALPHA)
        body.set_linewidth(VIOLIN_LINE_WIDTH)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": MEDIAN_LINE_WIDTH},
        boxprops={"facecolor": "white", "edgecolor": "#111827", "linewidth": BOX_LINE_WIDTH},
        whiskerprops={"color": "#111827", "linewidth": BOX_LINE_WIDTH},
        capprops={"color": "#111827", "linewidth": BOX_LINE_WIDTH},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.72)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_xlim(0.35, len(methods) + 0.65)
    ax.set_xlabel("Method")
    style_axis(ax, ylabel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def copy_outputs(output_dir: Path, paper_output_dir: Path, stems: list[str]) -> None:
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for suffix in (".png", ".pdf"):
            source = output_dir / f"{stem}{suffix}"
            if source.exists():
                shutil.copy2(source, paper_output_dir / source.name)


def main() -> None:
    args = parse_args()
    configure_style()
    methods = parse_methods(args.methods)
    chain_lengths = parse_chain_lengths(args.chain_lengths)

    rows_by_method = load_request_rows(args.input_dir, methods, chain_lengths)
    latency_values, sample_size = aligned_metric_values(rows_by_method, methods, "latency")
    energy_values, _ = aligned_metric_values(rows_by_method, methods, "energy_1e3")

    latency_path = args.output_dir / "elara_latency_distribution.png"
    energy_path = args.output_dir / "elara_energy_distribution.png"
    save_violin_figure(latency_path, latency_values, methods, "End-to-end latency (s)")
    save_violin_figure(energy_path, energy_values, methods, "Energy (x1k J)")

    if not args.no_paper_copy:
        copy_outputs(args.output_dir, args.paper_output_dir, [
            "elara_latency_distribution",
            "elara_energy_distribution",
        ])

    print(f"Aligned request samples: {sample_size}")
    for path in (
        latency_path,
        latency_path.with_suffix(".pdf"),
        energy_path,
        energy_path.with_suffix(".pdf"),
    ):
        print(path)


if __name__ == "__main__":
    main()
