from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from .plot_ablation_experiments import (
    ABLATION_LABELS,
    DEFAULT_INPUT_DIR,
    METRICS,
    aggregate_by_slot,
    finite_values,
    load_cycle_rows,
    load_slot_rows,
    mean,
    numeric_value,
    parse_names,
    plot_box,
    plot_slot_curve,
    plot_summary_bar,
    std,
    summary_means,
    values_by_ablation,
)


DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "core_metric_plots"
CORE_METRICS = [
    "task_completion_rate",
    "average_end_to_end_delay_s",
    "average_energy_j",
    "p95_end_to_end_delay_s",
    "p95_energy_j",
]

METRICS.update(
    {
        "average_energy_j": ("Average energy consumption", "Energy (J)"),
        "p95_energy_j": ("P95 energy consumption", "Energy (J)"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot core ablation metrics only: task completion, average delay, "
            "average energy, P95 delay, and P95 energy."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ablations",
        default="full no_bandit shortest_hop_routing nearest_replica service_pressure sc_nfv fairness_nfv_greedy",
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


def percentile(values: list[float], pct: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    index = min(
        len(values) - 1,
        max(0, int(round((pct / 100.0) * (len(values) - 1)))),
    )
    return values[index]


def group_request_energies(request_rows: list[dict]) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str, str], list[float]]]:
    by_cycle: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_slot: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in request_rows:
        energy = numeric_value(row, "total_energy_j")
        if energy is None:
            continue
        ablation = str(row.get("ablation", ""))
        seed = str(row.get("seed", ""))
        epoch = str(row.get("epoch", ""))
        if not ablation or not seed:
            continue
        by_cycle[(ablation, seed)].append(energy)
        if epoch:
            by_slot[(ablation, seed, epoch)].append(energy)
    return by_cycle, by_slot


def add_missing_p95_energy(
    cycle_rows: list[dict],
    slot_rows: list[dict],
    request_rows: list[dict],
) -> None:
    by_cycle, by_slot = group_request_energies(request_rows)
    for row in cycle_rows:
        if numeric_value(row, "p95_energy_j") is not None:
            continue
        key = (str(row.get("ablation", "")), str(row.get("seed", "")))
        value = percentile(by_cycle.get(key, []), 95)
        if value is not None:
            row["p95_energy_j"] = value

    for row in slot_rows:
        if numeric_value(row, "p95_energy_j") is not None:
            continue
        key = (
            str(row.get("ablation", "")),
            str(row.get("seed", "")),
            str(row.get("epoch", "")),
        )
        value = percentile(by_slot.get(key, []), 95)
        if value is not None:
            row["p95_energy_j"] = value


def write_metric_summary(
    output_dir: Path,
    rows: list[dict],
    ablations: list[str],
) -> None:
    summary_rows = []
    for ablation in ablations:
        ablation_rows = [row for row in rows if row.get("ablation") == ablation]
        item = {
            "ablation": ablation,
            "label": ABLATION_LABELS.get(ablation, ablation),
            "row_count": len(ablation_rows),
            "seeds": " ".join(
                sorted({str(row.get("seed", "")) for row in ablation_rows if row.get("seed", "")})
            ),
        }
        for metric in CORE_METRICS:
            values = finite_values(row.get(metric, "") for row in ablation_rows)
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = std(values)
        summary_rows.append(item)
    write_csv_rows(output_dir / "core_ablation_metric_summary.csv", summary_rows)


def main() -> None:
    args = parse_args()
    ablations = parse_names(args.ablations)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slot_rows = load_slot_rows(args.input_dir)
    cycle_rows = load_cycle_rows(args.input_dir)
    request_rows = read_csv_rows(args.input_dir / "all_ablation_request_metrics.csv")
    add_missing_p95_energy(cycle_rows, slot_rows, request_rows)

    summary_rows = cycle_rows if cycle_rows else slot_rows
    write_metric_summary(args.output_dir, summary_rows, ablations)

    generated: list[Path] = []
    summaries = summary_means(summary_rows, ablations, CORE_METRICS)
    generated.append(
        plot_summary_bar(
            args.output_dir / "core_ablation_metric_summary",
            summaries,
            CORE_METRICS,
            ablations,
            args.format,
        )
    )

    for metric in CORE_METRICS:
        title, ylabel = METRICS[metric]
        series = aggregate_by_slot(slot_rows, ablations, metric, args.window)
        generated.append(
            plot_slot_curve(
                args.output_dir / f"{metric}_by_slot",
                f"{title} by time slot",
                ylabel,
                series,
                args.format,
            )
        )

    for metric in CORE_METRICS:
        title, ylabel = METRICS[metric]
        data = values_by_ablation(summary_rows, ablations, metric)
        generated.append(
            plot_box(
                args.output_dir / f"{metric}_distribution",
                f"{title} distribution by ablation",
                ylabel,
                data,
                ablations,
                args.format,
            )
        )

    manifest = args.output_dir / "plot_manifest.txt"
    manifest.write_text(
        "\n".join(str(path) for path in generated) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} core metric plot files to {args.output_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
