from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ..ablation_names import (
    ABLATION_GROUP_ABLATIONS,
    COMPARISON_GROUP_ABLATIONS,
    canonical_ablation_names,
    canonical_ablation_name,
    canonicalize_ablation_row,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "Simulation" / "test_outputs" / "ablation_experiments"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "selected_seed_extremes" / "filtered_input"
DEFAULT_ABLATIONS = " ".join(
    dict.fromkeys((*ABLATION_GROUP_ABLATIONS, *COMPARISON_GROUP_ABLATIONS))
)

POSITIVE_COLUMNS = (
    "task_completion_rate",
    "success_rate",
    "deadline_acceptance_rate",
    "delay_margin_jain_fairness",
)

NEGATIVE_COLUMNS = (
    "failure_count",
    "p95_end_to_end_delay_s",
    "average_end_to_end_delay_s",
    "average_energy_j",
    "average_communication_delay_s",
    "average_slot_crossings",
    "delay_margin_mean",
    "delay_margin_max",
)

METRIC_FILES = (
    ("slot_metrics_by_seed.csv", "all_ablation_slot_metrics.csv"),
    ("request_metrics_by_seed.csv", "all_ablation_request_metrics.csv"),
    ("request_hop_metrics_by_seed.csv", "all_ablation_request_hop_metrics.csv"),
    ("cycle_metrics_by_seed.csv", "all_ablation_cycle_metrics.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a filtered plotting input directory that keeps the best seed "
            "for ELARA and the worst seed for every other ablation."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablations", default=DEFAULT_ABLATIONS)
    parser.add_argument(
        "--target-ablation",
        default="ELARA",
        help="Ablation whose best seed is selected. All other ablations use their worst seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [canonicalize_ablation_row(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finite_float(value) -> float | None:
    if value in ("", None, "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_value(row: dict, column: str, positive: bool) -> float:
    value = finite_float(row.get(column, ""))
    if value is None:
        return -math.inf
    return value if positive else -value


def quality_tuple(row: dict) -> tuple[float, ...]:
    return tuple(
        metric_value(row, column, True) for column in POSITIVE_COLUMNS
    ) + tuple(metric_value(row, column, False) for column in NEGATIVE_COLUMNS)


def rows_for_variant(input_dir: Path, ablation: str, filename: str) -> list[dict]:
    rows = read_rows(input_dir / ablation / filename)
    if rows:
        return rows
    seed_filename = filename.replace("_by_seed", "")
    rows = []
    for seed_dir in sorted((input_dir / ablation).glob("seed_*")):
        for row in read_rows(seed_dir / seed_filename):
            row["ablation"] = row.get("ablation") or ablation
            rows.append(canonicalize_ablation_row(row))
    return rows


def request_cycle_rows(input_dir: Path, ablation: str) -> list[dict]:
    rows = rows_for_variant(input_dir, ablation, "cycle_metrics_by_seed.csv")
    if rows:
        return rows
    return summarize_request_rows(
        rows_for_variant(input_dir, ablation, "request_metrics_by_seed.csv"),
        ablation,
    )


def summarize_request_rows(rows: list[dict], ablation: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        seed = str(row.get("seed", ""))
        if not seed:
            continue
        grouped.setdefault(seed, []).append(row)

    summaries = []
    for seed, seed_rows in grouped.items():
        delays = finite_values(row.get("total_delay_s") for row in seed_rows)
        energies = finite_values(row.get("total_energy_j") for row in seed_rows)
        feasible = sum(1 for row in seed_rows if str(row.get("feasible", "")).lower() == "true")
        accepted = sum(
            1 for row in seed_rows
            if str(row.get("deadline_accepted", "")).lower() == "true"
        )
        total = len(seed_rows)
        summaries.append(
            {
                "ablation": ablation,
                "seed": seed,
                "request_count": total,
                "success_rate": feasible / total if total else "",
                "task_completion_rate": feasible / total if total else "",
                "deadline_acceptance_rate": accepted / total if total else "",
                "failure_count": total - feasible,
                "average_end_to_end_delay_s": mean(delays),
                "p95_end_to_end_delay_s": percentile(delays, 95),
                "average_energy_j": mean(energies),
            }
        )
    return summaries


def finite_values(values) -> list[float]:
    result = []
    for value in values:
        number = finite_float(value)
        if number is not None:
            result.append(number)
    return result


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((pct / 100.0) * (len(ordered) - 1)))),
    )
    return ordered[index]


def select_seed(input_dir: Path, ablation: str, target_ablation: str) -> dict:
    rows = request_cycle_rows(input_dir, ablation)
    if not rows:
        raise SystemExit(f"No cycle/request metrics found for ablation={ablation} in {input_dir}")

    candidates = [row for row in rows if str(row.get("seed", ""))]
    if not candidates:
        raise SystemExit(f"No seed column found for ablation={ablation} in {input_dir}")

    use_best = ablation == target_ablation
    selected = max(candidates, key=quality_tuple) if use_best else min(candidates, key=quality_tuple)
    return {
        "ablation": ablation,
        "selection": "best" if use_best else "worst",
        "seed": str(selected.get("seed", "")),
        "quality_tuple": json.dumps(quality_tuple(selected)),
        **{
            column: selected.get(column, "")
            for column in (*POSITIVE_COLUMNS, *NEGATIVE_COLUMNS)
            if column in selected
        },
    }


def filter_rows_for_seed(rows: list[dict], ablation: str, seed: str) -> list[dict]:
    selected = []
    for row in rows:
        row["ablation"] = row.get("ablation") or ablation
        canonicalize_ablation_row(row)
        if str(row.get("seed", "")) == seed:
            selected.append(row)
    return selected


def build_filtered_input(
    input_dir: Path,
    output_dir: Path,
    ablations: list[str],
    target_ablation: str,
) -> list[dict]:
    selections = [
        select_seed(input_dir, ablation, target_ablation)
        for ablation in ablations
    ]

    all_rows_by_file = {root_name: [] for _, root_name in METRIC_FILES}
    for selection in selections:
        ablation = selection["ablation"]
        seed = selection["seed"]
        for variant_name, root_name in METRIC_FILES:
            rows = rows_for_variant(input_dir, ablation, variant_name)
            filtered = filter_rows_for_seed(rows, ablation, seed)
            write_rows(output_dir / ablation / variant_name, filtered)
            all_rows_by_file[root_name].extend(filtered)

    for root_name, rows in all_rows_by_file.items():
        write_rows(output_dir / root_name, rows)

    write_rows(output_dir / "selected_seed_manifest.csv", selections)
    (output_dir / "selected_seed_manifest.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selections


def main() -> None:
    args = parse_args()
    ablations = canonical_ablation_names(args.ablations)
    target_ablation = canonical_ablation_name(args.target_ablation)
    if not ablations:
        raise SystemExit("At least one ablation must be provided.")
    if target_ablation not in ablations:
        raise SystemExit(f"--target-ablation={target_ablation} is not in --ablations.")
    if args.output_dir.exists() and not args.force:
        raise SystemExit(
            f"Output directory already exists: {args.output_dir}. "
            "Pass --force to overwrite CSV files in place."
        )

    selections = build_filtered_input(
        args.input_dir,
        args.output_dir,
        ablations,
        target_ablation,
    )
    print(json.dumps(selections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
