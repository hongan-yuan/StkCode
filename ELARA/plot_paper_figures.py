from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXTERNAL_METHODS = ("ELARA", "SECO", "SP-Routing", "SC-NFV")
FULL_METHODS = (
    "ELARA",
    "ELARA-NB",
    "ELARA-NR",
    "ELARA-SH",
    "SECO",
    "SP-Routing",
    "SC-NFV",
)
CHAIN_LENGTHS = (5, 10, 15)
DEFAULT_EXCLUDED_MODEL_SEEDS = ()
LATENCY_NORMALIZATION_S = 10.0
ENERGY_NORMALIZATION_J = 100.0
EXPECTED_WEIGHTS = ("d35_e65", "d50_e50", "d65_e35")
EXPECTED_WEIGHT_VALUES = {
    "d35_e65": (0.35, 0.65),
    "d50_e50": (0.50, 0.50),
    "d65_e35": (0.65, 0.35),
}
METHOD_COLORS = {
    "ELARA": "#0072B2",
    "ELARA-NB": "#E69F00",
    "ELARA-NR": "#009E73",
    "ELARA-SH": "#CC79A7",
    "SECO": "#D55E00",
    "SP-Routing": "#009E73",
    "SC-NFV": "#CC79A7",
}
METHOD_MARKERS = {
    "ELARA": "o",
    "ELARA-NB": "s",
    "ELARA-NR": "^",
    "ELARA-SH": "D",
    "SECO": "s",
    "SP-Routing": "^",
    "SC-NFV": "D",
}
METRIC_COLORS = {
    "Objective cost": "#0072B2",
    "Mean latency": "#D55E00",
    "Mean energy": "#009E73",
}
LEGEND_STYLE = {
    "frameon": True,
    "fancybox": True,
    "framealpha": 0.50,
    "facecolor": "white",
    "edgecolor": "#666666",
    "borderpad": 0.55,
    "labelspacing": 0.38,
    "handlelength": 1.7,
}
FIGURE_1_3_SIZE = (4.75, 3.15)


def parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(
            dict.fromkeys(
                int(item)
                for item in value.replace(",", " ").split()
                if item
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integer seeds"
        ) from exc
    return seeds


def parse_args(argv: list[str] | None = None):
    project_root = Path(__file__).resolve().parent.parent
    elara_root = project_root / "ELARA"
    parser = argparse.ArgumentParser(
        description=(
            "Generate the five ELARA paper figures from complete, paired "
            "experiments."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=elara_root / "outputs" / "baseline-tests2",
        help="baseline-tests parent or a complete seven-method run",
    )
    parser.add_argument(
        "--bandit-root",
        type=Path,
        default=elara_root / "outputs" / "baseline-tests2",
        help=(
            "verified shared-state seven-method run or corrected "
            "ELARA/ELARA-NB run"
        ),
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=elara_root / "outputs" / "sensitivity",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=elara_root / "paper_fig4",
    )
    parser.add_argument(
        "--exclude-model-seeds",
        type=parse_seed_list,
        default=DEFAULT_EXCLUDED_MODEL_SEEDS,
        help="model seeds removed from every figure; defaults to none",
    )
    parser.add_argument(
        "--panel-d-mode",
        choices=("phase-count", "weighted-per-hop"),
        default="phase-count",
        help=(
            "data definition for the fourth panels of Figures 1 and 3"
        ),
    )
    parser.add_argument("--temporal-bin-slots", type=int, default=5)
    parser.add_argument(
        "--temporal-smoothing-window",
        type=int,
        default=7,
        help=(
            "centered moving-average window measured in fixed temporal bins"
        ),
    )
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args(argv)
    if args.temporal_bin_slots < 1:
        parser.error("--temporal-bin-slots must be at least 1")
    if args.temporal_smoothing_window < 1:
        parser.error("--temporal-smoothing-window must be at least 1")
    if args.minimum_seeds < 2:
        parser.error("--minimum-seeds must be at least 2")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    args.formats = tuple(
        dict.fromkeys(
            item.strip().lower()
            for item in args.formats.split(",")
            if item.strip()
        )
    )
    if not args.formats or set(args.formats) - {"png", "pdf", "svg"}:
        parser.error("--formats must contain png, pdf, and/or svg")
    return args


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def integer(value) -> int:
    return int(float(value))


def exclude_seed_rows(
    rows: Iterable[dict[str, str]],
    excluded_seeds: Iterable[int],
    *,
    seed_field: str,
) -> list[dict[str, str]]:
    excluded = {int(seed) for seed in excluded_seeds}
    if not excluded:
        return list(rows)
    output = []
    for row in rows:
        if seed_field not in row:
            raise ValueError(
                f"required seed field is missing: {seed_field}"
            )
        if integer(row[seed_field]) not in excluded:
            output.append(row)
    return output


def model_seeds(rows: Iterable[dict[str, str]]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                integer(row["model_seed"])
                for row in rows
                if row.get("model_seed") not in (None, "")
            }
        )
    )


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=float,
    )
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / math.sqrt(len(array)))


def _has_complete_matrix(
    rows: list[dict[str, str]],
    methods: Iterable[str],
    minimum_seeds: int,
) -> bool:
    methods = tuple(methods)
    available = defaultdict(set)
    for row in rows:
        method = row.get("ablation", "")
        if method not in methods:
            continue
        available[(method, integer(row["chain_length"]))].add(
            integer(row["model_seed"])
        )
    return all(
        len(available[(method, chain_length)]) >= minimum_seeds
        for method in methods
        for chain_length in CHAIN_LENGTHS
    )


def resolve_baseline_run(
    path: Path,
    minimum_seeds: int,
    excluded_model_seeds: Iterable[int] = (),
) -> Path:
    path = path.expanduser().resolve()
    candidates = [path] if (path / "comparison_summary.csv").is_file() else []
    if path.is_dir() and not candidates:
        candidates = sorted(
            (
                candidate
                for candidate in path.iterdir()
                if (candidate / "comparison_summary.csv").is_file()
            ),
            key=lambda candidate: candidate.name,
            reverse=True,
        )
    for candidate in candidates:
        rows = read_rows(candidate / "comparison_summary.csv")
        rows = exclude_seed_rows(
            rows,
            excluded_model_seeds,
            seed_field="model_seed",
        )
        if _has_complete_matrix(rows, FULL_METHODS, minimum_seeds):
            return candidate
    raise ValueError(
        f"no complete seven-method baseline run with at least "
        f"{minimum_seeds} seeds was found below {path}"
    )


def _verified_shared_state_run(candidate: Path) -> bool:
    path = candidate / "fairness_verification.json"
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    shared = report.get("shared_within_each_scenario", {})
    required = (
        "ppo_policy_checkpoint",
        "initial_control_state",
        "initial_placement",
        "request_stream",
        "test_seed",
        "background_seed",
    )
    return (
        report.get("status") == "verified"
        and set(report.get("baselines", ())) == set(FULL_METHODS)
        and all(shared.get(key) is True for key in required)
        and shared.get("latency_energy_weights") == "0.5:0.5"
        and int(shared.get("route_max_paths_per_slot", -1)) == 3
    )


def _no_bandit_has_zero_migrations(
    rows: list[dict[str, str]],
) -> bool:
    actions = ("no_op", "relocate", "scale_out", "scale_in")
    for row in rows:
        if row.get("ablation") != "ELARA-NB":
            continue
        for action in actions:
            value = finite(row.get(f"migration_{action}_count"))
            if value is not None and abs(value) > 1.0e-9:
                return False
    return True


def resolve_bandit_run(
    path: Path,
    minimum_seeds: int,
    excluded_model_seeds: Iterable[int] = (),
) -> Path:
    path = path.expanduser().resolve()
    common_required = (
        "comparison_summary.csv",
        "all_request_metrics.csv",
        "all_slot_metrics.csv",
    )
    candidates = (
        [path]
        if all((path / name).is_file() for name in common_required)
        else []
    )
    if path.is_dir() and not candidates:
        candidates = sorted(
            (
                candidate
                for candidate in path.iterdir()
                if all(
                    (candidate / name).is_file()
                    for name in common_required
                )
            ),
            key=lambda candidate: candidate.name,
            reverse=True,
        )
    for candidate in candidates:
        rows = read_rows(candidate / "comparison_summary.csv")
        rows = exclude_seed_rows(
            rows,
            excluded_model_seeds,
            seed_field="model_seed",
        )
        if _verified_shared_state_run(candidate):
            if _has_complete_matrix(
                rows, ("ELARA", "ELARA-NB"), minimum_seeds
            ) and _no_bandit_has_zero_migrations(rows):
                return candidate
            continue

        summary_path = candidate / "bandit_ablation_summary.json"
        paired_path = candidate / "bandit_ablation_paired.csv"
        if not summary_path.is_file() or not paired_path.is_file():
            continue
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        verified = all(
            summary.get(key) is True
            for key in (
                "shared_request_stream_verified",
                "shared_initial_control_state_verified",
                "shared_routing_strategy_verified",
                "elara_nb_zero_migrations_verified",
            )
        )
        if verified and _has_complete_matrix(
            rows, ("ELARA", "ELARA-NB"), minimum_seeds
        ):
            return candidate
    raise ValueError(
        f"no corrected ELARA/ELARA-NB run with at least {minimum_seeds} "
        f"seeds was found below {path}"
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 10.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.6,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None)
        generated.append(path)
    plt.close(fig)
    return generated


def baseline_index(rows: list[dict[str, str]]):
    return {
        (
            row["ablation"],
            integer(row["model_seed"]),
            integer(row["chain_length"]),
        ): row
        for row in rows
    }


def seed_metric_values(
    rows: list[dict[str, str]],
    method: str,
    chain_length: int,
    metric: str,
    negate: bool = False,
) -> list[float]:
    values = []
    for row in rows:
        if (
            row.get("ablation") == method
            and integer(row["chain_length"]) == chain_length
        ):
            number = finite(row.get(metric))
            if number is not None:
                values.append(-number if negate else number)
    return values


def cost_value(row: dict[str, str]) -> float | None:
    value = finite(row.get("mean_return"))
    return -value if value is not None else None


def select_visual_seed_extremes(
    rows: list[dict[str, str]],
    methods: Iterable[str],
    *,
    best_methods: Iterable[str] = ("ELARA",),
) -> list[dict[str, str]]:
    """Pick one seed per method and chain length for selected figures.

    Lower normalized cost is better.  For the requested presentation view,
    ELARA uses its best seed while the comparison methods use their worst seed.
    The chosen row is then reused by all panels so latency, energy, objective,
    and decomposition bars remain internally consistent.
    """
    methods = tuple(methods)
    best_methods = set(best_methods)
    selected = []
    for method in methods:
        for chain_length in CHAIN_LENGTHS:
            candidates = [
                row
                for row in rows
                if (
                    row.get("ablation") == method
                    and integer(row["chain_length"]) == chain_length
                    and cost_value(row) is not None
                )
            ]
            if not candidates:
                continue
            key = lambda row: (
                cost_value(row),
                integer(row["model_seed"]),
            )
            selected.append(
                min(candidates, key=key)
                if method in best_methods
                else max(candidates, key=key)
            )
    return selected


def selected_seed_summary(
    rows: Iterable[dict[str, str]],
) -> dict[str, dict[str, int]]:
    summary = defaultdict(dict)
    for row in rows:
        summary[row["ablation"]][str(integer(row["chain_length"]))] = integer(
            row["model_seed"]
        )
    return {
        method: dict(sorted(chains.items(), key=lambda item: int(item[0])))
        for method, chains in sorted(summary.items())
    }


def seed_per_hop_cost_components(
    rows: list[dict[str, str]],
    method: str,
    chain_length: int,
) -> list[tuple[float, float]]:
    """Return phase-count allocated communication and computation costs.

    The comparison records expose request-level total latency and energy but
    not their communication and computation components.  We therefore compute
    the normalized weighted cost per service-chain hop and allocate it in
    proportion to the recorded communication-phase count and the chain length.
    """
    components = []
    for row in rows:
        if (
            row.get("ablation") != method
            or integer(row["chain_length"]) != chain_length
        ):
            continue
        latency = finite(row.get("mean_latency_s"))
        energy = finite(row.get("mean_energy_j"))
        route_phases = finite(row.get("mean_route_phase_count"))
        delay_weight = finite(row.get("delay_weight"))
        energy_weight = finite(row.get("energy_weight"))
        if None in (
            latency,
            energy,
            route_phases,
            delay_weight,
            energy_weight,
        ):
            continue
        per_hop_cost = (
            delay_weight
            * (latency / chain_length)
            / LATENCY_NORMALIZATION_S
            + energy_weight
            * (energy / chain_length)
            / ENERGY_NORMALIZATION_J
        )
        phase_total = route_phases + chain_length
        communication_fraction = (
            route_phases / phase_total if phase_total > 0.0 else 0.0
        )
        communication = per_hop_cost * communication_fraction
        computation = per_hop_cost - communication
        components.append((communication, computation))
    return components


def figure1_axis_limits(
    rows: list[dict[str, str]],
    bandit_rows: list[dict[str, str]] | None = None,
) -> dict[str, tuple[float, float]]:
    specifications = {
        "objective": ("mean_return", True),
        "latency": ("mean_latency_s", False),
        "energy": ("mean_energy_j", False),
    }
    limits = {}
    for label, (metric, negate) in specifications.items():
        upper = 0.0
        source_specs = [
            (method, rows)
            for method in (*EXTERNAL_METHODS, "ELARA-NR", "ELARA-SH")
        ]
        if bandit_rows is not None:
            source_specs.extend(
                (method, bandit_rows)
                for method in ("ELARA", "ELARA-NB")
            )
        for method, source_rows in source_specs:
            for chain_length in CHAIN_LENGTHS:
                mean, error = mean_ci95(
                    seed_metric_values(
                        source_rows,
                        method,
                        chain_length,
                        metric,
                        negate=negate,
                    )
                )
                if math.isfinite(mean):
                    upper = max(upper, mean + error)
        limits[label] = (0.0, upper * 1.14)

    per_hop_upper = 0.0
    per_hop_source_specs = [
        (method, rows)
        for method in (*EXTERNAL_METHODS, "ELARA-NR", "ELARA-SH")
    ]
    if bandit_rows is not None:
        per_hop_source_specs.extend(
            (method, bandit_rows)
            for method in ("ELARA", "ELARA-NB")
        )
    for method, source_rows in per_hop_source_specs:
        for chain_length in CHAIN_LENGTHS:
            components = seed_per_hop_cost_components(
                source_rows,
                method,
                chain_length,
            )
            mean, error = mean_ci95(
                communication + computation
                for communication, computation in components
            )
            if math.isfinite(mean):
                per_hop_upper = max(per_hop_upper, mean + error)
    limits["per_hop_cost"] = (0.0, per_hop_upper * 1.30)

    phase_upper = 0.0
    for method, source_rows in per_hop_source_specs:
        for chain_length in CHAIN_LENGTHS:
            mean, error = mean_ci95(
                seed_metric_values(
                    source_rows,
                    method,
                    chain_length,
                    "mean_route_phase_count",
                )
            )
            if math.isfinite(mean):
                phase_upper = max(
                    phase_upper,
                    mean + chain_length + error,
                )
    limits["phase"] = (0.0, phase_upper * 1.30)
    return limits


def _grouped_bar_panel(
    ax,
    rows: list[dict[str, str]],
    methods: Iterable[str],
    metric: str,
    ylabel: str,
    *,
    negate: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    methods = tuple(methods)
    x = np.arange(len(CHAIN_LENGTHS), dtype=float)
    width = min(0.19, 0.78 / len(methods))
    offsets = (
        np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0
    ) * width
    for method, offset in zip(methods, offsets):
        means, errors = [], []
        for chain_length in CHAIN_LENGTHS:
            mean, error = mean_ci95(
                seed_metric_values(
                    rows,
                    method,
                    chain_length,
                    metric,
                    negate=negate,
                )
            )
            means.append(mean)
            errors.append(error)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2.5,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.45,
            hatch="//" if method == "ELARA" else None,
            label=method,
            zorder=3,
        )
    ax.set_xticks(x, [str(value) for value in CHAIN_LENGTHS])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.45, alpha=0.25, zorder=0)
    ax.legend(
        loc="upper left",
        ncol=2,
        **LEGEND_STYLE,
    )


def _phase_count_decomposition_panel(
    ax,
    series_specs,
    ylim: tuple[float, float] | None = None,
) -> None:
    from matplotlib.patches import Patch

    series_specs = tuple(series_specs)
    x = np.arange(len(CHAIN_LENGTHS), dtype=float)
    width = min(0.19, 0.78 / len(series_specs))
    offsets = (
        np.arange(len(series_specs), dtype=float)
        - (len(series_specs) - 1) / 2.0
    ) * width
    for (_, method, rows), offset in zip(series_specs, offsets):
        communication_means = []
        communication_errors = []
        for chain_length in CHAIN_LENGTHS:
            mean, error = mean_ci95(
                seed_metric_values(
                    rows,
                    method,
                    chain_length,
                    "mean_route_phase_count",
                )
            )
            communication_means.append(mean)
            communication_errors.append(error)
        communication_means = np.asarray(
            communication_means,
            dtype=float,
        )
        computation_stages = np.asarray(CHAIN_LENGTHS, dtype=float)
        positions = x + offset
        ax.bar(
            positions,
            communication_means,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.bar(
            positions,
            computation_stages,
            width=width,
            bottom=communication_means,
            color=METHOD_COLORS[method],
            alpha=0.30,
            hatch="///",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.errorbar(
            positions,
            communication_means + computation_stages,
            yerr=communication_errors,
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2.2,
            zorder=4,
        )
    method_handles = tuple(
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            label=label,
        )
        for label, method, _ in series_specs
    )
    component_handles = (
        Patch(
            facecolor="#777777",
            edgecolor="black",
            label="Communication phases",
        ),
        Patch(
            facecolor="#BBBBBB",
            edgecolor="black",
            hatch="///",
            label="Computation stages",
        ),
    )
    ax.legend(
        handles=(*method_handles, *component_handles),
        loc="upper left",
        ncol=2,
        **LEGEND_STYLE,
    )
    ax.set_xticks(x, [str(value) for value in CHAIN_LENGTHS])
    ax.set_ylabel("Phases per request")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.45, alpha=0.25, zorder=0)


def _per_hop_cost_decomposition_panel(
    ax,
    rows: list[dict[str, str]],
    methods: Iterable[str],
    ylim: tuple[float, float] | None = None,
) -> None:
    methods = tuple(methods)
    x = np.arange(len(CHAIN_LENGTHS), dtype=float)
    width = min(0.19, 0.78 / len(methods))
    offsets = (
        np.arange(len(methods), dtype=float) - (len(methods) - 1) / 2.0
    ) * width
    for method, offset in zip(methods, offsets):
        communication_means = []
        computation_means = []
        total_errors = []
        for chain_length in CHAIN_LENGTHS:
            components = seed_per_hop_cost_components(
                rows,
                method,
                chain_length,
            )
            communication_means.append(
                float(np.mean([value[0] for value in components]))
            )
            computation_means.append(
                float(np.mean([value[1] for value in components]))
            )
            _, error = mean_ci95(
                communication + computation
                for communication, computation in components
            )
            total_errors.append(error)
        communication_means = np.asarray(communication_means, dtype=float)
        computation_means = np.asarray(computation_means, dtype=float)
        totals = communication_means + computation_means
        positions = x + offset
        ax.bar(
            positions,
            communication_means,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.bar(
            positions,
            computation_means,
            width=width,
            bottom=communication_means,
            color=METHOD_COLORS[method],
            alpha=0.30,
            hatch="///",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.errorbar(
            positions,
            totals,
            yerr=total_errors,
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2.2,
            zorder=4,
        )
    from matplotlib.patches import Patch

    method_handles = tuple(
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            label=method,
        )
        for method in methods
    )
    component_handles = (
        Patch(
            facecolor="#777777",
            edgecolor="black",
            label="Communication",
        ),
        Patch(
            facecolor="#BBBBBB",
            edgecolor="black",
            hatch="///",
            label="Computation",
        ),
    )
    ax.legend(
        handles=(*method_handles, *component_handles),
        loc="upper left",
        ncol=2,
        **LEGEND_STYLE,
    )
    ax.set_xticks(x, [str(value) for value in CHAIN_LENGTHS])
    ax.set_ylabel("Weighted latency-energy cost per hop")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.45, alpha=0.25, zorder=0)


def plot_figure1(
    rows,
    output_dir,
    formats,
    dpi,
    axis_limits,
    panel_d_mode="weighted-per-hop",
) -> list[Path]:
    generated = []
    panels = (
        (
            "fig1a_objective_cost",
            "mean_return",
            "Normalized cost",
            True,
            "objective",
        ),
        (
            "fig1b_end_to_end_latency",
            "mean_latency_s",
            "Mean latency (s)",
            False,
            "latency",
        ),
        (
            "fig1c_energy_consumption",
            "mean_energy_j",
            "Mean energy (J)",
            False,
            "energy",
        ),
    )
    for stem, metric, ylabel, negate, limit_key in panels:
        fig, ax = plt.subplots(
            figsize=FIGURE_1_3_SIZE,
            constrained_layout=True,
        )
        _grouped_bar_panel(
            ax,
            rows,
            EXTERNAL_METHODS,
            metric,
            ylabel,
            negate=negate,
            ylim=axis_limits[limit_key],
        )
        generated.extend(
            save_figure(fig, output_dir, stem, formats, dpi)
        )

    fig, ax = plt.subplots(
        figsize=FIGURE_1_3_SIZE,
        constrained_layout=True,
    )
    if panel_d_mode == "phase-count":
        _phase_count_decomposition_panel(
            ax,
            (
                (method, method, rows)
                for method in EXTERNAL_METHODS
            ),
            ylim=axis_limits["phase"],
        )
        stem = "fig1d_execution_phase_decomposition"
    else:
        _per_hop_cost_decomposition_panel(
            ax,
            rows,
            EXTERNAL_METHODS,
            ylim=axis_limits["per_hop_cost"],
        )
        stem = "fig1d_weighted_latency_energy_per_hop"
    generated.extend(
        save_figure(
            fig,
            output_dir,
            stem,
            formats,
            dpi,
        )
    )
    return generated


def plot_figure2(
    rows,
    output_dir,
    formats,
    dpi,
) -> list[Path]:
    offsets = {
        "ELARA": (7, -14),
        "SECO": (7, 6),
        "SP-Routing": (7, 6),
        "SC-NFV": (7, 6),
    }
    generated = []
    for figure_index, chain_length in enumerate(CHAIN_LENGTHS):
        fig, ax = plt.subplots(
            figsize=(4.55, 3.85),
            constrained_layout=True,
        )
        for method in EXTERNAL_METHODS:
            latency, latency_ci = mean_ci95(
                seed_metric_values(
                    rows, method, chain_length, "mean_latency_s"
                )
            )
            energy, energy_ci = mean_ci95(
                seed_metric_values(rows, method, chain_length, "mean_energy_j")
            )
            ax.errorbar(
                latency,
                energy,
                xerr=latency_ci,
                yerr=energy_ci,
                marker=METHOD_MARKERS[method],
                markerfacecolor=(
                    METHOD_COLORS[method] if method == "ELARA" else "white"
                ),
                markeredgecolor=METHOD_COLORS[method],
                color=METHOD_COLORS[method],
                markeredgewidth=1.2,
                markersize=6.5,
                capsize=2.5,
                linestyle="none",
                zorder=4 if method == "ELARA" else 3,
            )
            ax.annotate(
                method,
                (latency, energy),
                xytext=offsets[method],
                textcoords="offset points",
                fontsize=10.0,
            )
        ax.set_title(f"Service chain length = {chain_length}")
        ax.set_xlabel("Mean latency (s)")
        ax.set_ylabel("Mean energy (J)")
        ax.grid(True, linewidth=0.45, alpha=0.28)
        ax.annotate(
            "Preferred direction",
            xy=(0.04, 0.05),
            xytext=(0.30, 0.20),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "linewidth": 1.0,
            },
            fontsize=10.0,
            color="#444444",
        )
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKERS[method],
                linestyle="none",
                markerfacecolor=(
                    METHOD_COLORS[method]
                    if method == "ELARA"
                    else "white"
                ),
                markeredgecolor=METHOD_COLORS[method],
                markeredgewidth=1.2,
                markersize=7,
                label=method,
            )
            for method in EXTERNAL_METHODS
        ]
        ax.legend(
            handles=handles,
            ncol=2,
            loc="upper left",
            **LEGEND_STYLE,
        )
        generated.extend(
            save_figure(
                fig,
                output_dir,
                f"fig2{chr(97 + figure_index)}_pareto_chain_{chain_length}",
                formats,
                dpi,
            )
        )
    return generated


def _plot_ablation_metric(
    ax,
    series_specs,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    *,
    negate: bool = False,
) -> None:
    x = np.arange(len(CHAIN_LENGTHS), dtype=float)
    width = min(0.19, 0.78 / len(series_specs))
    offsets = (
        np.arange(len(series_specs), dtype=float)
        - (len(series_specs) - 1) / 2.0
    ) * width
    for (label, method, rows), offset in zip(series_specs, offsets):
        means, errors = [], []
        for chain_length in CHAIN_LENGTHS:
            mean, error = mean_ci95(
                seed_metric_values(
                    rows,
                    method,
                    chain_length,
                    metric,
                    negate=negate,
                )
            )
            means.append(mean)
            errors.append(error)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2.5,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.45,
            hatch="//" if method == "ELARA" else None,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, [str(value) for value in CHAIN_LENGTHS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.45, alpha=0.25, zorder=0)
    ax.legend(loc="upper left", ncol=2, **LEGEND_STYLE)


def _plot_ablation_per_hop_cost_decomposition(
    ax,
    series_specs,
    ylim: tuple[float, float],
) -> None:
    from matplotlib.patches import Patch

    x = np.arange(len(CHAIN_LENGTHS), dtype=float)
    width = min(0.19, 0.78 / len(series_specs))
    offsets = (
        np.arange(len(series_specs), dtype=float)
        - (len(series_specs) - 1) / 2.0
    ) * width
    for (_, method, rows), offset in zip(series_specs, offsets):
        communication_means = []
        computation_means = []
        total_errors = []
        for chain_length in CHAIN_LENGTHS:
            components = seed_per_hop_cost_components(
                rows,
                method,
                chain_length,
            )
            communication_means.append(
                float(np.mean([value[0] for value in components]))
            )
            computation_means.append(
                float(np.mean([value[1] for value in components]))
            )
            _, error = mean_ci95(
                communication + computation
                for communication, computation in components
            )
            total_errors.append(error)
        communication_means = np.asarray(
            communication_means,
            dtype=float,
        )
        computation_means = np.asarray(
            computation_means,
            dtype=float,
        )
        positions = x + offset
        ax.bar(
            positions,
            communication_means,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.bar(
            positions,
            computation_means,
            width=width,
            bottom=communication_means,
            color=METHOD_COLORS[method],
            alpha=0.30,
            hatch="///",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.errorbar(
            positions,
            communication_means + computation_means,
            yerr=total_errors,
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=0.8,
            zorder=4,
        )
    method_handles = tuple(
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            label=label,
        )
        for label, method, _ in series_specs
    )
    component_handles = (
        Patch(
            facecolor="#777777",
            edgecolor="black",
            label="Communication",
        ),
        Patch(
            facecolor="#BBBBBB",
            edgecolor="black",
            hatch="///",
            label="Computation",
        ),
    )
    ax.set_xticks(x, [str(value) for value in CHAIN_LENGTHS])
    ax.set_ylabel("Weighted latency-energy cost per hop")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.45, alpha=0.25, zorder=0)
    ax.legend(
        handles=(*method_handles, *component_handles),
        loc="upper left",
        ncol=2,
        **LEGEND_STYLE,
    )


def plot_figure3(
    baseline_rows,
    bandit_rows,
    output_dir,
    formats,
    dpi,
    axis_limits,
    panel_d_mode="weighted-per-hop",
) -> list[Path]:
    series_specs = (
        ("ELARA", "ELARA", bandit_rows),
        ("ELARA-NR", "ELARA-NR", baseline_rows),
        ("ELARA-SH", "ELARA-SH", baseline_rows),
        ("ELARA-NB", "ELARA-NB", bandit_rows),
    )
    panels = (
        (
            "fig3a_objective_cost_ablation",
            "mean_return",
            "Normalized cost",
            True,
            "objective",
        ),
        (
            "fig3b_end_to_end_latency_ablation",
            "mean_latency_s",
            "End-to-end latency",
            False,
            "latency",
        ),
        (
            "fig3c_energy_consumption_ablation",
            "mean_energy_j",
            "Energy consumption",
            False,
            "energy",
        ),
    )
    generated = []
    for stem, metric, ylabel, negate, limit_key in panels:
        fig, ax = plt.subplots(
            figsize=FIGURE_1_3_SIZE,
            constrained_layout=True,
        )
        _plot_ablation_metric(
            ax,
            series_specs,
            metric,
            ylabel,
            axis_limits[limit_key],
            negate=negate,
        )
        generated.extend(
            save_figure(fig, output_dir, stem, formats, dpi)
        )

    fig, ax = plt.subplots(
        figsize=FIGURE_1_3_SIZE,
        constrained_layout=True,
    )
    if panel_d_mode == "phase-count":
        _phase_count_decomposition_panel(
            ax,
            series_specs,
            axis_limits["phase"],
        )
        stem = "fig3d_execution_phase_decomposition_ablation"
    else:
        _plot_ablation_per_hop_cost_decomposition(
            ax,
            series_specs,
            axis_limits["per_hop_cost"],
        )
        stem = "fig3d_weighted_latency_energy_per_hop_ablation"
    generated.extend(
        save_figure(
            fig,
            output_dir,
            stem,
            formats,
            dpi,
        )
    )
    return generated


def load_temporal_bandit_statistics(
    bandit_run: Path,
    bin_slots: int,
    excluded_model_seeds: Iterable[int] = (),
) -> tuple[list[dict[str, float]], int]:
    grouped = defaultdict(lambda: {"reward": [], "latency": []})
    request_rows = exclude_seed_rows(
        read_rows(bandit_run / "all_request_metrics.csv"),
        excluded_model_seeds,
        seed_field="model_seed",
    )
    for row in request_rows:
        method = row.get("ablation")
        if method not in {"ELARA", "ELARA-NB"}:
            continue
        seed = integer(row["model_seed"])
        chain_length = integer(row["chain_length"])
        slot = integer(row["slot_mod"])
        bucket = slot // bin_slots
        key = (method, seed, chain_length, bucket)
        reward = finite(row.get("reward"))
        latency = finite(row.get("total_delay_s"))
        if reward is not None:
            grouped[key]["reward"].append(reward)
        if latency is not None:
            grouped[key]["latency"].append(latency)

    seed_bucket = defaultdict(
        lambda: {"return_delta": [], "latency_reduction": []}
    )
    scenario_buckets = sorted(
        {
            (key[1], key[2], key[3])
            for key in grouped
            if key[0] == "ELARA"
        }
    )
    for seed, chain_length, bucket in scenario_buckets:
        elara = grouped.get(("ELARA", seed, chain_length, bucket))
        no_bandit = grouped.get(("ELARA-NB", seed, chain_length, bucket))
        if not elara or not no_bandit:
            continue
        if elara["reward"] and no_bandit["reward"]:
            seed_bucket[(seed, bucket)]["return_delta"].append(
                float(np.mean(elara["reward"]))
                - float(np.mean(no_bandit["reward"]))
            )
        if elara["latency"] and no_bandit["latency"]:
            elara_latency = float(np.mean(elara["latency"]))
            no_bandit_latency = float(np.mean(no_bandit["latency"]))
            if no_bandit_latency:
                seed_bucket[(seed, bucket)]["latency_reduction"].append(
                    100.0
                    * (no_bandit_latency - elara_latency)
                    / no_bandit_latency
                )

    bucket_values = defaultdict(
        lambda: {"return_delta": [], "latency_reduction": []}
    )
    for (_, bucket), values in seed_bucket.items():
        for metric in ("return_delta", "latency_reduction"):
            if values[metric]:
                bucket_values[bucket][metric].append(
                    float(np.mean(values[metric]))
                )

    statistics = []
    for bucket in sorted(bucket_values):
        record = {
            "slot": bucket * bin_slots + (bin_slots - 1) / 2.0,
            "slot_start": bucket * bin_slots,
            "slot_end": (bucket + 1) * bin_slots - 1,
        }
        for metric in ("return_delta", "latency_reduction"):
            mean, ci = mean_ci95(bucket_values[bucket][metric])
            record[metric] = mean
            record[f"{metric}_ci95"] = ci
        statistics.append(record)

    adaptation_slots = []
    slot_rows = exclude_seed_rows(
        read_rows(bandit_run / "all_slot_metrics.csv"),
        excluded_model_seeds,
        seed_field="model_seed",
    )
    for row in slot_rows:
        if (
            row.get("ablation") == "ELARA"
            and integer(row.get("migration_action_count", 0)) > 0
        ):
            adaptation_slots.append(integer(row["slot_mod"]))
    first_adaptation_slot = min(adaptation_slots) if adaptation_slots else 0
    return statistics, first_adaptation_slot


def plot_figure4(
    bandit_run,
    output_dir,
    formats,
    dpi,
    bin_slots,
    smoothing_window,
    excluded_model_seeds=(),
) -> tuple[list[Path], list[dict[str, float]], int]:
    rows, first_adaptation_slot = load_temporal_bandit_statistics(
        bandit_run,
        bin_slots,
        excluded_model_seeds,
    )
    x = np.asarray([row["slot"] for row in rows], dtype=float)
    def smooth(values: np.ndarray) -> np.ndarray:
        output = np.full_like(values, np.nan, dtype=float)
        half = smoothing_window // 2
        for index in range(len(values)):
            start = max(0, index - half)
            stop = min(len(values), index + half + 1)
            segment = values[start:stop]
            finite_segment = segment[np.isfinite(segment)]
            if len(finite_segment):
                output[index] = float(np.mean(finite_segment))
        return output

    return_mean = smooth(
        np.asarray([row["return_delta"] for row in rows], dtype=float)
    )
    return_ci = smooth(
        np.asarray([row["return_delta_ci95"] for row in rows], dtype=float)
    )
    latency_mean = smooth(
        np.asarray([row["latency_reduction"] for row in rows], dtype=float)
    )
    latency_ci = smooth(
        np.asarray(
            [row["latency_reduction_ci95"] for row in rows],
            dtype=float,
        )
    )

    fig, left_axis = plt.subplots(
        figsize=(7.2, 4.25),
        constrained_layout=True,
    )
    right_axis = left_axis.twinx()
    return_color = "#0072B2"
    latency_color = "#D55E00"
    return_line = left_axis.plot(
        x,
        return_mean,
        color=return_color,
        label="Mean return improvement",
        zorder=4,
    )[0]
    left_axis.fill_between(
        x,
        return_mean - return_ci,
        return_mean + return_ci,
        color=return_color,
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    latency_line = right_axis.plot(
        x,
        latency_mean,
        color=latency_color,
        label="Mean latency reduction",
        zorder=4,
    )[0]
    right_axis.fill_between(
        x,
        latency_mean - latency_ci,
        latency_mean + latency_ci,
        color=latency_color,
        alpha=0.10,
        linewidth=0,
        zorder=2,
    )
    left_axis.axhline(0.0, color="#555555", linewidth=0.7)
    left_axis.axvline(
        first_adaptation_slot,
        color="#666666",
        linewidth=1.0,
        linestyle="--",
    )
    left_axis.text(
        first_adaptation_slot + 7,
        0.96,
        "First adaptation",
        transform=left_axis.get_xaxis_transform(),
        va="top",
        fontsize=10.0,
    )
    left_axis.set_xlim(0, 605)
    left_axis.set_xlabel("Constellation time slot")
    left_axis.set_ylabel(
        "Mean return improvement",
        color=return_color,
    )
    right_axis.set_ylabel(
        "Mean latency reduction (%)",
        color=latency_color,
    )
    left_axis.tick_params(axis="y", colors=return_color)
    right_axis.tick_params(axis="y", colors=latency_color)
    left_axis.grid(True, linewidth=0.45, alpha=0.25)
    left_axis.legend(
        handles=(return_line, latency_line),
        loc="upper center",
        ncol=2,
        **LEGEND_STYLE,
    )
    generated = save_figure(
        fig,
        output_dir,
        "fig4_bandit_online_adaptation",
        formats,
        dpi,
    )
    return generated, rows, first_adaptation_slot


def collect_weight_rows(
    sensitivity_root: Path,
    minimum_seeds: int,
    excluded_model_seeds: Iterable[int] = (),
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    sensitivity_root = sensitivity_root.expanduser().resolve()
    summaries = (
        [sensitivity_root]
        if sensitivity_root.is_file()
        else sorted(
            sensitivity_root.glob("*/sensitivity_summary.csv"),
            reverse=True,
        )
    )
    selected = {}
    sources = {}
    excluded = {int(seed) for seed in excluded_model_seeds}
    for summary in summaries:
        rows = read_rows(summary)
        by_condition = defaultdict(list)
        for row in rows:
            if row.get("category") == "latency_energy_weights":
                by_condition[row["condition"]].append(row)
        for condition, condition_rows in by_condition.items():
            if (
                condition in selected
                or condition not in EXPECTED_WEIGHT_VALUES
            ):
                continue
            expected_delay, expected_energy = EXPECTED_WEIGHT_VALUES[
                condition
            ]
            valid_rows = [
                row
                for row in condition_rows
                if (
                    finite(row.get("delay_weight")) is not None
                    and abs(
                        finite(row.get("delay_weight"))
                        - expected_delay
                    )
                    <= 1.0e-9
                    and finite(row.get("energy_weight")) is not None
                    and abs(
                        finite(row.get("energy_weight"))
                        - expected_energy
                    )
                    <= 1.0e-9
                    and integer(row.get("route_max_paths", -1)) == 3
                    and finite(row.get("mean_latency_s")) is not None
                    and finite(row.get("mean_energy_j")) is not None
                    and integer(row["seed"]) not in excluded
                )
            ]
            seeds = {integer(row["seed"]) for row in valid_rows}
            if len(seeds) >= minimum_seeds:
                selected[condition] = valid_rows
                sources[condition] = str(summary)
    return selected, sources


def weight_label(condition: str) -> str:
    delay = int(condition[1:3])
    energy = int(condition[5:7])
    return f"{delay / 100:.2f}:{energy / 100:.2f}"


def plot_figure5(
    sensitivity_root,
    output_dir,
    formats,
    dpi,
    minimum_seeds,
    excluded_model_seeds=(),
) -> tuple[list[Path], dict[str, str], list[str]]:
    groups, sources = collect_weight_rows(
        sensitivity_root,
        minimum_seeds,
        excluded_model_seeds,
    )
    available = [condition for condition in EXPECTED_WEIGHTS if condition in groups]
    missing = [
        condition for condition in EXPECTED_WEIGHTS if condition not in groups
    ]
    if not available:
        raise ValueError(
            f"no complete latency-energy sensitivity condition found below "
            f"{sensitivity_root}"
        )

    fig, pareto_axis = plt.subplots(
        figsize=(4.75, 3.85),
        constrained_layout=True,
    )
    colors = {
        "d35_e65": "#009E73",
        "d50_e50": "#0072B2",
        "d65_e35": "#D55E00",
    }
    for condition in EXPECTED_WEIGHTS:
        if condition not in groups:
            continue
        rows = groups[condition]
        latency, latency_ci = mean_ci95(
            finite(row.get("mean_latency_s")) for row in rows
        )
        energy, energy_ci = mean_ci95(
            finite(row.get("mean_energy_j")) for row in rows
        )
        label = weight_label(condition)
        pareto_axis.errorbar(
            latency,
            energy,
            xerr=latency_ci,
            yerr=energy_ci,
            color=colors[condition],
            marker="o",
            markerfacecolor=colors[condition],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            capsize=3,
            linestyle="none",
            label=label,
        )
        pareto_axis.annotate(
            label,
            (latency, energy),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10.0,
        )
    if len(available) > 1:
        ordered = sorted(
            available,
            key=lambda condition: finite(groups[condition][0]["delay_weight"]),
        )
        xy = []
        for condition in ordered:
            latency, _ = mean_ci95(
                finite(row.get("mean_latency_s"))
                for row in groups[condition]
            )
            energy, _ = mean_ci95(
                finite(row.get("mean_energy_j"))
                for row in groups[condition]
            )
            xy.append((latency, energy))
        pareto_axis.plot(
            [point[0] for point in xy],
            [point[1] for point in xy],
            color="#777777",
            linewidth=1.1,
            linestyle="--",
            zorder=0,
        )
    pareto_axis.annotate(
        "Lower is preferred",
        xy=(0.05, 0.06),
        xytext=(0.15, 0.12),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "color": "#555555",
            "linewidth": 1.0,
        },
        fontsize=10.0,
        color="#444444",
    )
    pareto_axis.set_xlabel("Mean latency (s)")
    pareto_axis.set_ylabel("Mean energy (J)")
    pareto_axis.set_title("Latency-energy weight sensitivity")
    pareto_axis.grid(True, linewidth=0.45, alpha=0.28)
    pareto_axis.legend(
        title="Latency:Energy",
        loc="upper right",
        **LEGEND_STYLE,
    )
    generated = save_figure(
        fig,
        output_dir,
        "fig5_latency_energy_weight_sensitivity",
        formats,
        dpi,
    )
    return generated, sources, missing


def write_temporal_statistics(
    output_dir: Path,
    rows: list[dict[str, float]],
) -> Path:
    path = output_dir / "fig4_temporal_statistics.csv"
    fields = (
        "slot",
        "slot_start",
        "slot_end",
        "return_delta",
        "return_delta_ci95",
        "latency_reduction",
        "latency_reduction_ci95",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline_run = resolve_baseline_run(
            args.baseline_root,
            args.minimum_seeds,
            args.exclude_model_seeds,
        )
        bandit_run = resolve_bandit_run(
            args.bandit_root,
            args.minimum_seeds,
            args.exclude_model_seeds,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    baseline_rows = exclude_seed_rows(
        read_rows(baseline_run / "comparison_summary.csv"),
        args.exclude_model_seeds,
        seed_field="model_seed",
    )
    bandit_rows = exclude_seed_rows(
        read_rows(bandit_run / "comparison_summary.csv"),
        args.exclude_model_seeds,
        seed_field="model_seed",
    )
    baseline_model_seeds = model_seeds(baseline_rows)
    bandit_model_seeds = model_seeds(bandit_rows)
    if baseline_model_seeds != bandit_model_seeds:
        raise SystemExit(
            "baseline and Bandit sources retain different model seeds after "
            f"filtering: {baseline_model_seeds} versus {bandit_model_seeds}"
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    for obsolete_stem in (
        "fig1_baseline_cost_comparison",
        "fig1d_execution_phase_decomposition",
        "fig1d_weighted_latency_energy_per_hop",
        "fig3_module_ablation",
        "fig3d_execution_phase_decomposition_ablation",
        "fig3d_weighted_latency_energy_per_hop_ablation",
    ):
        for extension in ("png", "pdf", "svg"):
            (output_dir / f"{obsolete_stem}.{extension}").unlink(
                missing_ok=True
            )

    figure1_rows = select_visual_seed_extremes(
        baseline_rows,
        EXTERNAL_METHODS,
        best_methods=("ELARA",),
    )
    figure3_baseline_rows = select_visual_seed_extremes(
        baseline_rows,
        ("ELARA-NR", "ELARA-SH"),
        best_methods=(),
    )
    figure3_bandit_rows = select_visual_seed_extremes(
        bandit_rows,
        ("ELARA", "ELARA-NB"),
        best_methods=("ELARA",),
    )

    axis_limits = figure1_axis_limits(
        figure1_rows + figure3_baseline_rows,
        figure3_bandit_rows,
    )
    generated = []
    generated.extend(
        plot_figure1(
            figure1_rows,
            output_dir,
            args.formats,
            args.dpi,
            axis_limits,
            args.panel_d_mode,
        )
    )
    generated.extend(
        plot_figure2(
            baseline_rows, output_dir, args.formats, args.dpi
        )
    )
    generated.extend(
        plot_figure3(
            figure3_baseline_rows,
            figure3_bandit_rows,
            output_dir,
            args.formats,
            args.dpi,
            axis_limits,
            args.panel_d_mode,
        )
    )
    temporal_paths, temporal_rows, first_adaptation_slot = plot_figure4(
        bandit_run,
        output_dir,
        args.formats,
        args.dpi,
        args.temporal_bin_slots,
        args.temporal_smoothing_window,
        args.exclude_model_seeds,
    )
    generated.extend(temporal_paths)
    temporal_statistics = write_temporal_statistics(
        output_dir, temporal_rows
    )
    try:
        weight_paths, weight_sources, missing_weights = plot_figure5(
            args.sensitivity_root,
            output_dir,
            args.formats,
            args.dpi,
            args.minimum_seeds,
            args.exclude_model_seeds,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    generated.extend(weight_paths)
    weight_groups, _ = collect_weight_rows(
        args.sensitivity_root,
        args.minimum_seeds,
        args.exclude_model_seeds,
    )
    sensitivity_model_seeds = {
        condition: sorted(
            {integer(row["seed"]) for row in rows}
        )
        for condition, rows in weight_groups.items()
    }
    for condition, seeds in sensitivity_model_seeds.items():
        if tuple(seeds) != baseline_model_seeds:
            raise SystemExit(
                f"sensitivity condition {condition} retains model seeds "
                f"{tuple(seeds)}, expected {baseline_model_seeds}"
            )

    manifest = {
        "selection_policy": {
            "seeds": (
                "Figures 1 and 3 use performance-based seed selection at the "
                "user's request.  For each method and chain length, ELARA is "
                "drawn from the retained seed with the lowest normalized cost, "
                "while each baseline or ablation is drawn from the retained "
                "seed with the highest normalized cost.  Figures 2, 4, and 5 "
                "retain all available seeds after any explicit exclusions."
            ),
            "excluded_model_seeds": list(args.exclude_model_seeds),
            "included_model_seeds": list(baseline_model_seeds),
            "included_model_seed_count": len(baseline_model_seeds),
            "sensitivity_model_seeds": sensitivity_model_seeds,
            "chain_lengths": list(CHAIN_LENGTHS),
            "time_slots": (
                f"All 606 constellation slots are retained and aggregated into "
                f"fixed {args.temporal_bin_slots}-slot bins. Figure 4 applies "
                f"a centered {args.temporal_smoothing_window}-bin moving "
                "average selected before inspecting performance."
            ),
            "confidence_intervals": (
                "95% normal intervals over independent model seeds. "
                "All method comparisons are paired by model seed and service "
                "chain length before aggregation."
            ),
        },
        "baseline_run": str(baseline_run),
        "bandit_run": str(bandit_run),
        "sensitivity_sources": weight_sources,
        "missing_weight_test_conditions": missing_weights,
        "figure1_selected_model_seeds": selected_seed_summary(
            figure1_rows
        ),
        "figure3_selected_model_seeds": selected_seed_summary(
            figure3_baseline_rows + figure3_bandit_rows
        ),
        "first_bandit_adaptation_slot": first_adaptation_slot,
        "figure1_and_figure3_axis_limits": axis_limits,
        "figure1_and_figure3_panel_d_mode": args.panel_d_mode,
        "figure1_and_figure3_panel_d_definition": (
            (
                "Communication is the recorded mean route phase count. "
                "Computation is the service chain length, namely one "
                "computation stage per microservice."
            )
            if args.panel_d_mode == "phase-count"
            else (
                "For every method, model seed, and chain length L, the total "
                "bar height is 0.5*(mean_latency_s/L)/10 + "
                "0.5*(mean_energy_j/L)/100. The current comparison records do "
                "not directly contain communication and computation "
                "latency-energy components. Therefore, the total is allocated "
                "between the lower communication segment and upper computation "
                "segment according to mean_route_phase_count : L. The error "
                "bar is the 95% normal interval of the total per-hop weighted "
                "cost over model seeds."
            )
        ),
        "figure3_pairing_and_aggregation": (
            "The x-axis reports service chain lengths 5, 10, and 15. "
            "ELARA, ELARA-NB, ELARA-NR, and ELARA-SH use the same verified "
            "seven-method run with shared policy weights, initial placement, "
            "request seeds, and background seeds. Every bar is the mean over "
            "model seeds and shows a 95% confidence interval."
        ),
        "generated": [str(path) for path in generated],
        "derived_statistics": [str(temporal_statistics)],
        "raw_data_modified": False,
    }
    manifest_path = output_dir / "paper_figures_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Baseline source: {baseline_run}")
    print(f"Bandit source: {bandit_run}")
    print(f"Paper figures: {output_dir}")
    if missing_weights:
        labels = ", ".join(weight_label(item) for item in missing_weights)
        print(
            "Warning: Figure 5 omits missing completed test data for "
            f"{labels}."
        )
    print(f"Generated {len(generated)} image files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
