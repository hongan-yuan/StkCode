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
EXPECTED_WEIGHTS = ("d35_e65", "d50_e50", "d65_e35")
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
    "framealpha": 0.94,
    "facecolor": "white",
    "edgecolor": "#666666",
    "borderpad": 0.55,
    "labelspacing": 0.38,
    "handlelength": 1.7,
}


def parse_args(argv: list[str] | None = None):
    project_root = Path(__file__).resolve().parent.parent
    elara_root = project_root / "ELARA"
    parser = argparse.ArgumentParser(
        description=(
            "Generate the five ELARA paper figures from complete, paired "
            "experiments without performance-based seed or time-slot selection."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=elara_root / "outputs" / "baseline-tests",
        help="baseline-tests parent or a complete seven-method run",
    )
    parser.add_argument(
        "--bandit-root",
        type=Path,
        default=elara_root / "outputs" / "baseline-tests",
        help="baseline-tests parent or the corrected ELARA/ELARA-NB run",
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=elara_root / "outputs" / "sensitivity",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=elara_root / "paper_figs",
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
    parser.add_argument("--minimum-seeds", type=int, default=4)
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


def resolve_baseline_run(path: Path, minimum_seeds: int) -> Path:
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
        if _has_complete_matrix(rows, FULL_METHODS, minimum_seeds):
            return candidate
    raise ValueError(
        f"no complete seven-method baseline run with at least "
        f"{minimum_seeds} seeds was found below {path}"
    )


def resolve_bandit_run(path: Path, minimum_seeds: int) -> Path:
    path = path.expanduser().resolve()
    required = (
        "comparison_summary.csv",
        "bandit_ablation_paired.csv",
        "bandit_ablation_summary.json",
        "all_request_metrics.csv",
        "all_slot_metrics.csv",
    )
    candidates = [path] if all((path / name).is_file() for name in required) else []
    if path.is_dir() and not candidates:
        candidates = sorted(
            (
                candidate
                for candidate in path.iterdir()
                if all((candidate / name).is_file() for name in required)
            ),
            key=lambda candidate: candidate.name,
            reverse=True,
        )
    for candidate in candidates:
        summary = json.loads(
            (candidate / "bandit_ablation_summary.json").read_text(
                encoding="utf-8"
            )
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
        rows = read_rows(candidate / "comparison_summary.csv")
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

    phase_upper = 0.0
    phase_source_specs = [
        (method, rows)
        for method in (*EXTERNAL_METHODS, "ELARA-NR", "ELARA-SH")
    ]
    if bandit_rows is not None:
        phase_source_specs.extend(
            (method, bandit_rows)
            for method in ("ELARA", "ELARA-NB")
        )
    for method, source_rows in phase_source_specs:
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


def _phase_decomposition_panel(
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
        communication_means, communication_errors = [], []
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
        computation_stages = np.asarray(CHAIN_LENGTHS, dtype=float)
        communication_means = np.asarray(communication_means, dtype=float)
        totals = communication_means + computation_stages
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
            totals,
            yerr=communication_errors,
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


def plot_figure1(
    rows,
    output_dir,
    formats,
    dpi,
    axis_limits,
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
            figsize=(4.75, 3.85),
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
        figsize=(4.75, 3.85),
        constrained_layout=True,
    )
    _phase_decomposition_panel(
        ax,
        rows,
        EXTERNAL_METHODS,
        ylim=axis_limits["phase"],
    )
    generated.extend(
        save_figure(
            fig,
            output_dir,
            "fig1d_execution_phase_decomposition",
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


def _plot_ablation_phase_decomposition(
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
        communication_means, communication_errors = [], []
        for chain_length in CHAIN_LENGTHS:
            communication_mean, communication_error = mean_ci95(
                seed_metric_values(
                    rows,
                    method,
                    chain_length,
                    "mean_route_phase_count",
                )
            )
            communication_means.append(communication_mean)
            communication_errors.append(communication_error)
        computation_means = np.asarray(CHAIN_LENGTHS, dtype=float)
        communication_means = np.asarray(
            communication_means,
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
            yerr=communication_errors,
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
            label="Communication phases",
        ),
        Patch(
            facecolor="#BBBBBB",
            edgecolor="black",
            hatch="///",
            label="Computation stages",
        ),
    )
    ax.set_xticks(x, [str(value) for value in CHAIN_LENGTHS])
    ax.set_ylabel("Phase per request")
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
            figsize=(4.75, 3.85),
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
        figsize=(4.75, 3.85),
        constrained_layout=True,
    )
    _plot_ablation_phase_decomposition(
        ax,
        series_specs,
        axis_limits["phase"],
    )
    generated.extend(
        save_figure(
            fig,
            output_dir,
            "fig3d_execution_phase_decomposition_ablation",
            formats,
            dpi,
        )
    )
    return generated


def load_temporal_bandit_statistics(
    bandit_run: Path,
    bin_slots: int,
) -> tuple[list[dict[str, float]], int]:
    grouped = defaultdict(lambda: {"reward": [], "latency": []})
    for row in read_rows(bandit_run / "all_request_metrics.csv"):
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
    for row in read_rows(bandit_run / "all_slot_metrics.csv"):
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
) -> tuple[list[Path], list[dict[str, float]], int]:
    rows, first_adaptation_slot = load_temporal_bandit_statistics(
        bandit_run, bin_slots
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
    for summary in summaries:
        rows = read_rows(summary)
        by_condition = defaultdict(list)
        for row in rows:
            if row.get("category") == "latency_energy_weights":
                by_condition[row["condition"]].append(row)
        for condition, condition_rows in by_condition.items():
            if condition in selected:
                continue
            seeds = {integer(row["seed"]) for row in condition_rows}
            if len(seeds) >= minimum_seeds:
                selected[condition] = condition_rows
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
) -> tuple[list[Path], dict[str, str], list[str]]:
    groups, sources = collect_weight_rows(sensitivity_root, minimum_seeds)
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
        xytext=(0.37, 0.22),
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
            args.baseline_root, args.minimum_seeds
        )
        bandit_run = resolve_bandit_run(
            args.bandit_root, args.minimum_seeds
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    baseline_rows = read_rows(baseline_run / "comparison_summary.csv")
    bandit_rows = read_rows(bandit_run / "comparison_summary.csv")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    for obsolete_stem in (
        "fig1_baseline_cost_comparison",
        "fig3_module_ablation",
    ):
        for extension in ("png", "pdf", "svg"):
            (output_dir / f"{obsolete_stem}.{extension}").unlink(
                missing_ok=True
            )

    axis_limits = figure1_axis_limits(baseline_rows, bandit_rows)
    generated = []
    generated.extend(
        plot_figure1(
            baseline_rows,
            output_dir,
            args.formats,
            args.dpi,
            axis_limits,
        )
    )
    generated.extend(
        plot_figure2(
            baseline_rows, output_dir, args.formats, args.dpi
        )
    )
    generated.extend(
        plot_figure3(
            baseline_rows,
            bandit_rows,
            output_dir,
            args.formats,
            args.dpi,
            axis_limits,
        )
    )
    temporal_paths, temporal_rows, first_adaptation_slot = plot_figure4(
        bandit_run,
        output_dir,
        args.formats,
        args.dpi,
        args.temporal_bin_slots,
        args.temporal_smoothing_window,
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
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    generated.extend(weight_paths)

    manifest = {
        "selection_policy": {
            "seeds": (
                "All available matched seeds are retained. No seed is selected "
                "according to performance."
            ),
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
        "first_bandit_adaptation_slot": first_adaptation_slot,
        "figure1_and_figure3_axis_limits": axis_limits,
        "figure1_panel_d_definition": (
            "Communication is the recorded mean route phase count. "
            "Computation is the service chain length, namely one computation "
            "stage per microservice. The panel reports phase counts because "
            "the current baseline records do not contain communication and "
            "computation latency components."
        ),
        "figure3_pairing_and_aggregation": (
            "The x-axis reports service chain lengths 5, 10, and 15. "
            "ELARA and ELARA-NB use the corrected shared-initial-state "
            "Bandit run. ELARA-NR and ELARA-SH use the complete baseline "
            "run. Every bar is the mean over model seeds and shows a 95% "
            "confidence interval."
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
