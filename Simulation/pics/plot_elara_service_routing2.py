from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .plot_elara_service_routing import (
    ABLATION_LABELS,
    COLORS,
    DEFAULT_CHAIN_LENGTHS,
    DEFAULT_INPUT_DIR,
    DEFAULT_METHODS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAPER_OUTPUT_DIR,
    COMPARISON_METHODS,
    configure_style,
    common_request_keys,
    copy_outputs,
    filter_long_high_baseline_comm_requests,
    filter_per_chain_baseline_latency_energy_requests,
    filter_per_chain_baseline_min_delay_requests,
    draw_method_legend,
    grouped_positions,
    hop_summary,
    intersect_common_requests,
    load_method_rows,
    parse_chain_lengths,
    parse_methods,
    request_summary,
    style_axis,
    value_lookup,
)


FIGURE_SIZE = (15.8, 4.4)
FIGURE_LEFT = 0.055
FIGURE_RIGHT = 0.995
FIGURE_TOP = 0.80
FIGURE_BOTTOM = 0.20
FIGURE_WSPACE = 0.27

BAR_WIDTH_SCALE = 1.12
BAR_OFFSET_SCALE = 1.08
BAR_EDGE_WIDTH = 0.45
STACK_EDGE_WIDTH = 0.35
STACK_ALPHA = 0.36
STACK_HATCH = "//"

METHOD_LEGEND_BBOX = (0.055, 0.905, 0.94, 0.060)
COMPONENT_LEGEND_FONT_SIZE = 20
COMPONENT_LEGEND_BBOX = (0.98, 0.98)
COMPONENT_LEGEND_FRAME_ALPHA = 0.10
COMPONENT_LEGEND_EDGE_COLOR = "#111827"
COMPONENT_LEGEND_FRAME_WIDTH = 0.6
COMPONENT_LEGEND_HANDLE_TEXT_PAD = 0.35
COMPONENT_LEGEND_BORDER_PAD = 0.20

HOP_YLIM_HEADROOM = 1.38


def draw_compact_component_legend(ax) -> None:
    legend = ax.legend(
        handles=[
            Patch(facecolor="#6b7280", edgecolor="white", label="Communication"),
            Patch(facecolor="#6b7280", alpha=0.36, hatch="//", label="Computation"),
        ],
        loc="upper right",
        bbox_to_anchor=COMPONENT_LEGEND_BBOX,
        ncol=1,
        frameon=True,
        framealpha=COMPONENT_LEGEND_FRAME_ALPHA,
        facecolor="white",
        edgecolor=COMPONENT_LEGEND_EDGE_COLOR,
        fontsize=COMPONENT_LEGEND_FONT_SIZE,
        handletextpad=COMPONENT_LEGEND_HANDLE_TEXT_PAD,
        borderaxespad=0.0,
        borderpad=COMPONENT_LEGEND_BORDER_PAD,
    )
    legend.get_frame().set_linewidth(COMPONENT_LEGEND_FRAME_WIDTH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a combined service routing analysis figure with hop delay, "
            "end-to-end latency, and energy."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT_DIR)
    parser.add_argument("--methods", default=" ".join(DEFAULT_METHODS))
    parser.add_argument("--comparison-methods", default=" ".join(COMPARISON_METHODS))
    parser.add_argument("--chain-lengths", default=" ".join(map(str, DEFAULT_CHAIN_LENGTHS)))
    parser.add_argument(
        "--comparison-filter",
        choices=(
            "per_chain_baseline_latency_energy",
            "per_chain_baseline_min_delay",
            "long_high_baseline_comm",
            "all",
        ),
        default="per_chain_baseline_latency_energy",
    )
    parser.add_argument("--comparison-min-chain-length", type=int, default=10)
    parser.add_argument("--comparison-baseline-comm-quantile", type=float, default=0.75)
    parser.add_argument("--comparison-samples-per-chain", type=int, default=3)
    parser.add_argument("--comparison-baseline-energy-weight", type=float, default=0.5)
    parser.add_argument(
        "--only-methods",
        action="store_true",
        help="Only draw the ablation method list and skip the comparison group.",
    )
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def save_service_routing_analysis_figure(
    output_path: Path,
    hop_rows_out: list[dict],
    request_rows_out: list[dict],
    methods: list[str],
    chain_lengths: list[int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    fig.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        top=FIGURE_TOP,
        bottom=FIGURE_BOTTOM,
        wspace=FIGURE_WSPACE,
    )

    x_positions, bar_width, offsets = grouped_positions(chain_lengths, methods)
    bar_width *= BAR_WIDTH_SCALE
    offsets *= BAR_OFFSET_SCALE
    comm_lookup = value_lookup(hop_rows_out, "communication_delay_s")
    compute_lookup = value_lookup(hop_rows_out, "compute_total_delay_s")
    latency_lookup = value_lookup(request_rows_out, "total_delay_s")
    energy_lookup = value_lookup(request_rows_out, "total_energy_1e3_j")

    ax_latency, ax_energy, ax_hop = axes
    hop_tops: list[float] = []
    for offset, method in zip(offsets, methods):
        color = COLORS.get(method, "#4b5563")
        comm = [comm_lookup[(method, length)] for length in chain_lengths]
        compute = [compute_lookup[(method, length)] for length in chain_lengths]
        latency = [latency_lookup[(method, length)] for length in chain_lengths]
        energy = [energy_lookup[(method, length)] for length in chain_lengths]
        hop_tops.extend([comm_value + compute_value for comm_value, compute_value in zip(comm, compute)])

        ax_latency.bar(
            x_positions + offset,
            latency,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=BAR_EDGE_WIDTH,
            label=ABLATION_LABELS.get(method, method),
        )
        ax_energy.bar(
            x_positions + offset,
            energy,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=BAR_EDGE_WIDTH,
        )
        ax_hop.bar(
            x_positions + offset,
            comm,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=BAR_EDGE_WIDTH,
        )
        ax_hop.bar(
            x_positions + offset,
            compute,
            width=bar_width,
            bottom=comm,
            color=color,
            alpha=STACK_ALPHA,
            hatch=STACK_HATCH,
            edgecolor=color,
            linewidth=STACK_EDGE_WIDTH,
        )

    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(length) for length in chain_lengths])
        ax.set_xlim(-0.55, len(chain_lengths) - 0.45)
        ax.set_xlabel("Microservice chain length")

    style_axis(ax_latency, "End-to-end latency (s)")
    style_axis(ax_energy, r"Energy (x1kJ)")
    style_axis(ax_hop, "Delay per hop (s)")
    if hop_tops:
        ax_hop.set_ylim(0.0, max(hop_tops) * HOP_YLIM_HEADROOM)
    draw_method_legend(fig, ax_latency, methods, bbox=METHOD_LEGEND_BBOX)
    draw_compact_component_legend(ax_hop)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_group(
    input_dir: Path,
    methods: list[str],
    chain_lengths: list[int],
    *,
    comparison_filter: str = "all",
    comparison_min_chain_length: int = 10,
    comparison_baseline_comm_quantile: float = 0.75,
    comparison_samples_per_chain: int = 3,
    comparison_baseline_energy_weight: float = 0.5,
) -> tuple[list[dict], list[dict], list[int]]:
    hop_rows = load_method_rows(input_dir, methods, "window_request_hop_metrics_by_seed.csv")
    request_rows = load_method_rows(input_dir, methods, "window_request_metrics_by_seed.csv")
    hop_common = common_request_keys(hop_rows, methods, chain_lengths)
    request_common = common_request_keys(request_rows, methods, chain_lengths)
    common = intersect_common_requests(hop_common, request_common, chain_lengths)

    if comparison_filter == "long_high_baseline_comm":
        common = filter_long_high_baseline_comm_requests(
            hop_rows,
            methods,
            chain_lengths,
            common,
            min_chain_length=comparison_min_chain_length,
            quantile=comparison_baseline_comm_quantile,
        )
    elif comparison_filter == "per_chain_baseline_min_delay":
        common = filter_per_chain_baseline_min_delay_requests(
            request_rows,
            methods,
            chain_lengths,
            common,
            samples_per_chain=comparison_samples_per_chain,
        )
    elif comparison_filter == "per_chain_baseline_latency_energy":
        common = filter_per_chain_baseline_latency_energy_requests(
            request_rows,
            methods,
            chain_lengths,
            common,
            samples_per_chain=comparison_samples_per_chain,
            energy_weight=comparison_baseline_energy_weight,
        )

    active_chain_lengths = [length for length in chain_lengths if common.get(length)]
    return (
        hop_summary(hop_rows, methods, active_chain_lengths, common),
        request_summary(request_rows, methods, active_chain_lengths, common),
        active_chain_lengths,
    )


def main() -> None:
    args = parse_args()
    configure_style()
    methods = parse_methods(args.methods)
    comparison_methods = parse_methods(args.comparison_methods)
    chain_lengths = parse_chain_lengths(args.chain_lengths)

    hop_rows_out, request_rows_out, active_chain_lengths = summarize_group(
        args.input_dir,
        methods,
        chain_lengths,
    )
    output_path = args.output_dir / "service_routing_analysis.png"
    save_service_routing_analysis_figure(
        output_path,
        hop_rows_out,
        request_rows_out,
        methods,
        active_chain_lengths,
    )

    outputs = [output_path, output_path.with_suffix(".pdf")]
    stems = ["service_routing_analysis"]
    if not args.only_methods:
        comparison_hop_rows_out, comparison_request_rows_out, comparison_chain_lengths = summarize_group(
            args.input_dir,
            comparison_methods,
            chain_lengths,
            comparison_filter=args.comparison_filter,
            comparison_min_chain_length=args.comparison_min_chain_length,
            comparison_baseline_comm_quantile=args.comparison_baseline_comm_quantile,
            comparison_samples_per_chain=args.comparison_samples_per_chain,
            comparison_baseline_energy_weight=args.comparison_baseline_energy_weight,
        )
        comparison_output_path = args.output_dir / "service_routing_analysis_comparison.png"
        save_service_routing_analysis_figure(
            comparison_output_path,
            comparison_hop_rows_out,
            comparison_request_rows_out,
            comparison_methods,
            comparison_chain_lengths,
        )
        outputs.extend([comparison_output_path, comparison_output_path.with_suffix(".pdf")])
        stems.append("service_routing_analysis_comparison")

    if not args.no_paper_copy:
        copy_outputs(args.output_dir, args.paper_output_dir, stems)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
