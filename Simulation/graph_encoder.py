from __future__ import annotations

from .config import SimulationConfig
from .constellation import orbit_plane, sat_position
from .topology import TemporalGraph


def encode_satellite_graph(
    graph: TemporalGraph,
    context: dict,
    current_node: int | None = None,
    destination_node: int | None = None,
    current_service_id: int | None = None,
) -> dict[int, list[float]]:
    """Lightweight feature encoder used by heuristic and future GNN agents."""
    config: SimulationConfig = context["config"]
    resources = context["satellite_resources"]
    deployment_by_node = context["deployment_by_node"]
    queue_delay_table = context["queue_delay_table"]
    slot = graph.graph.get("time_slot", 0)

    features: dict[int, list[float]] = {}
    for node_id in graph.nodes:
        resource = resources[int(node_id)]
        deployed = deployment_by_node.get(int(node_id), set())
        has_service = (
            1.0 if current_service_id is not None and current_service_id in deployed else 0.0
        )
        features[int(node_id)] = [
            resource.base_freq_ghz / max(config.cpu_freq_choices_ghz),
            queue_delay_table[slot][int(node_id)] / max(1.0, context["slot_duration"]),
            len(deployed) / config.max_services_per_satellite,
            orbit_plane(int(node_id), config) / max(1, config.num_planes - 1),
            sat_position(int(node_id), config) / max(1, config.sats_per_plane - 1),
            1.0 if int(node_id) == current_node else 0.0,
            1.0 if int(node_id) == destination_node else 0.0,
            has_service,
            graph.degree[int(node_id)] / 4.0,
        ]
    return features
