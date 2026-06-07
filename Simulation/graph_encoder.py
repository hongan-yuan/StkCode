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
    discount_table = context.get("discount_table", {})
    utilization_table = context.get("compute_utilization_table", {})
    state_table = context.get("compute_load_state_table", {})
    slot = graph.graph.get("time_slot", 0)
    state_index = {
        state: idx / max(1, len(config.compute_load_states) - 1)
        for idx, state in enumerate(config.compute_load_states)
    }
    fallback_state = config.compute_load_states[0] if config.compute_load_states else "Idle"

    features: dict[int, list[float]] = {}
    for node_id in graph.nodes:
        resource = resources[int(node_id)]
        deployed = deployment_by_node.get(int(node_id), set())
        cpu_discount = float(discount_table.get(slot, {}).get(int(node_id), 1.0))
        utilization = float(utilization_table.get(slot, {}).get(int(node_id), 0.0))
        load_state = state_table.get(slot, {}).get(int(node_id), fallback_state)
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
            utilization,
            cpu_discount,
            state_index.get(load_state, 0.0),
        ]
    return features
