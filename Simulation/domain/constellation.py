from __future__ import annotations

from ..config import SimulationConfig


def node_id_to_sat_name(node_id: int, config: SimulationConfig | None = None) -> str:
    config = config or SimulationConfig()
    zero_based = int(node_id) - 1
    return f"SAT_{zero_based // config.sats_per_plane}_{zero_based % config.sats_per_plane}"


def sat_name_to_node_id(sat_name: str, config: SimulationConfig | None = None) -> int:
    config = config or SimulationConfig()
    _, plane, sat = str(sat_name).split("_")
    return int(plane) * config.sats_per_plane + int(sat) + 1


def orbit_plane(node_id: int, config: SimulationConfig | None = None) -> int:
    config = config or SimulationConfig()
    return (int(node_id) - 1) // config.sats_per_plane


def sat_position(node_id: int, config: SimulationConfig | None = None) -> int:
    config = config or SimulationConfig()
    return (int(node_id) - 1) % config.sats_per_plane


def same_orbit(node_a: int, node_b: int, config: SimulationConfig | None = None) -> bool:
    return orbit_plane(node_a, config) == orbit_plane(node_b, config)
