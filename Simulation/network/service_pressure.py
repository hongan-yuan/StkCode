from __future__ import annotations

from collections import deque
import math
from typing import MutableMapping

from ..config import SimulationConfig
from ..domain.constellation import node_id_to_sat_name
from ..domain.energy import communication_energy_j
from .topology import TemporalGraph, slot_from_time


def _edge_key(node_u: int, node_v: int) -> tuple[int, int]:
    return tuple(sorted((int(node_u), int(node_v))))


def _edge_background(
    context: dict, slot_mod: int, node_u: int, node_v: int
) -> dict[str, float]:
    table = context.get("link_background_table", {})
    return table.get(slot_mod, {}).get(_edge_key(node_u, node_v), {})


def _edge_effective_rate_mbps(
    edge_data: dict,
    context: dict,
    slot_mod: int,
    node_u: int,
    node_v: int,
) -> float:
    background = _edge_background(context, slot_mod, node_u, node_v)
    if "effective_rate_mbps" in background:
        return max(0.0, float(background["effective_rate_mbps"]))
    return max(0.0, float(edge_data.get("rate_mbps", 0.0)))


def _edge_queue_delay_s(
    context: dict, slot_mod: int, node_u: int, node_v: int
) -> float:
    return max(
        0.0,
        float(_edge_background(context, slot_mod, node_u, node_v).get("queue_delay_s", 0.0)),
    )


def _hop_distances_to_targets(
    graph: TemporalGraph, targets: list[int]
) -> dict[int, int]:
    distances: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()
    for target in targets:
        target = int(target)
        if target in graph and target not in distances:
            distances[target] = 0
            queue.append((target, 0))
    while queue:
        node_id, depth = queue.popleft()
        for neighbor in graph.neighbors(node_id):
            neighbor = int(neighbor)
            if neighbor in distances:
                continue
            distances[neighbor] = depth + 1
            queue.append((neighbor, depth + 1))
    return distances


def _future_path_failure_risk(
    context: dict,
    path: list[int] | None,
    start_slot_mod: int,
    horizon_slots: int | None = None,
) -> float:
    if not path or len(path) <= 1:
        return 0.0
    snapshots = context["snapshots"]
    slot_count = context["slot_count"]
    config: SimulationConfig = context["config"]
    horizon = max(1, int(horizon_slots or config.future_link_horizon_slots))
    failed_checks = 0
    total_checks = 0
    for offset in range(1, horizon + 1):
        graph = snapshots[(start_slot_mod + offset) % slot_count]
        for node_u, node_v in zip(path[:-1], path[1:]):
            total_checks += 1
            if not graph.has_edge(node_u, node_v):
                failed_checks += 1
    return failed_checks / total_checks if total_checks else 0.0


def _path_delay_energy_capacity(
    graph: TemporalGraph,
    path: list[int],
    data_gb: float,
    remaining_time_s: float,
    context: dict,
    slot_mod: int,
) -> tuple[float, float, float, float, float, float]:
    if len(path) <= 1 or data_gb <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, math.inf

    config: SimulationConfig = context["config"]
    tx_seconds_per_gb = 0.0
    propagation_s = 0.0
    queue_s = 0.0
    for node_u, node_v in zip(path[:-1], path[1:]):
        edge = graph[node_u][node_v]
        rate_mbps = _edge_effective_rate_mbps(edge, context, slot_mod, node_u, node_v)
        if rate_mbps <= 0.0:
            return math.inf, math.inf, propagation_s, queue_s, math.inf, 0.0
        tx_seconds_per_gb += 1.0e9 / (rate_mbps * 1.0e6)
        propagation_s += (
            float(edge.get("distance_km", 0.0))
            * 1000.0
            / config.speed_of_light_m_per_s
        )
        queue_s += _edge_queue_delay_s(context, slot_mod, node_u, node_v)

    switch_s = config.switch_penalty_s * max(0, len(path) - 1)
    fixed_delay_s = propagation_s + queue_s + switch_s
    usable_time_s = max(0.0, remaining_time_s - fixed_delay_s)
    if tx_seconds_per_gb <= 0.0 or not math.isfinite(tx_seconds_per_gb):
        capacity_gb = 0.0
    else:
        capacity_gb = usable_time_s / tx_seconds_per_gb
    delivered_gb = min(max(0.0, data_gb), max(0.0, capacity_gb))

    transmission_s = 0.0
    energy_j = 0.0
    for node_u, node_v in zip(path[:-1], path[1:]):
        edge = graph[node_u][node_v]
        rate_mbps = _edge_effective_rate_mbps(edge, context, slot_mod, node_u, node_v)
        edge_tx_s = delivered_gb * 1.0e9 / (rate_mbps * 1.0e6)
        transmission_s += edge_tx_s
        tx_power_w = float(edge.get("tx_power_w", config.default_tx_power_w))
        energy_j += communication_energy_j(tx_power_w, edge_tx_s, config)

    delay_s = transmission_s + fixed_delay_s
    return delay_s, transmission_s, propagation_s, queue_s, energy_j, capacity_gb


class ServicePressureBackpressureRouter:
    """Packet-pressure route estimator for the Service Pressure baseline.

    The production router in ``routing.py`` remains untouched. This class keeps
    the baseline close to the paper's local C/T-node backpressure rule by
    selecting each next hop from queue differential pressure and hop-distance
    progress toward the requested virtual compute node.
    """

    def __init__(
        self,
        config: SimulationConfig,
        virtual_backlog: MutableMapping[tuple[int, int], float] | None = None,
        virtual_backlog_weight: float = 0.25,
        priority_alpha: float = 1.0,
    ):
        self.config = config
        self.virtual_backlog = virtual_backlog if virtual_backlog is not None else {}
        self.virtual_backlog_weight = max(0.0, float(virtual_backlog_weight))
        self.priority_alpha = max(0.0, min(1.0, float(priority_alpha)))

    def route(
        self,
        source: int,
        target: int,
        data_gb: float,
        start_time: float,
        context: dict,
        service_id: int = 0,
        target_is_compute: bool = True,
    ) -> dict:
        source = int(source)
        target = int(target)
        service_id = int(service_id)
        if source == target or data_gb <= 0.0:
            abs_slot, slot_mod = slot_from_time(
                start_time, context["slot_duration"], context["slot_count"]
            )
            return {
                "reachable": True,
                "route_mode": "service_pressure_backpressure",
                "delay_s": 0.0,
                "transmission_delay_s": 0.0,
                "propagation_delay_s": 0.0,
                "link_queue_delay_s": 0.0,
                "communication_energy_j": 0.0,
                "arrival_time": start_time,
                "end_abs_slot": abs_slot,
                "end_slot_mod": slot_mod,
                "slot_crossings": 0,
                "path": [source],
                "slot_paths": [],
                "remaining_data_gb": 0.0,
                "delivered_data_gb": data_gb,
                "route_failure_risk": 0.0,
                "bottleneck_capacity_gb": math.inf,
                "end_to_end_delivery_only": True,
                "min_cost_flow_augmentations": 0,
            }

        snapshots = context["snapshots"]
        slot_duration = context["slot_duration"]
        slot_count = context["slot_count"]
        start_abs_slot, _ = slot_from_time(start_time, slot_duration, slot_count)

        current_time = start_time
        remaining_data = float(data_gb)
        total_tx = 0.0
        total_prop = 0.0
        total_queue = 0.0
        total_energy = 0.0
        slot_paths: list[dict] = []
        representative_path: list[int] = []
        max_failure_risk = 0.0
        max_bottleneck_capacity = 0.0

        for offset in range(self.config.route_horizon_slots):
            abs_slot, slot_mod = slot_from_time(current_time, slot_duration, slot_count)
            graph = snapshots[slot_mod]
            slot_end_time = (abs_slot + 1) * slot_duration
            remaining_time_s = max(0.0, slot_end_time - current_time)
            if remaining_time_s <= 0.0:
                current_time = slot_end_time
                continue

            path = self._select_backpressure_path(
                graph,
                source,
                target,
                remaining_data,
                context,
                slot_mod,
                service_id,
                target_is_compute,
            )
            if not path:
                current_time = slot_end_time
                continue

            (
                delay_s,
                tx_s,
                prop_s,
                queue_s,
                energy_j,
                bottleneck_capacity_gb,
            ) = _path_delay_energy_capacity(
                graph,
                path,
                remaining_data,
                remaining_time_s,
                context,
                slot_mod,
            )
            delivered_this_slot = min(remaining_data, bottleneck_capacity_gb)
            if (
                delivered_this_slot <= 1.0e-9
                or not math.isfinite(delay_s)
                or delay_s > remaining_time_s + 1.0e-9
            ):
                current_time = slot_end_time
                continue

            if not representative_path:
                representative_path = path
            failure_risk = _future_path_failure_risk(context, path, slot_mod)
            slot_paths.append(
                self._slot_path_record(
                    slot_mod,
                    path,
                    graph,
                    delivered_this_slot,
                    delay_s,
                    tx_s,
                    prop_s,
                    queue_s,
                    bottleneck_capacity_gb,
                    failure_risk,
                    context,
                )
            )
            total_tx += tx_s
            total_prop += prop_s
            total_queue += queue_s
            total_energy += energy_j
            max_failure_risk = max(max_failure_risk, failure_risk)
            max_bottleneck_capacity = max(max_bottleneck_capacity, bottleneck_capacity_gb)
            remaining_data = max(0.0, remaining_data - delivered_this_slot)

            if remaining_data <= 1.0e-9:
                arrival_time = current_time + delay_s
                end_abs_slot, end_slot_mod = slot_from_time(
                    arrival_time, slot_duration, slot_count
                )
                return {
                    "reachable": True,
                    "route_mode": "service_pressure_backpressure",
                    "delay_s": arrival_time - start_time,
                    "transmission_delay_s": total_tx,
                    "propagation_delay_s": total_prop,
                    "link_queue_delay_s": total_queue,
                    "communication_energy_j": total_energy,
                    "arrival_time": arrival_time,
                    "end_abs_slot": end_abs_slot,
                    "end_slot_mod": end_slot_mod,
                    "slot_crossings": max(0, end_abs_slot - start_abs_slot),
                    "path": representative_path,
                    "slot_paths": slot_paths,
                    "remaining_data_gb": 0.0,
                    "delivered_data_gb": data_gb,
                    "route_failure_risk": max_failure_risk,
                    "bottleneck_capacity_gb": max_bottleneck_capacity,
                    "end_to_end_delivery_only": True,
                    "min_cost_flow_augmentations": 0,
                }

            current_time = slot_end_time

        route = self._route_failure(
            source,
            target,
            start_time,
            context,
            "service_pressure_backpressure_route_failed",
        )
        route["remaining_data_gb"] = remaining_data
        route["delivered_data_gb"] = max(0.0, data_gb - remaining_data)
        route["route_failure_risk"] = max_failure_risk
        route["bottleneck_capacity_gb"] = max_bottleneck_capacity
        route["end_to_end_delivery_only"] = True
        route["slot_paths"] = slot_paths
        route["path"] = representative_path
        route["min_cost_flow_augmentations"] = 0
        return route

    def _select_backpressure_path(
        self,
        graph: TemporalGraph,
        source: int,
        target: int,
        data_gb: float,
        context: dict,
        slot_mod: int,
        service_id: int,
        target_is_compute: bool,
    ) -> list[int] | None:
        if source not in graph or target not in graph:
            return None
        distances = _hop_distances_to_targets(graph, [target])
        if source not in distances:
            return None

        path = [int(source)]
        visited = {int(source)}
        current = int(source)
        max_hops = max(1, len(graph.nodes))
        for _ in range(max_hops):
            if current == int(target):
                return path
            next_node = self._select_next_hop(
                graph,
                current,
                target,
                data_gb,
                context,
                slot_mod,
                service_id,
                distances,
                target_is_compute,
                visited,
            )
            if next_node is None:
                return None
            path.append(next_node)
            if next_node == int(target):
                return path
            visited.add(next_node)
            current = next_node
        return None

    def _select_next_hop(
        self,
        graph: TemporalGraph,
        current: int,
        target: int,
        data_gb: float,
        context: dict,
        slot_mod: int,
        service_id: int,
        distances: dict[int, int],
        target_is_compute: bool,
        visited: set[int],
    ) -> int | None:
        current_distance = distances.get(int(current), math.inf)
        if not math.isfinite(current_distance):
            return None
        current_pressure = self._transmission_pressure(
            current, service_id, slot_mod, context
        ) + max(0.0, float(data_gb))

        ranked = []
        for neighbor in sorted(int(node_id) for node_id in graph.neighbors(current)):
            if neighbor not in distances:
                continue
            edge = graph[current][neighbor]
            if _edge_effective_rate_mbps(edge, context, slot_mod, current, neighbor) <= 0.0:
                continue
            if neighbor == int(target) and target_is_compute:
                neighbor_pressure = self._compute_pressure(
                    neighbor, service_id, slot_mod, context
                )
            else:
                neighbor_pressure = self._transmission_pressure(
                    neighbor, service_id, slot_mod, context
                )
            differential = self._priority_weight(current_pressure) - self._priority_weight(
                neighbor_pressure
            )
            distance_factor = 1.0 + current_distance - distances[neighbor]
            adjusted = differential * distance_factor
            loop_penalty = 1 if neighbor in visited and neighbor != int(target) else 0
            ranked.append(
                (
                    adjusted,
                    -loop_penalty,
                    current_distance - distances[neighbor],
                    -neighbor_pressure,
                    -neighbor,
                    neighbor,
                )
            )

        ranked.sort(reverse=True)
        for adjusted, no_loop, *_rest, neighbor in ranked:
            if no_loop < 0:
                continue
            if adjusted > 0.0 or neighbor == int(target):
                return int(neighbor)
        return None

    def _priority_weight(self, pressure: float) -> float:
        pressure = max(0.0, float(pressure))
        if self.priority_alpha >= 1.0:
            return pressure
        return pressure ** self.priority_alpha

    def _compute_pressure(
        self, node_id: int, service_id: int, slot_mod: int, context: dict
    ) -> float:
        slot_duration = max(1.0e-9, float(context["slot_duration"]))
        queue_delay = float(
            context.get("queue_delay_table", {}).get(slot_mod, {}).get(node_id, 0.0)
        )
        utilization = float(
            context.get("compute_utilization_table", {})
            .get(slot_mod, {})
            .get(node_id, 0.0)
        )
        discount = float(
            context.get("discount_table", {}).get(slot_mod, {}).get(node_id, 1.0)
        )
        backlog = float(self.virtual_backlog.get((int(node_id), int(service_id)), 0.0))
        return (
            utilization
            + queue_delay / slot_duration
            + 0.50 * max(0.0, 1.0 - discount)
            + self.virtual_backlog_weight * backlog
        )

    def _transmission_pressure(
        self, node_id: int, service_id: int, slot_mod: int, context: dict
    ) -> float:
        graph = context["snapshots"][slot_mod]
        slot_duration = max(1.0e-9, float(context["slot_duration"]))
        edge_pressures = []
        if node_id in graph:
            for neighbor in graph.neighbors(node_id):
                background = _edge_background(context, slot_mod, node_id, int(neighbor))
                edge_pressures.append(
                    float(background.get("rho", 0.0))
                    + float(background.get("queue_delay_s", 0.0)) / slot_duration
                )
        link_pressure = (
            sum(edge_pressures) / len(edge_pressures) if edge_pressures else 1.0
        )
        backlog = float(self.virtual_backlog.get((int(node_id), int(service_id)), 0.0))
        return link_pressure + self.virtual_backlog_weight * backlog

    def _route_failure(
        self,
        source: int,
        target: int,
        start_time: float,
        context: dict,
        reason: str,
    ) -> dict:
        slot_duration = context["slot_duration"]
        slot_count = context["slot_count"]
        abs_slot, slot_mod = slot_from_time(start_time, slot_duration, slot_count)
        return {
            "reachable": False,
            "failure_reason": reason,
            "route_mode": "service_pressure_backpressure",
            "delay_s": math.inf,
            "transmission_delay_s": math.inf,
            "propagation_delay_s": math.inf,
            "link_queue_delay_s": math.inf,
            "communication_energy_j": math.inf,
            "arrival_time": math.inf,
            "end_abs_slot": abs_slot,
            "end_slot_mod": slot_mod,
            "slot_crossings": 0,
            "path": [],
            "slot_paths": [],
            "source_node": source,
            "target_node": target,
        }

    def _slot_path_record(
        self,
        slot: int,
        path: list[int],
        graph: TemporalGraph,
        data_gb: float,
        delay_s: float,
        tx_s: float,
        prop_s: float,
        queue_s: float,
        bottleneck_capacity_gb: float,
        route_failure_risk: float,
        context: dict,
    ) -> dict:
        return {
            "slot": slot,
            "role": "service_pressure_backpressure",
            "path": path,
            "path_satellites": [node_id_to_sat_name(node) for node in path],
            "data_gb": data_gb,
            "delay_s": delay_s,
            "transmission_delay_s": tx_s,
            "propagation_delay_s": prop_s,
            "link_queue_delay_s": queue_s,
            "next_slot_failure": route_failure_risk > 0.0,
            "bottleneck_capacity_gb": bottleneck_capacity_gb,
            "route_failure_risk": route_failure_risk,
            "edges": [
                {
                    "edge": (node_u, node_v),
                    "rate_mbps": float(graph[node_u][node_v].get("rate_mbps", 0.0)),
                    "effective_rate_mbps": _edge_effective_rate_mbps(
                        graph[node_u][node_v], context, slot, node_u, node_v
                    ),
                    "background_rho": _edge_background(
                        context, slot, node_u, node_v
                    ).get("rho", 0.0),
                    "background_queue_delay_s": _edge_background(
                        context, slot, node_u, node_v
                    ).get("queue_delay_s", 0.0),
                    "distance_km": float(graph[node_u][node_v].get("distance_km", 0.0)),
                    "link_type": graph[node_u][node_v].get("link_type", ""),
                    "link_id": graph[node_u][node_v].get("link_id", ""),
                }
                for node_u, node_v in zip(path[:-1], path[1:])
            ],
        }
