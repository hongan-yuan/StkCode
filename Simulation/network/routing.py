from __future__ import annotations

import heapq
import math

from ..config import SimulationConfig
from ..domain.constellation import node_id_to_sat_name
from ..domain.energy import communication_energy_j
from .topology import TemporalGraph, slot_from_time


def _edge_key(node_u: int, node_v: int) -> tuple[int, int]:
    return tuple(sorted((int(node_u), int(node_v))))


def _edge_background(context: dict, slot_mod: int, node_u: int, node_v: int) -> dict[str, float]:
    table = context.get("link_background_table", {})
    return table.get(slot_mod, {}).get(_edge_key(node_u, node_v), {})


def edge_effective_rate_mbps(
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


def edge_queue_delay_s(context: dict, slot_mod: int, node_u: int, node_v: int) -> float:
    return max(0.0, float(_edge_background(context, slot_mod, node_u, node_v).get("queue_delay_s", 0.0)))


def build_routing_cache(snapshots: dict[int, TemporalGraph], config: SimulationConfig) -> dict:
    """Allocate per-run routing cache and stable stats fields."""

    return {
        "route_results": {},
        "stats": {
            "route_result_hits": 0,
            "route_result_misses": 0,
            "shortest_path_hits": 0,
            "shortest_path_misses": 0,
            "precomputed_slots": len(snapshots),
            "precomputed_nodes": config.total_sats,
        },
    }


def path_capacity_gb(
        graph: TemporalGraph,
        path: list[int],
        remaining_time_s: float,
        context: dict,
        slot_mod: int,
) -> float:
    if len(path) <= 1:
        return math.inf
    config: SimulationConfig = context["config"]
    tx_seconds_per_gb = 0.0
    propagation_s = 0.0
    queue_s = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v]
        rate_mbps = edge_effective_rate_mbps(edge, context, slot_mod, u, v)
        if rate_mbps <= 0.0:
            return 0.0
        tx_seconds_per_gb += 1.0e9 / (rate_mbps * 1.0e6)
        propagation_s += float(edge.get("distance_km", 0.0)) * 1000.0 / config.speed_of_light_m_per_s
        queue_s += edge_queue_delay_s(context, slot_mod, u, v)
    fixed_delay = propagation_s + queue_s + config.switch_penalty_s * max(0, len(path) - 1)
    usable_time = max(0.0, remaining_time_s - fixed_delay)
    if tx_seconds_per_gb <= 0.0 or not math.isfinite(tx_seconds_per_gb):
        return 0.0
    return usable_time / tx_seconds_per_gb


def path_delay_and_energy(
        graph: TemporalGraph,
        path: list[int],
        data_gb: float,
        context: dict,
        slot_mod: int,
) -> tuple[float, float, float, float, float]:
    config: SimulationConfig = context["config"]
    if data_gb <= 0.0 or len(path) <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    transmission_s = 0.0
    propagation_s = 0.0
    queue_s = 0.0
    energy_j = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v]
        rate_mbps = edge_effective_rate_mbps(edge, context, slot_mod, u, v)
        if rate_mbps <= 0.0 or not math.isfinite(rate_mbps):
            return math.inf, math.inf, propagation_s, queue_s, math.inf
        edge_tx_s = data_gb * 1.0e9 / (rate_mbps * 1.0e6)
        transmission_s += edge_tx_s
        propagation_s += float(edge.get("distance_km", 0.0)) * 1000.0 / config.speed_of_light_m_per_s
        queue_s += edge_queue_delay_s(context, slot_mod, u, v)
        tx_power_w = float(edge.get("tx_power_w", config.default_tx_power_w))
        energy_j += communication_energy_j(tx_power_w, edge_tx_s, config)
    switch_s = config.switch_penalty_s * max(0, len(path) - 1)
    delay_s = transmission_s + propagation_s + queue_s + switch_s
    return (
        delay_s,
        transmission_s,
        propagation_s,
        queue_s,
        energy_j,
    )


def future_path_failure_risk(
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
    horizon = max(1, horizon_slots or config.future_link_horizon_slots)
    failed_checks = 0
    total_checks = 0
    for offset in range(1, horizon + 1):
        graph = snapshots[(start_slot_mod + offset) % slot_count]
        for u, v in zip(path[:-1], path[1:]):
            total_checks += 1
            if not graph.has_edge(u, v):
                failed_checks += 1
    return failed_checks / total_checks if total_checks else 0.0


def future_edge_failure_risk(
        context: dict,
        node_u: int,
        node_v: int,
        start_slot_mod: int,
        horizon_slots: int | None = None,
) -> float:
    snapshots = context["snapshots"]
    slot_count = context["slot_count"]
    config: SimulationConfig = context["config"]
    horizon = max(1, horizon_slots or config.future_link_horizon_slots)
    failures = 0
    for offset in range(1, horizon + 1):
        graph = snapshots[(start_slot_mod + offset) % slot_count]
        if not graph.has_edge(node_u, node_v):
            failures += 1
    return failures / horizon


def edge_min_cost_flow_unit_cost(
        graph: TemporalGraph,
        node_u: int,
        node_v: int,
        context: dict,
        slot_mod: int,
) -> float:
    config: SimulationConfig = context["config"]
    edge = graph[node_u][node_v]
    rate_mbps = edge_effective_rate_mbps(edge, context, slot_mod, node_u, node_v)
    if rate_mbps <= 0.0:
        return math.inf
    one_gb_tx_s = 1.0e9 / (rate_mbps * 1.0e6)
    prop_s = float(edge.get("distance_km", 0.0)) * 1000.0 / config.speed_of_light_m_per_s
    queue_s = edge_queue_delay_s(context, slot_mod, node_u, node_v)
    tx_power_w = float(edge.get("tx_power_w", config.default_tx_power_w))
    energy_per_gb_j = communication_energy_j(tx_power_w, one_gb_tx_s, config)
    risk = future_edge_failure_risk(context, node_u, node_v, slot_mod)
    return (
            config.delay_weight * (one_gb_tx_s + prop_s + queue_s + config.switch_penalty_s)
            + config.energy_weight * energy_per_gb_j / 1000.0
            + config.route_failure_risk_weight * risk
    )


def edge_service_pressure_unit_cost(
        graph: TemporalGraph,
        node_u: int,
        node_v: int,
        context: dict,
        slot_mod: int,
) -> float:
    config: SimulationConfig = context["config"]
    edge = graph[node_u][node_v]
    rate_mbps = edge_effective_rate_mbps(edge, context, slot_mod, node_u, node_v)
    if rate_mbps <= 0.0:
        return math.inf
    one_gb_tx_s = 1.0e9 / (rate_mbps * 1.0e6)
    prop_s = float(edge.get("distance_km", 0.0)) * 1000.0 / config.speed_of_light_m_per_s
    queue_s = edge_queue_delay_s(context, slot_mod, node_u, node_v)
    background = _edge_background(context, slot_mod, node_u, node_v)
    rho = max(0.0, float(background.get("rho", 0.0)))
    risk = future_edge_failure_risk(context, node_u, node_v, slot_mod)
    delay_scale = max(1.0e-9, float(config.service_pressure_delay_scale_s))
    slot_duration = max(1.0e-9, float(context["slot_duration"]))
    return (
        rho
        + queue_s / slot_duration
        + config.service_pressure_route_failure_weight * risk
        + config.service_pressure_route_delay_weight * (one_gb_tx_s + prop_s) / delay_scale
        + config.switch_penalty_s / delay_scale
    )


def _edge_capacity_gb(
        graph: TemporalGraph,
        node_u: int,
        node_v: int,
        context: dict,
        slot_mod: int,
        remaining_time_s: float,
) -> float:
    edge = graph[node_u][node_v]
    rate_mbps = edge_effective_rate_mbps(edge, context, slot_mod, node_u, node_v)
    config: SimulationConfig = context["config"]
    propagation_s = float(edge.get("distance_km", 0.0)) * 1000.0 / config.speed_of_light_m_per_s
    queue_s = edge_queue_delay_s(context, slot_mod, node_u, node_v)
    usable_time_s = max(0.0, remaining_time_s - propagation_s - queue_s - config.switch_penalty_s)
    return rate_mbps * 1.0e6 * usable_time_s / 1.0e9


def _add_residual_edge(
        residual: dict[int, list[dict]],
        node_u: int,
        node_v: int,
        capacity_gb: float,
        cost: float,
        flow_key: tuple[int, int],
        flow_sign: float,
) -> None:
    forward = {
        "to": int(node_v),
        "rev": len(residual.setdefault(int(node_v), [])),
        "cap": float(capacity_gb),
        "cost": float(cost),
        "flow_key": flow_key,
        "flow_sign": float(flow_sign),
    }
    reverse = {
        "to": int(node_u),
        "rev": len(residual.setdefault(int(node_u), [])),
        "cap": 0.0,
        "cost": -float(cost),
        "flow_key": flow_key,
        "flow_sign": -float(flow_sign),
    }
    residual.setdefault(int(node_u), []).append(forward)
    residual.setdefault(int(node_v), []).append(reverse)


def _shortest_residual_path(
        residual: dict[int, list[dict]],
        source: int,
        target: int,
) -> tuple[float, dict[int, tuple[int, int]]] | None:
    nodes = list(residual)
    dist = {node: math.inf for node in nodes}
    prev: dict[int, tuple[int, int]] = {}
    in_queue = {node: False for node in nodes}
    queue = [int(source)]
    dist[int(source)] = 0.0
    in_queue[int(source)] = True

    while queue:
        node = queue.pop(0)
        in_queue[node] = False
        for edge_index, edge in enumerate(residual.get(node, [])):
            if edge["cap"] <= 1.0e-12:
                continue
            neighbor = edge["to"]
            next_cost = dist[node] + edge["cost"]
            if next_cost + 1.0e-12 < dist.get(neighbor, math.inf):
                dist[neighbor] = next_cost
                prev[neighbor] = (node, edge_index)
                if not in_queue.get(neighbor, False):
                    queue.append(neighbor)
                    in_queue[neighbor] = True

    if not math.isfinite(dist.get(int(target), math.inf)):
        return None
    return dist[int(target)], prev


def _node_path_from_residual_prev(
        source: int,
        target: int,
        residual: dict[int, list[dict]],
        prev: dict[int, tuple[int, int]],
) -> tuple[list[int], list[tuple[int, int]]]:
    node = int(target)
    nodes = [node]
    edge_refs: list[tuple[int, int]] = []
    while node != int(source):
        from_node, edge_index = prev[node]
        edge_refs.append((from_node, edge_index))
        node = from_node
        nodes.append(node)
    nodes.reverse()
    edge_refs.reverse()
    return nodes, edge_refs


def _shortest_hop_path(
        graph: TemporalGraph,
        source: int,
        target: int,
) -> list[int] | None:
    if source == target:
        return [int(source)]
    if source not in graph or target not in graph:
        return None
    queue: list[int] = [int(source)]
    parent: dict[int, int | None] = {int(source): None}
    while queue:
        node = queue.pop(0)
        neighbors = sorted(int(neighbor) for neighbor in graph.neighbors(node))
        for neighbor in neighbors:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == int(target):
                path = [neighbor]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(neighbor)
    return None


def _service_pressure_path(
        graph: TemporalGraph,
        source: int,
        target: int,
        context: dict,
        slot_mod: int,
) -> list[int] | None:
    if source == target:
        return [int(source)]
    if source not in graph or target not in graph:
        return None

    source = int(source)
    target = int(target)
    dist: dict[int, float] = {source: 0.0}
    parent: dict[int, int | None] = {source: None}
    heap: list[tuple[float, int, int]] = [(0.0, 0, source)]

    while heap:
        cost, hops, node = heapq.heappop(heap)
        if cost > dist.get(node, math.inf) + 1.0e-12:
            continue
        if node == target:
            path = [target]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        for neighbor in sorted(int(item) for item in graph.neighbors(node)):
            edge_cost = edge_service_pressure_unit_cost(
                graph, node, neighbor, context, slot_mod
            )
            if not math.isfinite(edge_cost):
                continue
            next_cost = cost + edge_cost
            if next_cost + 1.0e-12 < dist.get(neighbor, math.inf):
                dist[neighbor] = next_cost
                parent[neighbor] = node
                heapq.heappush(heap, (next_cost, hops + 1, neighbor))

    return None


def _decompose_positive_flow_paths(
        source: int,
        target: int,
        flows: dict[tuple[int, int], float],
        max_paths: int,
) -> list[tuple[list[int], float]]:
    flow_adj: dict[int, list[list[float]]] = {}
    for (node_u, node_v), flow in flows.items():
        if flow > 1.0e-9:
            flow_adj.setdefault(node_u, []).append([node_v, flow])

    decomposed: list[tuple[list[int], float]] = []
    for _ in range(max_paths):
        path = [int(source)]
        visited = {int(source)}

        def dfs(node: int, bottleneck: float) -> float:
            if node == int(target):
                return bottleneck
            for edge in flow_adj.get(node, []):
                neighbor = int(edge[0])
                residual_flow = float(edge[1])
                if residual_flow <= 1.0e-9 or neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                delivered = dfs(neighbor, min(bottleneck, residual_flow))
                if delivered > 1.0e-9:
                    edge[1] -= delivered
                    return delivered
                path.pop()
                visited.remove(neighbor)
            return 0.0

        amount = dfs(int(source), math.inf)
        if amount <= 1.0e-9:
            break
        decomposed.append((path.copy(), amount))
    return decomposed


def _min_cost_max_flow_slot(
        graph: TemporalGraph,
        source: int,
        target: int,
        remaining_data_gb: float,
        remaining_time_s: float,
        context: dict,
        slot_mod: int,
) -> dict:
    config: SimulationConfig = context["config"]
    residual: dict[int, list[dict]] = {int(node): [] for node in graph.nodes}
    flows: dict[tuple[int, int], float] = {}

    for node_u, node_v, _ in graph.edges(data=True):
        cap = _edge_capacity_gb(graph, node_u, node_v, context, slot_mod, remaining_time_s)
        if cap <= 1.0e-9:
            continue
        cost_uv = edge_min_cost_flow_unit_cost(graph, node_u, node_v, context, slot_mod)
        cost_vu = edge_min_cost_flow_unit_cost(graph, node_v, node_u, context, slot_mod)
        if math.isfinite(cost_uv):
            flows[(int(node_u), int(node_v))] = 0.0
            _add_residual_edge(
                residual, node_u, node_v, cap, cost_uv, (int(node_u), int(node_v)), 1.0
            )
        if math.isfinite(cost_vu):
            flows[(int(node_v), int(node_u))] = 0.0
            _add_residual_edge(
                residual, node_v, node_u, cap, cost_vu, (int(node_v), int(node_u)), 1.0
            )

    target_flow = max(0.0, remaining_data_gb)
    delivered_flow = 0.0
    augmentations = 0
    while (
            target_flow - delivered_flow > 1.0e-9
            and augmentations < config.min_cost_flow_max_augmentations_per_slot
    ):
        shortest = _shortest_residual_path(residual, source, target)
        if shortest is None:
            break
        _, prev = shortest
        node_path, edge_refs = _node_path_from_residual_prev(source, target, residual, prev)
        path_fit_cap = path_capacity_gb(graph, node_path, remaining_time_s, context, slot_mod)
        augment = min(
            target_flow - delivered_flow,
            path_fit_cap,
            *(residual[from_node][edge_index]["cap"] for from_node, edge_index in edge_refs),
        )
        if augment <= 1.0e-9 or not math.isfinite(augment):
            for from_node, edge_index in edge_refs:
                residual[from_node][edge_index]["cap"] = 0.0
            continue

        for from_node, edge_index in edge_refs:
            edge = residual[from_node][edge_index]
            edge["cap"] -= augment
            residual[edge["to"]][edge["rev"]]["cap"] += augment
            flows[edge["flow_key"]] = flows.get(edge["flow_key"], 0.0) + edge["flow_sign"] * augment
        delivered_flow += augment
        augmentations += 1

    path_records = []
    total_tx = 0.0
    total_prop = 0.0
    total_queue = 0.0
    total_energy = 0.0
    max_path_delay = 0.0
    max_failure_risk = 0.0
    max_bottleneck_capacity = 0.0
    representative_path: list[int] = []

    decomposed_paths = _decompose_positive_flow_paths(
        source,
        target,
        flows,
        max_paths=config.min_cost_flow_max_augmentations_per_slot,
    )
    for path_index, (path, amount) in enumerate(decomposed_paths):
        if amount <= 1.0e-9:
            continue
        delay_s, tx_s, prop_s, queue_s, energy_j = path_delay_and_energy(
            graph, path, amount, context, slot_mod
        )
        if not math.isfinite(delay_s) or delay_s > remaining_time_s + 1.0e-9:
            continue
        if not representative_path:
            representative_path = path
        bottleneck_capacity = path_capacity_gb(graph, path, remaining_time_s, context, slot_mod)
        failure_risk = future_path_failure_risk(context, path, slot_mod)
        max_path_delay = max(max_path_delay, delay_s)
        total_tx += tx_s
        total_prop += prop_s
        total_queue += queue_s
        total_energy += energy_j
        max_failure_risk = max(max_failure_risk, failure_risk)
        max_bottleneck_capacity = max(max_bottleneck_capacity, bottleneck_capacity)
        path_records.append(
            _slot_path_record(
                slot_mod,
                path,
                graph,
                amount,
                delay_s,
                tx_s,
                prop_s,
                "min_cost_flow" if path_index == 0 else f"min_cost_flow_{path_index + 1}",
                next_slot_failure=failure_risk > 0.0,
                context=context,
                queue_s=queue_s,
                bottleneck_capacity_gb=bottleneck_capacity,
                route_failure_risk=failure_risk,
            )
        )

    delivered = sum(record["data_gb"] for record in path_records)
    return {
        "delivered_data_gb": delivered,
        "slot_paths": path_records,
        "representative_path": representative_path,
        "max_path_delay_s": max_path_delay,
        "transmission_delay_s": total_tx,
        "propagation_delay_s": total_prop,
        "link_queue_delay_s": total_queue,
        "communication_energy_j": total_energy,
        "route_failure_risk": max_failure_risk,
        "bottleneck_capacity_gb": max_bottleneck_capacity,
        "augmentations": augmentations,
    }


def route_data(
        source: int,
        target: int,
        data_gb: float,
        start_time: float,
        context: dict
) -> dict:

    config: SimulationConfig = context["config"]
    snapshots = context["snapshots"]
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    routing_cache = context.get("routing_cache")
    route_results = routing_cache.get("route_results") if routing_cache else None
    stats = routing_cache.get("stats") if routing_cache else None

    route_cache_key = (
        str(getattr(config, "service_routing_strategy", "min_cost_max_flow")),
        int(source),
        int(target),
        round(float(data_gb), 9),
        round(float(start_time), 9),
    )

    if route_results is not None and route_cache_key in route_results:
        if stats is not None:
            stats["route_result_hits"] += 1
        return route_results[route_cache_key]
    if stats is not None:
        stats["route_result_misses"] += 1

    if source == target or data_gb <= 0.0:
        abs_slot, slot_mod = slot_from_time(start_time, slot_duration, slot_count)
        route = {
            "reachable": True,
            "route_mode": "local",
            "delay_s": 0.0,
            "transmission_delay_s": 0.0,
            "propagation_delay_s": 0.0,
            "communication_energy_j": 0.0,
            "arrival_time": start_time,
            "end_abs_slot": abs_slot,
            "end_slot_mod": slot_mod,
            "slot_crossings": 0,
            "path": [source],
            "slot_paths": [],
        }
        if route_results is not None:
            route_results[route_cache_key] = route
        return route

    strategy = getattr(config, "service_routing_strategy", "min_cost_max_flow")
    if strategy == "shortest_hop_per_slot":
        route = _route_shortest_hop_per_slot(source, target, data_gb, start_time, context)
    elif strategy == "service_pressure":
        route = _route_service_pressure_per_slot(source, target, data_gb, start_time, context)
    else:
        route = _route_min_cost_max_flow(source, target, data_gb, start_time, context)
    if route_results is not None:
        route_results[route_cache_key] = route
    return route


def _route_shortest_hop_per_slot(
        source: int, target: int, data_gb: float, start_time: float, context: dict
) -> dict:
    snapshots = context["snapshots"]
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    config: SimulationConfig = context["config"]

    start_abs_slot, _ = slot_from_time(start_time, slot_duration, slot_count)
    current_time = start_time
    remaining_data = data_gb
    total_tx = 0.0
    total_prop = 0.0
    total_queue = 0.0
    total_energy = 0.0
    slot_paths = []
    representative_path: list[int] = []
    max_failure_risk = 0.0
    max_bottleneck_capacity = 0.0

    for offset in range(config.route_horizon_slots):
        abs_slot, slot_mod = slot_from_time(current_time, slot_duration, slot_count)
        graph = snapshots[slot_mod]
        slot_end_time = (abs_slot + 1) * slot_duration
        remaining_time = max(0.0, slot_end_time - current_time)
        if remaining_time <= 0.0:
            current_time = slot_end_time
            continue

        path = _shortest_hop_path(graph, source, target)
        if not path:
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        bottleneck_capacity = path_capacity_gb(
            graph, path, remaining_time, context, slot_mod
        )
        delivered_this_slot = min(remaining_data, bottleneck_capacity)
        if delivered_this_slot <= 1.0e-9 or not math.isfinite(delivered_this_slot):
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        delay_s, tx_s, prop_s, queue_s, energy_j = path_delay_and_energy(
            graph, path, delivered_this_slot, context, slot_mod
        )
        if not math.isfinite(delay_s) or delay_s > remaining_time + 1.0e-9:
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        if not representative_path:
            representative_path = path
        failure_risk = future_path_failure_risk(context, path, slot_mod)
        slot_paths.append(
            _slot_path_record(
                slot_mod,
                path,
                graph,
                delivered_this_slot,
                delay_s,
                tx_s,
                prop_s,
                "shortest_hop",
                next_slot_failure=failure_risk > 0.0,
                context=context,
                queue_s=queue_s,
                bottleneck_capacity_gb=bottleneck_capacity,
                route_failure_risk=failure_risk,
            )
        )
        total_tx += tx_s
        total_prop += prop_s
        total_queue += queue_s
        total_energy += energy_j
        max_failure_risk = max(max_failure_risk, failure_risk)
        max_bottleneck_capacity = max(max_bottleneck_capacity, bottleneck_capacity)
        remaining_data = max(0.0, remaining_data - delivered_this_slot)

        if remaining_data <= 1.0e-9:
            arrival_time = current_time + delay_s
            end_abs_slot, end_slot_mod = slot_from_time(arrival_time, slot_duration, slot_count)
            return {
                "reachable": True,
                "route_mode": "shortest_hop_per_slot",
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

    route = _route_failure(source, target, start_time, context, "shortest_hop_route_failed")
    route["remaining_data_gb"] = remaining_data
    route["delivered_data_gb"] = max(0.0, data_gb - remaining_data)
    route["route_failure_risk"] = max_failure_risk
    route["bottleneck_capacity_gb"] = max_bottleneck_capacity
    route["end_to_end_delivery_only"] = True
    route["slot_paths"] = slot_paths
    route["path"] = representative_path
    route["min_cost_flow_augmentations"] = 0
    return route


def _route_service_pressure_per_slot(
        source: int, target: int, data_gb: float, start_time: float, context: dict
) -> dict:
    snapshots = context["snapshots"]
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    config: SimulationConfig = context["config"]

    start_abs_slot, _ = slot_from_time(start_time, slot_duration, slot_count)
    current_time = start_time
    remaining_data = data_gb
    total_tx = 0.0
    total_prop = 0.0
    total_queue = 0.0
    total_energy = 0.0
    slot_paths = []
    representative_path: list[int] = []
    max_failure_risk = 0.0
    max_bottleneck_capacity = 0.0

    for offset in range(config.route_horizon_slots):
        abs_slot, slot_mod = slot_from_time(current_time, slot_duration, slot_count)
        graph = snapshots[slot_mod]
        slot_end_time = (abs_slot + 1) * slot_duration
        remaining_time = max(0.0, slot_end_time - current_time)
        if remaining_time <= 0.0:
            current_time = slot_end_time
            continue

        path = _service_pressure_path(graph, source, target, context, slot_mod)
        if not path:
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        bottleneck_capacity = path_capacity_gb(
            graph, path, remaining_time, context, slot_mod
        )
        delivered_this_slot = min(remaining_data, bottleneck_capacity)
        if delivered_this_slot <= 1.0e-9 or not math.isfinite(delivered_this_slot):
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        delay_s, tx_s, prop_s, queue_s, energy_j = path_delay_and_energy(
            graph, path, delivered_this_slot, context, slot_mod
        )
        if not math.isfinite(delay_s) or delay_s > remaining_time + 1.0e-9:
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        if not representative_path:
            representative_path = path
        failure_risk = future_path_failure_risk(context, path, slot_mod)
        slot_paths.append(
            _slot_path_record(
                slot_mod,
                path,
                graph,
                delivered_this_slot,
                delay_s,
                tx_s,
                prop_s,
                "service_pressure_next_hop",
                next_slot_failure=failure_risk > 0.0,
                context=context,
                queue_s=queue_s,
                bottleneck_capacity_gb=bottleneck_capacity,
                route_failure_risk=failure_risk,
            )
        )
        total_tx += tx_s
        total_prop += prop_s
        total_queue += queue_s
        total_energy += energy_j
        max_failure_risk = max(max_failure_risk, failure_risk)
        max_bottleneck_capacity = max(max_bottleneck_capacity, bottleneck_capacity)
        remaining_data = max(0.0, remaining_data - delivered_this_slot)

        if remaining_data <= 1.0e-9:
            arrival_time = current_time + delay_s
            end_abs_slot, end_slot_mod = slot_from_time(arrival_time, slot_duration, slot_count)
            return {
                "reachable": True,
                "route_mode": "service_pressure",
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

    route = _route_failure(source, target, start_time, context, "service_pressure_route_failed")
    route["remaining_data_gb"] = remaining_data
    route["delivered_data_gb"] = max(0.0, data_gb - remaining_data)
    route["route_failure_risk"] = max_failure_risk
    route["bottleneck_capacity_gb"] = max_bottleneck_capacity
    route["end_to_end_delivery_only"] = True
    route["slot_paths"] = slot_paths
    route["path"] = representative_path
    route["min_cost_flow_augmentations"] = 0
    return route


def _route_min_cost_max_flow(
        source: int, target: int, data_gb: float, start_time: float, context: dict
) -> dict:
    snapshots = context["snapshots"]
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    config: SimulationConfig = context["config"]

    start_abs_slot, _ = slot_from_time(start_time, slot_duration, slot_count)
    current_time = start_time
    remaining_data = data_gb
    total_tx = 0.0
    total_prop = 0.0
    total_queue = 0.0
    total_energy = 0.0
    slot_paths = []
    representative_path: list[int] = []
    max_failure_risk = 0.0
    max_bottleneck_capacity = 0.0
    total_augmentations = 0

    for offset in range(config.route_horizon_slots):
        abs_slot, slot_mod = slot_from_time(current_time, slot_duration, slot_count)
        graph = snapshots[slot_mod]
        slot_end_time = (abs_slot + 1) * slot_duration
        remaining_time = max(0.0, slot_end_time - current_time)
        if remaining_time <= 0.0:
            current_time = slot_end_time
            continue

        slot_flow = _min_cost_max_flow_slot(
            graph,
            source,
            target,
            remaining_data,
            remaining_time,
            context,
            slot_mod,
        )
        delivered_this_slot = min(remaining_data, slot_flow["delivered_data_gb"])
        if delivered_this_slot <= 1.0e-9:
            if offset < config.route_horizon_slots - 1:
                current_time = slot_end_time
                continue
            break

        if not representative_path:
            representative_path = slot_flow["representative_path"]
        slot_paths.extend(slot_flow["slot_paths"])
        total_tx += slot_flow["transmission_delay_s"]
        total_prop += slot_flow["propagation_delay_s"]
        total_queue += slot_flow["link_queue_delay_s"]
        total_energy += slot_flow["communication_energy_j"]
        max_failure_risk = max(max_failure_risk, slot_flow["route_failure_risk"])
        max_bottleneck_capacity = max(max_bottleneck_capacity, slot_flow["bottleneck_capacity_gb"])
        total_augmentations += slot_flow["augmentations"]
        remaining_data = max(0.0, remaining_data - delivered_this_slot)

        if remaining_data <= 1.0e-9:
            arrival_time = current_time + slot_flow["max_path_delay_s"]
            end_abs_slot, end_slot_mod = slot_from_time(arrival_time, slot_duration, slot_count)
            return {
                "reachable": True,
                "route_mode": "min_cost_max_flow",
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
                "min_cost_flow_augmentations": total_augmentations,
            }

        current_time = slot_end_time

    route = _route_failure(source, target, start_time, context, "route_horizon_exceeded")
    route["remaining_data_gb"] = remaining_data
    route["delivered_data_gb"] = max(0.0, data_gb - remaining_data)
    route["route_failure_risk"] = max_failure_risk
    route["bottleneck_capacity_gb"] = max_bottleneck_capacity
    route["end_to_end_delivery_only"] = True
    route["slot_paths"] = slot_paths
    route["path"] = representative_path
    route["min_cost_flow_augmentations"] = total_augmentations
    return route


def _route_failure(source: int, target: int, start_time: float, context: dict, reason: str) -> dict:
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    abs_slot, slot_mod = slot_from_time(start_time, slot_duration, slot_count)
    return {
        "reachable": False,
        "failure_reason": reason,
        "delay_s": math.inf,
        "transmission_delay_s": math.inf,
        "propagation_delay_s": math.inf,
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
        slot: int,
        path: list[int],
        graph: TemporalGraph,
        data_gb: float,
        delay_s: float,
        tx_s: float,
        prop_s: float,
        role: str,
        next_slot_failure: bool | None = None,
        context: dict | None = None,
        queue_s: float = 0.0,
        bottleneck_capacity_gb: float | None = None,
        route_failure_risk: float | None = None,
) -> dict:
    return {
        "slot": slot,
        "role": role,
        "path": path,
        "path_satellites": [node_id_to_sat_name(node) for node in path],
        "data_gb": data_gb,
        "delay_s": delay_s,
        "transmission_delay_s": tx_s,
        "propagation_delay_s": prop_s,
        "link_queue_delay_s": queue_s,
        "next_slot_failure": next_slot_failure,
        "bottleneck_capacity_gb": bottleneck_capacity_gb,
        "route_failure_risk": route_failure_risk,
        "edges": [
            {
                "edge": (u, v),
                "rate_mbps": float(graph[u][v].get("rate_mbps", 0.0)),
                "effective_rate_mbps": (
                    edge_effective_rate_mbps(graph[u][v], context, slot, u, v)
                    if context is not None
                    else float(graph[u][v].get("rate_mbps", 0.0))
                ),
                "background_rho": (
                    _edge_background(context, slot, u, v).get("rho", 0.0)
                    if context is not None
                    else 0.0
                ),
                "background_queue_delay_s": (
                    _edge_background(context, slot, u, v).get("queue_delay_s", 0.0)
                    if context is not None
                    else 0.0
                ),
                "distance_km": float(graph[u][v].get("distance_km", 0.0)),
                "link_type": graph[u][v].get("link_type", ""),
                "link_id": graph[u][v].get("link_id", ""),
            }
            for u, v in zip(path[:-1], path[1:])
        ],
    }
