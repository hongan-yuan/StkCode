from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Callable

from .topology import ISLEdge, SparseGraph, TemporalTopology


GraphProvider = Callable[[float], SparseGraph]
RateProvider = Callable[[ISLEdge, int], float]
QueueProvider = Callable[[ISLEdge, float], float]
ReservationCallback = Callable[[ISLEdge, float, float, float], None]


class CrossSlotMinCostRouter:
    """Capacity-aware splittable routing with complete per-slot paths.

    A phase reserves at most ``K`` complete source-to-target paths.  Only data
    that reaches the target before the slot boundary is committed.  If data is
    left, it remains at the phase source and is rerouted on the next snapshot.
    """

    def __init__(self, topology: TemporalTopology, config):
        self.topology = topology
        self.config = config

    def _future_edge_risk(self, edge: ISLEdge, absolute_slot: int) -> float:
        horizon = max(1, self.config.future_topology_horizon)
        failures = 0
        for offset in range(1, horizon + 1):
            if self.topology.graph_at_slot(absolute_slot + offset).edge(edge.u, edge.v) is None:
                failures += 1
        return failures / horizon

    def _edge_unit_cost(
        self,
        edge: ISLEdge,
        absolute_slot: int,
        current_time: float,
        rate_provider: RateProvider,
        queue_provider: QueueProvider,
        rate: float | None = None,
        queue: float | None = None,
    ) -> float:
        rate = max(
            1.0e-9,
            rate_provider(edge, absolute_slot) if rate is None else rate,
        )
        tx_per_gb = 8_000.0 / rate
        propagation = edge.distance_km / self.config.speed_of_light_km_s
        queue = (
            queue_provider(edge, current_time) if queue is None else queue
        )
        energy_per_gb = edge.tx_power_w * tx_per_gb
        risk_cost = 0.0
        if self.config.route_failure_risk_weight > 0.0:
            risk_cost = (
                self.config.route_failure_risk_weight
                * self._future_edge_risk(edge, absolute_slot)
            )
        return (
            self.config.delay_weight
            * (tx_per_gb + propagation + queue + self.config.route_switch_delay_s)
            / max(self.config.latency_scale_s, 1.0e-9)
            + self.config.energy_weight
            * energy_per_gb
            / max(self.config.energy_scale_j, 1.0e-9)
            + risk_cost
        )

    @staticmethod
    def _shortest_residual_path(
        graph: SparseGraph,
        source: int,
        target: int,
        capacities: dict[tuple[int, int], float],
        costs: dict[tuple[int, int], float],
    ) -> list[int] | None:
        distance = {int(source): 0.0}
        parent: dict[int, int] = {}
        heap = [(0.0, int(source))]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > distance.get(node, math.inf) + 1.0e-12:
                continue
            if node == target:
                break
            for neighbor in graph.neighbors(node):
                key = (min(node, neighbor), max(node, neighbor))
                if capacities.get(key, 0.0) <= 1.0e-12:
                    continue
                new_cost = cost + costs[key]
                if new_cost + 1.0e-12 < distance.get(neighbor, math.inf):
                    distance[neighbor] = new_cost
                    parent[neighbor] = node
                    heapq.heappush(heap, (new_cost, neighbor))
        if target not in distance:
            return None
        path = [int(target)]
        while path[-1] != source:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def _slot_flow(
        self,
        graph: SparseGraph,
        source: int,
        target: int,
        remaining_data_gb: float,
        current_time: float,
        slot_end: float,
        absolute_slot: int,
        rate_provider: RateProvider,
        queue_provider: QueueProvider,
        reserve: ReservationCallback | None,
        commit: bool,
    ) -> dict:
        remaining_time = max(0.0, slot_end - current_time)
        capacities: dict[tuple[int, int], float] = {}
        costs: dict[tuple[int, int], float] = {}
        rates: dict[tuple[int, int], float] = {}
        queues: dict[tuple[int, int], float] = {}
        edge_by_key: dict[tuple[int, int], ISLEdge] = {}
        for edge in graph.edges():
            key = edge.key
            rate = max(0.0, rate_provider(edge, absolute_slot))
            queue = max(0.0, queue_provider(edge, current_time))
            propagation = edge.distance_km / self.config.speed_of_light_km_s
            usable = max(
                0.0,
                remaining_time - queue - propagation - self.config.route_switch_delay_s,
            )
            capacity = rate * usable / 8_000.0
            if capacity <= 1.0e-12:
                continue
            capacities[key] = capacity
            costs[key] = self._edge_unit_cost(
                edge,
                absolute_slot,
                current_time,
                rate_provider,
                queue_provider,
                rate=rate,
                queue=queue,
            )
            rates[key] = rate
            queues[key] = queue
            edge_by_key[key] = edge

        paths: list[dict] = []
        edge_loads: dict[tuple[int, int], float] = defaultdict(float)
        delivered = 0.0
        for _ in range(self.config.route_max_paths_per_slot):
            demand = remaining_data_gb - delivered
            if demand <= 1.0e-12:
                break
            path = self._shortest_residual_path(graph, source, target, capacities, costs)
            if path is None:
                break
            keys = [(min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:])]
            fixed_delay = sum(
                queues[key]
                + edge_by_key[key].distance_km / self.config.speed_of_light_km_s
                + self.config.route_switch_delay_s
                for key in keys
            )
            seconds_per_gb = sum(8_000.0 / max(rates[key], 1.0e-9) for key in keys)
            fit_capacity = max(0.0, remaining_time - fixed_delay) / max(seconds_per_gb, 1.0e-9)
            amount = min(demand, fit_capacity, *(capacities[key] for key in keys))
            if amount <= 1.0e-12:
                for key in keys:
                    capacities[key] = 0.0
                continue
            for key in keys:
                capacities[key] -= amount
                edge_loads[key] += amount
            paths.append({"path": path, "data_gb": amount, "fixed_delay_s": fixed_delay})
            delivered += amount

        # Shared ISLs serialize aggregate data. Recompute every path completion
        # time with aggregate edge loads before committing the phase.
        phase_delay = 0.0
        transmission_delay = 0.0
        propagation_delay = 0.0
        queue_delay = 0.0
        energy = 0.0
        for record in paths:
            keys = [
                (min(u, v), max(u, v))
                for u, v in zip(record["path"][:-1], record["path"][1:])
            ]
            path_tx = sum(edge_loads[key] * 8_000.0 / rates[key] for key in keys)
            path_prop = sum(
                edge_by_key[key].distance_km / self.config.speed_of_light_km_s
                for key in keys
            )
            path_queue = sum(queues[key] for key in keys)
            path_delay = (
                path_tx
                + path_prop
                + path_queue
                + self.config.route_switch_delay_s * len(keys)
            )
            record.update(
                delay_s=path_delay,
                transmission_delay_s=path_tx,
                propagation_delay_s=path_prop,
                queue_delay_s=path_queue,
            )
            phase_delay = max(phase_delay, path_delay)
            transmission_delay += sum(
                record["data_gb"] * 8_000.0 / rates[key] for key in keys
            )
            propagation_delay += path_prop
            queue_delay += path_queue
            energy += sum(
                edge_by_key[key].tx_power_w
                * record["data_gb"]
                * 8_000.0
                / rates[key]
                for key in keys
            )

        if phase_delay > remaining_time + 1.0e-9:
            return {"delivered_data_gb": 0.0, "paths": [], "phase_delay_s": 0.0}
        if commit and reserve is not None:
            for key, load in edge_loads.items():
                reserve(edge_by_key[key], current_time, load, rates[key])
        return {
            "delivered_data_gb": delivered,
            "paths": paths,
            "phase_delay_s": phase_delay,
            "transmission_delay_s": transmission_delay,
            "propagation_delay_s": propagation_delay,
            "queue_delay_s": queue_delay,
            "energy_j": energy,
            "edge_loads_gb": dict(edge_loads),
        }

    def route(
        self,
        source: int,
        target: int,
        data_gb: float,
        start_time: float,
        graph_provider: GraphProvider,
        rate_provider: RateProvider,
        queue_provider: QueueProvider,
        reserve: ReservationCallback | None = None,
        commit: bool = True,
    ) -> dict:
        if source == target or data_gb <= 0.0:
            return {
                "reachable": True,
                "route_mode": "local",
                "source": source,
                "target": target,
                "data_gb": data_gb,
                "delay_s": 0.0,
                "energy_j": 0.0,
                "arrival_time": start_time,
                "remaining_data_gb": 0.0,
                "slot_phases": [],
                "slot_crossings": 0,
            }

        start_slot = self.topology.absolute_slot(start_time)
        current_time = float(start_time)
        remaining = float(data_gb)
        phases = []
        totals = defaultdict(float)
        for _ in range(self.config.route_horizon_slots):
            absolute_slot = self.topology.absolute_slot(current_time)
            slot_end = (absolute_slot + 1) * self.topology.slot_duration_s
            graph = graph_provider(current_time)
            phase = self._slot_flow(
                graph,
                source,
                target,
                remaining,
                current_time,
                slot_end,
                absolute_slot,
                rate_provider,
                queue_provider,
                reserve,
                commit,
            )
            delivered = min(remaining, phase.get("delivered_data_gb", 0.0))
            if delivered > 1.0e-12:
                phase["absolute_slot"] = absolute_slot
                phase["start_time"] = current_time
                phases.append(phase)
                remaining -= delivered
                totals["transmission_delay_s"] += phase["transmission_delay_s"]
                totals["propagation_delay_s"] += phase["propagation_delay_s"]
                totals["queue_delay_s"] += phase["queue_delay_s"]
                totals["energy_j"] += phase["energy_j"]
                if remaining <= 1.0e-12:
                    arrival = current_time + phase["phase_delay_s"]
                    return {
                        "reachable": True,
                        "route_mode": "cross_slot_min_cost_flow",
                        "source": source,
                        "target": target,
                        "data_gb": data_gb,
                        "delay_s": arrival - start_time,
                        "energy_j": totals["energy_j"],
                        "arrival_time": arrival,
                        "remaining_data_gb": 0.0,
                        "delivered_data_gb": data_gb,
                        "slot_phases": phases,
                        "slot_crossings": max(
                            0, self.topology.absolute_slot(arrival) - start_slot
                        ),
                        "transmission_delay_s": totals["transmission_delay_s"],
                        "propagation_delay_s": totals["propagation_delay_s"],
                        "queue_delay_s": totals["queue_delay_s"],
                    }
            current_time = slot_end

        return {
            "reachable": False,
            "failure_reason": "route_horizon_exceeded",
            "route_mode": "cross_slot_min_cost_flow",
            "source": source,
            "target": target,
            "data_gb": data_gb,
            "delay_s": math.inf,
            "energy_j": math.inf,
            "arrival_time": math.inf,
            "remaining_data_gb": remaining,
            "delivered_data_gb": data_gb - remaining,
            "slot_phases": phases,
        }
