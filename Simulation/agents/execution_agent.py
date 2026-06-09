from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import SimulationConfig
from ..domain.request import SFCRequest
from ..domain.service import compute_service_execution
from ..network.routing import route_data
from ..network.topology import slot_from_time


@dataclass
class CandidateDecision:
    service_id: int
    selected_node: int | None
    score: float
    route_estimate: dict | None
    compute_estimate: dict | None
    candidate_scores: list[dict]
    metadata: dict | None = None


class ServiceExecutionAgent:
    """Fast-layer placeholder agent.

    This implements the candidate-scoring interface that can later be replaced
    by PPO + GNN. The score is a normalized weighted sum of estimated route
    delay, compute delay, and energy for each valid replica.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def _route_bottleneck_shortage(self, route: dict, data_gb: float) -> float:
        if data_gb <= 1.0e-9:
            return 0.0
        bottleneck_gb = float(route.get("bottleneck_capacity_gb", 0.0))
        return max(0.0, 1.0 - min(1.0, bottleneck_gb / data_gb))

    def _bucket_float(self, value: float, bucket_size: float) -> float:
        bucket_size = float(bucket_size)
        if bucket_size <= 0.0:
            return round(float(value), 9)
        return round(round(float(value) / bucket_size) * bucket_size, 9)

    def _cached_route_data(
        self,
        source: int,
        target: int,
        data_gb: float,
        start_time: float,
        context: dict,
    ) -> dict:
        if not self.config.route_estimate_cache_enabled:
            return route_data(source, target, data_gb, start_time, context)
        cache = context.setdefault("route_estimate_cache", {})
        stats = context.setdefault(
            "route_estimate_cache_stats", {"hits": 0, "misses": 0}
        )
        _, slot_mod = slot_from_time(
            start_time, context["slot_duration"], context["slot_count"]
        )
        key = (
            int(slot_mod),
            int(source),
            int(target),
            self._bucket_float(data_gb, self.config.route_estimate_data_bucket_gb),
            self._bucket_float(start_time, self.config.route_estimate_time_bucket_s),
        )
        if key in cache:
            stats["hits"] += 1
            return cache[key]
        stats["misses"] += 1
        route = route_data(source, target, data_gb, start_time, context)
        cache[key] = route
        return route

    def _candidate_hop_distance(
        self,
        source: int,
        target: int,
        current_time: float,
        context: dict,
    ) -> int:
        if source == target:
            return 0
        _, slot_mod = slot_from_time(
            current_time, context["slot_duration"], context["slot_count"]
        )
        graph = context["snapshots"][slot_mod]
        visited = {source}
        frontier = [(source, 0)]
        while frontier:
            node_id, distance = frontier.pop(0)
            for neighbor in graph.neighbors(node_id):
                if neighbor == target:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, distance + 1))
        return self.config.total_sats + 1

    def _candidate_queue_delay(
        self,
        node_id: int,
        current_time: float,
        context: dict,
    ) -> float:
        _, slot_mod = slot_from_time(
            current_time, context["slot_duration"], context["slot_count"]
        )
        table = context.get("queue_delay_table", {})
        slot_table = table.get(slot_mod, {}) if isinstance(table, dict) else {}
        return float(slot_table.get(node_id, 0.0))

    def _select_candidate_subset(
        self,
        candidates: list[int],
        current_node: int,
        current_time: float,
        context: dict,
    ) -> list[int]:
        max_candidates = int(self.config.max_candidate_replicas)
        if max_candidates <= 0 or len(candidates) <= max_candidates:
            return list(candidates)
        ranked = sorted(
            candidates,
            key=lambda node_id: (
                self._candidate_hop_distance(current_node, node_id, current_time, context),
                self._candidate_queue_delay(node_id, current_time, context),
                node_id,
            ),
        )
        return ranked[:max_candidates]

    def _exact_execution_estimates(
        self,
        service_id: int,
        source_node: int,
        selected_node: int,
        data_gb: float,
        current_time: float,
        context: dict,
    ) -> tuple[dict, dict | None]:
        route = route_data(source_node, selected_node, data_gb, current_time, context)
        if not route["reachable"]:
            return route, None
        compute = compute_service_execution(
            service_id, selected_node, route["arrival_time"], context
        )
        return route, compute

    def _estimate_egress_capacity(
        self,
        request: SFCRequest,
        service_index: int,
        candidate_node: int,
        egress_start_time: float,
        context: dict,
    ) -> dict:
        if service_index + 1 < len(request.services):
            egress_data_gb = request.data_gb_between_services[service_index]
            target_nodes = context["microservices"][request.services[service_index + 1]].replicas
            egress_type = "next_service"
        else:
            egress_data_gb = request.output_data_gb
            target_nodes = [request.destination_node]
            egress_type = "destination"

        best: dict | None = None
        failure_reasons: dict[str, int] = {}
        target_nodes = self._select_candidate_subset(
            list(target_nodes), candidate_node, egress_start_time, context
        )
        for target_node in target_nodes:
            route = self._cached_route_data(
                candidate_node, target_node, egress_data_gb, egress_start_time, context
            )
            if not route["reachable"]:
                reason = route.get("failure_reason", "route_failed")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                continue
            shortage = self._route_bottleneck_shortage(route, egress_data_gb)
            item = {
                "egress_type": egress_type,
                "egress_target_node": target_node,
                "egress_data_gb": egress_data_gb,
                "egress_reachable": True,
                "egress_bottleneck_capacity_gb": float(route.get("bottleneck_capacity_gb", 0.0)),
                "egress_bottleneck_shortage": shortage,
                "egress_delay_s": route["delay_s"],
                "egress_route": route,
            }
            if best is None or (
                item["egress_bottleneck_shortage"],
                item["egress_delay_s"],
            ) < (
                best["egress_bottleneck_shortage"],
                best["egress_delay_s"],
            ):
                best = item

        if best is not None:
            return best
        return {
            "egress_type": egress_type,
            "egress_target_node": None,
            "egress_data_gb": egress_data_gb,
            "egress_reachable": False,
            "egress_bottleneck_capacity_gb": 0.0,
            "egress_bottleneck_shortage": 1.0 if egress_data_gb > 1.0e-9 else 0.0,
            "egress_delay_s": math.inf,
            "egress_failure_reasons": failure_reasons,
        }

    def select_replica(
        self,
        request: SFCRequest,
        service_index: int,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> CandidateDecision:

        service_id = request.services[service_index]
        candidates = self._select_candidate_subset(
            context["microservices"][service_id].replicas,
            current_node,
            current_time,
            context,
        )
        candidate_scores: list[dict] = []

        for node_id in candidates:
            route = self._cached_route_data(current_node, node_id, data_gb, current_time, context)
            if not route["reachable"]:
                candidate_scores.append(
                    {
                        "node_id": node_id,
                        "reachable": False,
                        "score": math.inf,
                        "failure_reason": route.get("failure_reason", "route_failed"),
                        "route": route,
                    }
                )
                continue

            compute = compute_service_execution(
                service_id, node_id, route["arrival_time"], context
            )
            delay_cost = route["delay_s"] + compute["queue_delay_s"] + compute["compute_delay_s"]
            energy_cost = route["communication_energy_j"] + compute["compute_energy_j"]
            slot_penalty = self.config.slot_switch_penalty_weight * route["slot_crossings"]
            score = (
                self.config.delay_weight * delay_cost
                + self.config.energy_weight * energy_cost / 1000.0
                + slot_penalty
                + self.config.route_failure_risk_weight
                * float(route.get("route_failure_risk", 0.0))
            )
            bottleneck_gb = float(route.get("bottleneck_capacity_gb", 0.0))
            bottleneck_shortage = 0.0
            if data_gb > 1.0e-9:
                bottleneck_shortage = max(0.0, 1.0 - min(1.0, bottleneck_gb / data_gb))
            score += self.config.candidate_bottleneck_shortage_penalty_weight * bottleneck_shortage
            egress = self._estimate_egress_capacity(
                request,
                service_index,
                node_id,
                compute["compute_finish_s"],
                context,
            )
            egress_shortage = float(egress["egress_bottleneck_shortage"])
            score += self.config.candidate_egress_shortage_penalty_weight * egress_shortage
            candidate_scores.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": score,
                    "delay_cost_s": delay_cost,
                    "energy_cost_j": energy_cost,
                    "bottleneck_capacity_gb": bottleneck_gb,
                    "bottleneck_shortage": bottleneck_shortage,
                    **{
                        key: value
                        for key, value in egress.items()
                        if key != "egress_route"
                    },
                    "route": route,
                    "compute": compute,
                }
            )

        reachable = [item for item in candidate_scores if item["reachable"]]
        if not reachable:
            return CandidateDecision(service_id, None, math.inf, None, None, candidate_scores)

        best = min(reachable, key=lambda item: item["score"])
        selected_route, selected_compute = self._exact_execution_estimates(
            service_id, current_node, best["node_id"], data_gb, current_time, context
        )
        return CandidateDecision(
            service_id=service_id,
            selected_node=best["node_id"],
            score=best["score"],
            route_estimate=selected_route,
            compute_estimate=selected_compute,
            candidate_scores=candidate_scores,
        )
