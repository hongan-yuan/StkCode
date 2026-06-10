from __future__ import annotations

import math

from ..domain.request import SFCRequest
from ..domain.service import compute_service_execution
from .execution_agent import CandidateDecision, ServiceExecutionAgent


class NearestReplicaExecutionAgent(ServiceExecutionAgent):
    """Select the deployed replica with the fewest current-slot ISL hops."""

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
        candidates = list(context["microservices"][service_id].replicas)
        candidate_scores: list[dict] = []

        for node_id in candidates:
            hop_distance = self._candidate_hop_distance(
                current_node, node_id, current_time, context
            )
            route = self._cached_route_data(current_node, node_id, data_gb, current_time, context)
            if not route["reachable"]:
                candidate_scores.append(
                    {
                        "node_id": node_id,
                        "reachable": False,
                        "score": math.inf,
                        "hop_distance": hop_distance,
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
            candidate_scores.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": float(hop_distance),
                    "hop_distance": hop_distance,
                    "delay_cost_s": delay_cost,
                    "energy_cost_j": energy_cost,
                    "route": route,
                    "compute": compute,
                }
            )

        reachable = [item for item in candidate_scores if item["reachable"]]
        if not reachable:
            return CandidateDecision(service_id, None, math.inf, None, None, candidate_scores)

        best = min(
            reachable,
            key=lambda item: (
                item["hop_distance"],
                item["route"]["delay_s"],
                item["compute"]["queue_delay_s"],
                item["node_id"],
            ),
        )
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
