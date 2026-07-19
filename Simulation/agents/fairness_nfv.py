from __future__ import annotations

import math

from ..config import SimulationConfig
from ..domain.request import SFCRequest
from ..domain.service import compute_service_execution
from ..network.routing import route_data
from .execution_agent import CandidateDecision, ServiceExecutionAgent


class FairnessAwareGreedyVNFExecutionAgent(ServiceExecutionAgent):
    """Fair-NFV adapted baseline.

    This implements a lightweight adaptation of FAGD_MASC from the
    Fair-NFV paper. For each time slot, it greedily builds a VNF mapping
    plan for active SFC requests. Each stage selects a deployed replica and a
    shortest/least-cost route while minimizing the worst elapsed delay among
    requests already planned in the slot. No synthetic per-request time bound
    is imposed in this adaptation.
    """

    def __init__(
        self,
        config: SimulationConfig,
        egress_shortage_weight: float = 2.0,
    ):
        super().__init__(config)
        self.egress_shortage_weight = float(egress_shortage_weight)
        self._slot_plan: dict[tuple[int, int], int] = {}
        self._slot_plan_metadata: dict[tuple[int, int], dict] = {}
        self._slot_request_delays: dict[int, float] = {}

    def _elapsed_delay(self, request: SFCRequest, finish_time_s: float) -> float:
        if not math.isfinite(finish_time_s):
            return math.inf
        return max(0.0, finish_time_s - request.start_time)

    def _existing_worst_delay(self, exclude_request_id: int | None = None) -> float:
        values = [
            delay
            for request_id, delay in self._slot_request_delays.items()
            if exclude_request_id is None or request_id != exclude_request_id
        ]
        return max(values) if values else 0.0

    def _data_for_stage(self, request: SFCRequest, service_index: int) -> float:
        if service_index == 0:
            return request.input_data_gb
        return request.data_gb_between_services[service_index - 1]

    def plan_slot_requests(self, requests: list[SFCRequest], context: dict) -> None:
        self._slot_plan = {}
        self._slot_plan_metadata = {}
        self._slot_request_delays = {}

        for request in sorted(requests, key=lambda item: (item.start_time, item.request_id)):
            current_node = request.source_node
            current_time = request.start_time
            request_failed = False

            for service_index, service_id in enumerate(request.services):
                data_gb = self._data_for_stage(request, service_index)
                selected = self._select_fairness_candidate(
                    request,
                    service_index,
                    current_node,
                    current_time,
                    data_gb,
                    context,
                    exact=False,
                )
                if selected.selected_node is None or not selected.route_estimate:
                    request_failed = True
                    break
                compute = selected.compute_estimate
                if compute is None:
                    request_failed = True
                    break

                key = (request.request_id, service_index)
                self._slot_plan[key] = selected.selected_node
                self._slot_plan_metadata[key] = selected.metadata or {}
                current_node = selected.selected_node
                current_time = compute["compute_finish_s"]
                self._slot_request_delays[request.request_id] = self._elapsed_delay(
                    request, current_time
                )

            if request_failed:
                self._slot_request_delays[request.request_id] = math.inf
                continue

            final_route = route_data(
                current_node,
                request.destination_node,
                request.output_data_gb,
                current_time,
                context,
            )
            finish_time = (
                final_route["arrival_time"] if final_route.get("reachable", False) else math.inf
            )
            self._slot_request_delays[request.request_id] = self._elapsed_delay(
                request, finish_time
            )

    def select_replica(
        self,
        request: SFCRequest,
        service_index: int,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> CandidateDecision:
        key = (request.request_id, service_index)
        planned_node = self._slot_plan.get(key)
        if planned_node is not None:
            decision = self._decision_for_node(
                request,
                service_index,
                planned_node,
                current_node,
                current_time,
                data_gb,
                context,
            )
            if decision.selected_node is not None:
                return decision

        return self._select_fairness_candidate(
            request,
            service_index,
            current_node,
            current_time,
            data_gb,
            context,
            exact=True,
        )

    def route_to_destination(
        self,
        request: SFCRequest,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> dict:
        return route_data(current_node, request.destination_node, data_gb, current_time, context)

    def _decision_for_node(
        self,
        request: SFCRequest,
        service_index: int,
        node_id: int,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> CandidateDecision:
        service_id = request.services[service_index]
        if node_id not in context["microservices"][service_id].replicas:
            return CandidateDecision(service_id, None, math.inf, None, None, [])

        route, compute = self._exact_execution_estimates(
            service_id, current_node, node_id, data_gb, current_time, context
        )
        if not route.get("reachable", False) or compute is None:
            return CandidateDecision(
                service_id,
                None,
                math.inf,
                None,
                None,
                [
                    {
                        "node_id": node_id,
                        "reachable": False,
                        "failure_reason": route.get("failure_reason", "route_failed"),
                        "route": route,
                    }
                ],
            )
        finish_time = compute["compute_finish_s"]
        elapsed_delay = self._elapsed_delay(request, finish_time)
        metadata = dict(self._slot_plan_metadata.get((request.request_id, service_index), {}))
        metadata.update(
            {
                "fairness_elapsed_delay_s": elapsed_delay,
                "fairness_worst_delay_s": max(
                    self._existing_worst_delay(request.request_id), elapsed_delay
                ),
            }
        )
        return CandidateDecision(
            service_id=service_id,
            selected_node=node_id,
            score=metadata["fairness_worst_delay_s"],
            route_estimate=route,
            compute_estimate=compute,
            candidate_scores=[
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": metadata["fairness_worst_delay_s"],
                    "elapsed_delay_s": elapsed_delay,
                    "route": route,
                    "compute": compute,
                }
            ],
            metadata=metadata,
        )

    def _select_fairness_candidate(
        self,
        request: SFCRequest,
        service_index: int,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
        *,
        exact: bool,
    ) -> CandidateDecision:
        service_id = request.services[service_index]
        candidates = self._select_candidate_subset(
            context["microservices"][service_id].replicas,
            current_node,
            current_time,
            context,
        )
        candidate_scores: list[dict] = []
        existing_worst = self._existing_worst_delay(request.request_id)

        for node_id in candidates:
            route = (
                route_data(current_node, node_id, data_gb, current_time, context)
                if exact
                else self._cached_route_data(current_node, node_id, data_gb, current_time, context)
            )
            if not route.get("reachable", False):
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
            stage_finish = compute["compute_finish_s"]
            stage_delay = self._elapsed_delay(request, stage_finish)
            egress = self._estimate_egress_capacity(
                request, service_index, node_id, stage_finish, context
            )
            egress_delay = float(egress.get("egress_delay_s", math.inf))
            egress_shortage = float(egress.get("egress_bottleneck_shortage", 1.0))
            if math.isfinite(egress_delay):
                lookahead_finish = stage_finish + egress_delay
            else:
                lookahead_finish = math.inf
            lookahead_delay = self._elapsed_delay(request, lookahead_finish)
            fairness_score = max(existing_worst, lookahead_delay) + (
                self.egress_shortage_weight * egress_shortage
            )
            delay_cost = route["delay_s"] + compute["queue_delay_s"] + compute["compute_delay_s"]
            energy_cost = route["communication_energy_j"] + compute["compute_energy_j"]
            candidate_scores.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": fairness_score,
                    "stage_elapsed_delay_s": stage_delay,
                    "lookahead_elapsed_delay_s": lookahead_delay,
                    "existing_worst_delay_s": existing_worst,
                    "delay_cost_s": delay_cost,
                    "energy_cost_j": energy_cost,
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

        best = min(
            reachable,
            key=lambda item: (
                item["score"],
                item["lookahead_elapsed_delay_s"],
                item["stage_elapsed_delay_s"],
                item["delay_cost_s"],
                item["node_id"],
            ),
        )
        selected_route = best["route"]
        selected_compute = best["compute"]
        if exact:
            selected_route, selected_compute = self._exact_execution_estimates(
                service_id, current_node, best["node_id"], data_gb, current_time, context
            )
        metadata = {
            "fairness_score": best["score"],
            "fairness_stage_elapsed_delay_s": best["stage_elapsed_delay_s"],
            "fairness_lookahead_elapsed_delay_s": best["lookahead_elapsed_delay_s"],
            "fairness_existing_worst_delay_s": existing_worst,
        }
        return CandidateDecision(
            service_id=service_id,
            selected_node=best["node_id"],
            score=best["score"],
            route_estimate=selected_route,
            compute_estimate=selected_compute,
            candidate_scores=candidate_scores,
            metadata=metadata,
        )
