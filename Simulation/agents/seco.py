from __future__ import annotations

import math

from ..config import SimulationConfig
from ..domain.request import SFCRequest
from ..domain.service import compute_service_execution
from ..network.routing import route_data
from ..network.topology import slot_from_time
from .execution_agent import CandidateDecision, ServiceExecutionAgent


class SECOGreedyExecutionAgent(ServiceExecutionAgent):
    """SECO adapted processing baseline.

    The SECO paper jointly considers observation scheduling, image splitting,
    routing, and computing. The simulator already fixes request sources,
    destinations, and SFC chains, so this baseline keeps the processing part:
    a queue-aware greedy priority order and a layered-graph style joint choice
    of path and computing node for each service stage.
    """

    def __init__(
        self,
        config: SimulationConfig,
        route_queue_weight: float = 1.0,
        compute_queue_weight: float = 1.0,
        egress_delay_weight: float = 0.35,
        bottleneck_shortage_weight: float = 2.0,
    ):
        super().__init__(config)
        self.route_queue_weight = float(route_queue_weight)
        self.compute_queue_weight = float(compute_queue_weight)
        self.egress_delay_weight = float(egress_delay_weight)
        self.bottleneck_shortage_weight = float(bottleneck_shortage_weight)
        self._slot_plan: dict[tuple[int, int], int] = {}
        self._slot_plan_metadata: dict[tuple[int, int], dict] = {}
        self._temp_node_queue_s: dict[tuple[int, int], float] = {}
        self._temp_edge_queue_s: dict[tuple[int, int, int], float] = {}

    def plan_slot_requests(self, requests: list[SFCRequest], context: dict) -> None:
        self._slot_plan = {}
        self._slot_plan_metadata = {}
        self._temp_node_queue_s = {}
        self._temp_edge_queue_s = {}

        remaining = sorted(requests, key=lambda item: (item.start_time, item.request_id))
        priority_rank = 0
        while remaining:
            best_index = None
            best_plan = None
            best_score = math.inf
            for index, request in enumerate(remaining):
                plan = self._estimate_request_plan(request, context)
                score = plan["score"]
                if (score, request.start_time, request.request_id) < (
                    best_score,
                    remaining[best_index].start_time if best_index is not None else math.inf,
                    remaining[best_index].request_id if best_index is not None else math.inf,
                ):
                    best_index = index
                    best_plan = plan
                    best_score = score

            if best_index is None or best_plan is None:
                break
            request = remaining.pop(best_index)
            priority_rank += 1
            self._commit_request_plan(request, best_plan, priority_rank)

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

        return self._select_stage_candidate(
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

    def _estimate_request_plan(self, request: SFCRequest, context: dict) -> dict:
        current_node = request.source_node
        current_time = request.start_time
        stages = []
        total_score = 0.0

        for service_index, _service_id in enumerate(request.services):
            data_gb = self._data_for_stage(request, service_index)
            decision = self._select_stage_candidate(
                request,
                service_index,
                current_node,
                current_time,
                data_gb,
                context,
                exact=False,
            )
            if decision.selected_node is None or not decision.route_estimate or not decision.compute_estimate:
                return {"feasible": False, "score": math.inf, "stages": stages}

            metadata = decision.metadata or {}
            total_score += float(metadata.get("seco_stage_score", decision.score))
            stages.append(
                {
                    "service_index": service_index,
                    "node_id": decision.selected_node,
                    "route": decision.route_estimate,
                    "compute": decision.compute_estimate,
                    "metadata": metadata,
                }
            )
            current_node = decision.selected_node
            current_time = float(metadata.get("seco_estimated_finish_s", current_time))

        final_route = self._cached_route_data(
            current_node,
            request.destination_node,
            request.output_data_gb,
            current_time,
            context,
        )
        if not final_route.get("reachable", False):
            total_score = math.inf
        else:
            total_score += (
                float(final_route.get("delay_s", 0.0))
                + self.route_queue_weight * self._route_temp_queue_delay(final_route, context)
            )
        return {"feasible": math.isfinite(total_score), "score": total_score, "stages": stages}

    def _commit_request_plan(
        self,
        request: SFCRequest,
        plan: dict,
        priority_rank: int,
    ) -> None:
        for stage in plan.get("stages", []):
            service_index = int(stage["service_index"])
            key = (request.request_id, service_index)
            self._slot_plan[key] = int(stage["node_id"])
            metadata = dict(stage.get("metadata", {}))
            metadata.update(
                {
                    "seco_greedy_priority": priority_rank,
                    "seco_request_score": plan.get("score", math.inf),
                }
            )
            self._slot_plan_metadata[key] = metadata
            self._add_route_temp_load(stage.get("route", {}))
            self._add_compute_temp_load(stage.get("compute", {}), int(stage["node_id"]))

    def _select_stage_candidate(
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
        candidates = list(context["microservices"][service_id].replicas)
        candidate_scores: list[dict] = []

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
            route_queue_s = self._route_temp_queue_delay(route, context)
            node_queue_s = self._node_temp_queue_delay(node_id, route["arrival_time"], context)
            estimated_finish_s = compute["compute_finish_s"] + node_queue_s
            egress = self._estimate_egress_capacity(
                request, service_index, node_id, estimated_finish_s, context
            )
            egress_delay_s = float(egress.get("egress_delay_s", math.inf))
            if not math.isfinite(egress_delay_s):
                egress_delay_s = self.config.failure_penalty
            bottleneck_shortage = self._route_bottleneck_shortage(route, data_gb)
            stage_delay_s = (
                route["delay_s"]
                + compute["queue_delay_s"]
                + compute["compute_delay_s"]
            )
            score = (
                stage_delay_s
                + self.route_queue_weight * route_queue_s
                + self.compute_queue_weight * node_queue_s
                + self.egress_delay_weight * egress_delay_s
                + self.bottleneck_shortage_weight * bottleneck_shortage
            )
            candidate_scores.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": score,
                    "seco_stage_delay_s": stage_delay_s,
                    "seco_route_queue_s": route_queue_s,
                    "seco_compute_queue_s": node_queue_s,
                    "seco_estimated_finish_s": estimated_finish_s,
                    "seco_bottleneck_shortage": bottleneck_shortage,
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
                item["seco_estimated_finish_s"],
                item["seco_route_queue_s"] + item["seco_compute_queue_s"],
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
            "seco_stage_score": best["score"],
            "seco_stage_delay_s": best["seco_stage_delay_s"],
            "seco_route_queue_s": best["seco_route_queue_s"],
            "seco_compute_queue_s": best["seco_compute_queue_s"],
            "seco_estimated_finish_s": best["seco_estimated_finish_s"],
            "seco_bottleneck_shortage": best["seco_bottleneck_shortage"],
        }
        if (request.request_id, service_index) in self._slot_plan_metadata:
            metadata.update(self._slot_plan_metadata[(request.request_id, service_index)])
        return CandidateDecision(
            service_id=service_id,
            selected_node=best["node_id"],
            score=best["score"],
            route_estimate=selected_route,
            compute_estimate=selected_compute,
            candidate_scores=candidate_scores,
            metadata=metadata,
        )

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

        metadata = dict(self._slot_plan_metadata.get((request.request_id, service_index), {}))
        score = float(metadata.get("seco_stage_score", route["delay_s"] + compute["compute_delay_s"]))
        return CandidateDecision(
            service_id=service_id,
            selected_node=node_id,
            score=score,
            route_estimate=route,
            compute_estimate=compute,
            candidate_scores=[
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": score,
                    "route": route,
                    "compute": compute,
                }
            ],
            metadata=metadata,
        )

    def _data_for_stage(self, request: SFCRequest, service_index: int) -> float:
        if service_index == 0:
            return request.input_data_gb
        return request.data_gb_between_services[service_index - 1]

    def _node_temp_queue_delay(self, node_id: int, arrival_time: float, context: dict) -> float:
        _, slot_mod = slot_from_time(arrival_time, context["slot_duration"], context["slot_count"])
        return float(self._temp_node_queue_s.get((slot_mod, int(node_id)), 0.0))

    def _add_compute_temp_load(self, compute: dict, node_id: int) -> None:
        slot_mod = compute.get("queue_slot_mod")
        if slot_mod is None:
            return
        key = (int(slot_mod), int(node_id))
        self._temp_node_queue_s[key] = self._temp_node_queue_s.get(key, 0.0) + float(
            compute.get("compute_delay_s", 0.0)
        )

    def _route_temp_queue_delay(self, route: dict, context: dict) -> float:
        delay_s = 0.0
        for slot_mod, edge_u, edge_v, _edge_load_s in self._route_edge_loads(route, context):
            delay_s += self._temp_edge_queue_s.get((slot_mod, edge_u, edge_v), 0.0)
        return delay_s

    def _add_route_temp_load(self, route: dict) -> None:
        for slot_mod, edge_u, edge_v, edge_load_s in self._route_edge_loads(route, None):
            key = (slot_mod, edge_u, edge_v)
            self._temp_edge_queue_s[key] = self._temp_edge_queue_s.get(key, 0.0) + edge_load_s

    def _route_edge_loads(
        self, route: dict, context: dict | None
    ) -> list[tuple[int, int, int, float]]:
        loads: list[tuple[int, int, int, float]] = []
        for slot_path in route.get("slot_paths", []) or []:
            path = [int(node) for node in slot_path.get("path", [])]
            if len(path) <= 1:
                continue
            edge_count = max(1, len(path) - 1)
            tx_s = float(slot_path.get("transmission_delay_s", 0.0)) / edge_count
            slot_mod = int(slot_path.get("slot", 0))
            for node_u, node_v in zip(path[:-1], path[1:]):
                edge_u, edge_v = sorted((int(node_u), int(node_v)))
                loads.append((slot_mod, edge_u, edge_v, tx_s))

        if loads or context is None:
            return loads

        path = [int(node) for node in route.get("path", [])]
        if len(path) <= 1:
            return loads
        _, slot_mod = slot_from_time(
            float(route.get("arrival_time", 0.0)) - float(route.get("delay_s", 0.0)),
            context["slot_duration"],
            context["slot_count"],
        )
        tx_s = float(route.get("transmission_delay_s", 0.0)) / max(1, len(path) - 1)
        for node_u, node_v in zip(path[:-1], path[1:]):
            edge_u, edge_v = sorted((int(node_u), int(node_v)))
            loads.append((slot_mod, edge_u, edge_v, tx_s))
        return loads
