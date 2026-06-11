from __future__ import annotations

import math
from collections import defaultdict

from ..domain.constellation import orbit_plane, sat_position
from ..domain.request import SFCRequest
from ..domain.service import compute_service_execution
from ..network.service_pressure import ServicePressureBackpressureRouter
from ..network.topology import slot_from_time
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


class ServicePressureExecutionAgent(ServiceExecutionAgent):
    """Service Pressure baseline with local queue-differential routing.

    The baseline keeps an explicit virtual backlog per satellite/service label
    and uses ``ServicePressureBackpressureRouter`` for service-stage and egress
    routes. It intentionally does not call the simulator's min-cost max-flow
    route strategy for request traffic.
    """

    def __init__(
        self,
        config,
        backlog_decay: float = 0.90,
        virtual_backlog_weight: float = 0.25,
    ):
        super().__init__(config)
        self.backlog_decay = max(0.0, min(1.0, float(backlog_decay)))
        self.virtual_backlog_weight = max(0.0, float(virtual_backlog_weight))
        self.virtual_service_backlog: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.backpressure_router = ServicePressureBackpressureRouter(
            config,
            self.virtual_service_backlog,
            virtual_backlog_weight=self.virtual_backlog_weight,
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
        service_id = request.services[service_index]
        candidates = self._select_candidate_subset(
            context["microservices"][service_id].replicas,
            current_node,
            current_time,
            context,
        )
        source_pressure = self._node_service_pressure(
            current_node, service_id, current_time, context
        ) + max(0.0, float(data_gb))
        candidate_scores: list[dict] = []

        for node_id in candidates:
            route = self._backpressure_route(
                current_node,
                node_id,
                data_gb,
                current_time,
                context,
                service_id,
                target_is_compute=True,
            )
            if not route["reachable"]:
                candidate_scores.append(
                    {
                        "node_id": node_id,
                        "reachable": False,
                        "score": math.inf,
                        "pressure_score": -math.inf,
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
            candidate_pressure = self._node_service_pressure(
                node_id, service_id, route["arrival_time"], context
            )
            route_pressure = self._route_backlog_pressure(route, data_gb, context)
            progress_gain = self._downstream_progress_gain(
                request, service_index, current_node, node_id, current_time, context
            )
            egress = self._estimate_egress_capacity(
                request,
                service_index,
                node_id,
                compute["compute_finish_s"],
                context,
            )
            egress_shortage = float(egress["egress_bottleneck_shortage"])

            differential_pressure = source_pressure - candidate_pressure
            pressure_score = (
                differential_pressure
                + 0.75 * progress_gain
                - 1.50 * route_pressure
                - self.config.candidate_egress_shortage_penalty_weight * egress_shortage
                - 0.02 * min(100.0, delay_cost)
                - 0.01 * min(100.0, energy_cost / 1000.0)
            )
            bottleneck_gb = float(route.get("bottleneck_capacity_gb", 0.0))
            bottleneck_shortage = self._route_bottleneck_shortage(route, data_gb)
            candidate_scores.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": -pressure_score,
                    "pressure_score": pressure_score,
                    "source_pressure": source_pressure,
                    "candidate_pressure": candidate_pressure,
                    "differential_pressure": differential_pressure,
                    "route_pressure": route_pressure,
                    "progress_gain": progress_gain,
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

        best = max(
            reachable,
            key=lambda item: (
                item["pressure_score"],
                -item["route_pressure"],
                -item["delay_cost_s"],
                -item["candidate_pressure"],
                -item["node_id"],
            ),
        )
        selected_route = self._backpressure_route(
            current_node,
            best["node_id"],
            data_gb,
            current_time,
            context,
            service_id,
            target_is_compute=True,
        )
        selected_compute = (
            compute_service_execution(
                service_id, best["node_id"], selected_route["arrival_time"], context
            )
            if selected_route["reachable"]
            else None
        )
        self._remember_selection(current_node, best["node_id"], service_id, data_gb)
        return CandidateDecision(
            service_id=service_id,
            selected_node=best["node_id"],
            score=best["score"],
            route_estimate=selected_route,
            compute_estimate=selected_compute,
            candidate_scores=candidate_scores,
            metadata={
                "source_node": current_node,
                "selected_node": best["node_id"],
                "service_id": service_id,
                "data_gb": data_gb,
            },
        )

    def route_to_destination(
        self,
        request: SFCRequest,
        current_node: int,
        current_time: float,
        output_data_gb: float,
        context: dict,
    ) -> dict:
        return self._backpressure_route(
            current_node,
            request.destination_node,
            output_data_gb,
            current_time,
            context,
            service_id=0,
            target_is_compute=False,
        )

    def record_step_outcome(
        self, decision: CandidateDecision, reward: float, done: bool = False
    ) -> None:
        metadata = decision.metadata or {}
        selected_node = metadata.get("selected_node")
        service_id = metadata.get("service_id")
        data_gb = float(metadata.get("data_gb") or 0.0)
        self._decay_virtual_backlog()
        if selected_node is not None and service_id is not None and math.isfinite(reward):
            key = (int(selected_node), int(service_id))
            self.virtual_service_backlog[key] = max(
                0.0,
                self.virtual_service_backlog[key] - max(0.1, data_gb),
            )

    def _remember_selection(
        self,
        source_node: int,
        selected_node: int,
        service_id: int,
        data_gb: float,
    ) -> None:
        demand = max(0.1, float(data_gb))
        self.virtual_service_backlog[(int(source_node), int(service_id))] += demand
        self.virtual_service_backlog[(int(selected_node), int(service_id))] += 0.5 * demand

    def _decay_virtual_backlog(self) -> None:
        if self.backlog_decay >= 1.0:
            return
        stale = []
        for key, value in self.virtual_service_backlog.items():
            next_value = value * self.backlog_decay
            if next_value <= 1.0e-6:
                stale.append(key)
            else:
                self.virtual_service_backlog[key] = next_value
        for key in stale:
            del self.virtual_service_backlog[key]

    def _node_service_pressure(
        self,
        node_id: int,
        service_id: int,
        current_time: float,
        context: dict,
    ) -> float:
        _, slot_mod = slot_from_time(
            current_time, context["slot_duration"], context["slot_count"]
        )
        slot_duration = max(1.0e-9, float(context["slot_duration"]))
        queue_delay = float(
            context.get("queue_delay_table", {}).get(slot_mod, {}).get(node_id, 0.0)
        )
        utilization = float(
            context.get("compute_utilization_table", {}).get(slot_mod, {}).get(node_id, 0.0)
        )
        discount = float(
            context.get("discount_table", {}).get(slot_mod, {}).get(node_id, 1.0)
        )
        deployed = context.get("deployment_by_node", {}).get(node_id, set())
        deployment_pressure = len(deployed) / max(1, self.config.max_services_per_satellite)
        virtual_backlog = self.virtual_service_backlog[(int(node_id), int(service_id))]
        return (
            utilization
            + queue_delay / slot_duration
            + 0.50 * deployment_pressure
            + 0.50 * max(0.0, 1.0 - discount)
            + self.virtual_backlog_weight * virtual_backlog
        )

    def _route_backlog_pressure(self, route: dict, data_gb: float, context: dict) -> float:
        slot_paths = route.get("slot_paths") or []
        edge_rhos = []
        edge_queue_delays = []
        for path in slot_paths:
            for edge in path.get("edges", []):
                edge_rhos.append(float(edge.get("background_rho", 0.0)))
                edge_queue_delays.append(float(edge.get("background_queue_delay_s", 0.0)))
        avg_rho = sum(edge_rhos) / len(edge_rhos) if edge_rhos else 0.0
        avg_queue = sum(edge_queue_delays) / len(edge_queue_delays) if edge_queue_delays else 0.0
        slot_duration = max(1.0e-9, float(context["slot_duration"]))
        bottleneck_shortage = self._route_bottleneck_shortage(route, data_gb)
        horizon = max(1, int(self.config.route_horizon_slots))
        return (
            avg_rho
            + avg_queue / slot_duration
            + bottleneck_shortage
            + float(route.get("route_failure_risk", 0.0))
            + float(route.get("slot_crossings", 0.0)) / horizon
        )

    def _backpressure_route(
        self,
        source: int,
        target: int,
        data_gb: float,
        start_time: float,
        context: dict,
        service_id: int,
        target_is_compute: bool,
    ) -> dict:
        return self.backpressure_router.route(
            source,
            target,
            data_gb,
            start_time,
            context,
            service_id=service_id,
            target_is_compute=target_is_compute,
        )

    def _exact_execution_estimates(
        self,
        service_id: int,
        source_node: int,
        selected_node: int,
        data_gb: float,
        current_time: float,
        context: dict,
    ) -> tuple[dict, dict | None]:
        route = self._backpressure_route(
            source_node,
            selected_node,
            data_gb,
            current_time,
            context,
            service_id,
            target_is_compute=True,
        )
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
            egress_service_id = request.services[service_index + 1]
            target_is_compute = True
        else:
            egress_data_gb = request.output_data_gb
            target_nodes = [request.destination_node]
            egress_type = "destination"
            egress_service_id = 0
            target_is_compute = False

        best: dict | None = None
        failure_reasons: dict[str, int] = {}
        target_nodes = self._select_candidate_subset(
            list(target_nodes), candidate_node, egress_start_time, context
        )
        for target_node in target_nodes:
            route = self._backpressure_route(
                candidate_node,
                target_node,
                egress_data_gb,
                egress_start_time,
                context,
                egress_service_id,
                target_is_compute,
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
                "egress_bottleneck_capacity_gb": float(
                    route.get("bottleneck_capacity_gb", 0.0)
                ),
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

    def _downstream_progress_gain(
        self,
        request: SFCRequest,
        service_index: int,
        current_node: int,
        candidate_node: int,
        current_time: float,
        context: dict,
    ) -> float:
        if service_index + 1 < len(request.services):
            next_service_id = request.services[service_index + 1]
            targets = context["microservices"][next_service_id].replicas
        else:
            targets = [request.destination_node]
        current_distance = self._nearest_hop_distance(
            current_node, targets, current_time, context
        )
        candidate_distance = self._nearest_hop_distance(
            candidate_node, targets, current_time, context
        )
        if not math.isfinite(current_distance) and not math.isfinite(candidate_distance):
            return 0.0
        if not math.isfinite(current_distance):
            current_distance = self.config.total_sats + 1
        if not math.isfinite(candidate_distance):
            candidate_distance = self.config.total_sats + 1
        return (current_distance - candidate_distance) / max(1.0, current_distance)

    def _nearest_hop_distance(
        self,
        source: int,
        targets: list[int],
        current_time: float,
        context: dict,
    ) -> float:
        if source in targets:
            return 0.0
        return min(
            (
                self._candidate_hop_distance(source, target, current_time, context)
                for target in targets
            ),
            default=math.inf,
        )


class SCNFVChainingOrbitExecutionAgent(ServiceExecutionAgent):
    """SC-NFV-inspired chaining-orbit baseline adapted to deployed replicas.

    The original SC-NFV work jointly selects an access satellite, a chaining
    orbit, per-request VNF instances, and routes. This simulator already has
    persistent microservice replicas and a separate Bandit redeployment layer,
    so this baseline keeps the deployment model unchanged:

    * choose one Walker plane as the request's chaining orbit,
    * prefer replicas of every service in that plane,
    * use first-fit-style proximity around the orbit for ordered service stages,
    * fall back to global replicas only when the chosen orbit lacks a service.
    """

    def __init__(
        self,
        config,
        missing_service_penalty: float = 8.0,
        off_orbit_penalty: float = 5.0,
        orbit_distance_weight: float = 0.05,
    ):
        super().__init__(config)
        self.missing_service_penalty = max(0.0, float(missing_service_penalty))
        self.off_orbit_penalty = max(0.0, float(off_orbit_penalty))
        self.orbit_distance_weight = max(0.0, float(orbit_distance_weight))
        self.request_orbit_state: dict[tuple[int, float], dict] = {}

    def select_replica(
        self,
        request: SFCRequest,
        service_index: int,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> CandidateDecision:
        state = self._state_for_request(
            request, service_index, current_node, current_time, data_gb, context
        )
        service_id = request.services[service_index]
        selected_plane = int(state["chaining_plane"])
        service_replicas = list(context["microservices"][service_id].replicas)
        in_orbit = [
            node_id for node_id in service_replicas
            if orbit_plane(node_id, self.config) == selected_plane
        ]
        orbit_service_missing = not in_orbit
        candidate_pool = self._candidate_pool(
            service_replicas,
            in_orbit,
            selected_plane,
            state,
            service_index,
            current_node,
            current_time,
            context,
        )
        candidate_scores: list[dict] = []

        for node_id in candidate_pool:
            route = self._cached_route_data(current_node, node_id, data_gb, current_time, context)
            in_chaining_orbit = orbit_plane(node_id, self.config) == selected_plane
            anchor = self._stage_anchor(state, service_index, current_node)
            orbit_distance = self._orbit_distance(anchor, node_id)
            if not route["reachable"]:
                candidate_scores.append(
                    {
                        "node_id": node_id,
                        "reachable": False,
                        "score": math.inf,
                        "chaining_plane": selected_plane,
                        "in_chaining_orbit": in_chaining_orbit,
                        "orbit_service_missing": orbit_service_missing,
                        "orbit_distance": orbit_distance,
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
            egress = self._estimate_egress_capacity(
                request,
                service_index,
                node_id,
                compute["compute_finish_s"],
                context,
            )
            egress_shortage = float(egress["egress_bottleneck_shortage"])
            bottleneck_shortage = self._route_bottleneck_shortage(route, data_gb)
            off_orbit_cost = (
                0.0
                if in_chaining_orbit or orbit_service_missing
                else self.off_orbit_penalty
            )
            next_missing_penalty = self._next_service_missing_penalty(
                request, service_index, selected_plane, context
            )
            score = (
                self.config.delay_weight * delay_cost
                + self.config.energy_weight * energy_cost / 1000.0
                + self.config.route_failure_risk_weight
                * float(route.get("route_failure_risk", 0.0))
                + self.config.candidate_bottleneck_shortage_penalty_weight
                * bottleneck_shortage
                + self.config.candidate_egress_shortage_penalty_weight
                * egress_shortage
                + off_orbit_cost
                + next_missing_penalty
                + self.orbit_distance_weight * orbit_distance
                + 0.25 * self._node_load_pressure(node_id, route["arrival_time"], context)
            )
            candidate_scores.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": score,
                    "chaining_plane": selected_plane,
                    "connecting_node": state["connecting_node"],
                    "in_chaining_orbit": in_chaining_orbit,
                    "orbit_service_missing": orbit_service_missing,
                    "orbit_distance": orbit_distance,
                    "delay_cost_s": delay_cost,
                    "energy_cost_j": energy_cost,
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

        best = min(
            reachable,
            key=lambda item: (
                item["score"],
                not item["in_chaining_orbit"],
                item["orbit_distance"],
                item["delay_cost_s"],
                item["node_id"],
            ),
        )
        selected_route, selected_compute = self._exact_execution_estimates(
            service_id, current_node, best["node_id"], data_gb, current_time, context
        )
        state.setdefault("stage_nodes", {})[int(service_index)] = int(best["node_id"])
        return CandidateDecision(
            service_id=service_id,
            selected_node=best["node_id"],
            score=best["score"],
            route_estimate=selected_route,
            compute_estimate=selected_compute,
            candidate_scores=candidate_scores,
            metadata={
                "request_key": self._request_key(request),
                "service_index": service_index,
                "chaining_plane": selected_plane,
                "connecting_node": state["connecting_node"],
                "in_chaining_orbit": best["in_chaining_orbit"],
                "orbit_service_missing": best["orbit_service_missing"],
            },
        )

    def record_step_outcome(
        self, decision: CandidateDecision, reward: float, done: bool = False
    ) -> None:
        if not done:
            return
        metadata = decision.metadata or {}
        request_key = metadata.get("request_key")
        if isinstance(request_key, tuple):
            self.request_orbit_state.pop(request_key, None)

    def record_terminal_outcome(
        self,
        terminal_reward: float,
        chain_length: int | None = None,
        success: bool = True,
    ) -> None:
        self._trim_request_state_cache()

    def _state_for_request(
        self,
        request: SFCRequest,
        service_index: int,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> dict:
        key = self._request_key(request)
        self._trim_request_state_cache()
        if service_index == 0 or key not in self.request_orbit_state:
            self.request_orbit_state[key] = self._select_chaining_orbit(
                request, current_node, current_time, data_gb, context
            )
        return self.request_orbit_state[key]

    def _trim_request_state_cache(self) -> None:
        max_states = 1024
        if len(self.request_orbit_state) <= max_states:
            return
        overflow = len(self.request_orbit_state) - max_states
        for key in list(self.request_orbit_state)[:overflow]:
            self.request_orbit_state.pop(key, None)

    def _request_key(self, request: SFCRequest) -> tuple[int, float]:
        return (int(request.request_id), round(float(request.start_time), 9))

    def _plane_nodes(self, plane: int) -> list[int]:
        start = int(plane) * self.config.sats_per_plane + 1
        return list(range(start, start + self.config.sats_per_plane))

    def _select_chaining_orbit(
        self,
        request: SFCRequest,
        current_node: int,
        current_time: float,
        data_gb: float,
        context: dict,
    ) -> dict:
        orbit_scores = []
        for plane in range(self.config.num_planes):
            plane_nodes = self._plane_nodes(plane)
            coverage_count = 0
            replica_count = 0
            for service_id in request.services:
                replicas = context["microservices"][service_id].replicas
                in_plane_count = sum(
                    1 for node_id in replicas
                    if orbit_plane(node_id, self.config) == plane
                )
                if in_plane_count:
                    coverage_count += 1
                    replica_count += in_plane_count

            missing_count = len(request.services) - coverage_count
            connecting = self._select_connecting_node(
                request, current_node, current_time, data_gb, plane_nodes, context
            )
            chain_cost = self._estimate_orbit_chain_cost(
                request, connecting["node_id"], plane, current_time, context
            )
            avg_load = sum(
                self._node_load_pressure(node_id, current_time, context)
                for node_id in plane_nodes
            ) / max(1, len(plane_nodes))
            score = (
                connecting["score"]
                + chain_cost
                + self.missing_service_penalty * missing_count
                + 0.75 * avg_load
                - 0.10 * replica_count
            )
            orbit_scores.append(
                {
                    "chaining_plane": plane,
                    "connecting_node": connecting["node_id"],
                    "score": score,
                    "coverage_count": coverage_count,
                    "missing_count": missing_count,
                    "replica_count": replica_count,
                    "connection_score": connecting["score"],
                    "estimated_chain_cost": chain_cost,
                    "average_orbit_load": avg_load,
                }
            )

        best = min(
            orbit_scores,
            key=lambda item: (
                item["score"],
                item["missing_count"],
                -item["coverage_count"],
                item["chaining_plane"],
            ),
        )
        return {
            "chaining_plane": int(best["chaining_plane"]),
            "connecting_node": int(best["connecting_node"]),
            "orbit_scores": orbit_scores,
            "stage_nodes": {},
        }

    def _select_connecting_node(
        self,
        request: SFCRequest,
        current_node: int,
        current_time: float,
        data_gb: float,
        plane_nodes: list[int],
        context: dict,
    ) -> dict:
        best: dict | None = None
        for node_id in self._connecting_candidates(
            request, current_node, current_time, plane_nodes, context
        ):
            ingress = self._cached_route_data(
                current_node, node_id, data_gb, current_time, context
            )
            egress = self._cached_route_data(
                node_id,
                request.destination_node,
                request.output_data_gb,
                current_time,
                context,
            )
            score = self._route_cost(ingress, data_gb) + self._route_cost(
                egress, request.output_data_gb
            )
            score += 0.50 * self._node_load_pressure(node_id, current_time, context)
            item = {"node_id": node_id, "score": score}
            if best is None or (item["score"], item["node_id"]) < (
                best["score"],
                best["node_id"],
            ):
                best = item
        return best or {"node_id": plane_nodes[0], "score": math.inf}

    def _connecting_candidates(
        self,
        request: SFCRequest,
        current_node: int,
        current_time: float,
        plane_nodes: list[int],
        context: dict,
    ) -> list[int]:
        max_candidates = max(2, int(self.config.max_candidate_replicas))
        hosting_nodes = {
            int(node_id)
            for service_id in request.services
            for node_id in context["microservices"][service_id].replicas
            if node_id in plane_nodes
        }
        primary = sorted(
            hosting_nodes or set(plane_nodes),
            key=lambda node_id: (
                self._candidate_hop_distance(current_node, node_id, current_time, context)
                + self._candidate_hop_distance(
                    node_id, request.destination_node, current_time, context
                ),
                self._node_load_pressure(node_id, current_time, context),
                node_id,
            ),
        )[:max_candidates]
        backup = sorted(
            plane_nodes,
            key=lambda node_id: (
                self._candidate_hop_distance(current_node, node_id, current_time, context),
                self._node_load_pressure(node_id, current_time, context),
                node_id,
            ),
        )[:2]
        candidates = []
        seen = set()
        for node_id in [*primary, *backup]:
            if node_id in seen:
                continue
            seen.add(node_id)
            candidates.append(node_id)
        return candidates

    def _estimate_orbit_chain_cost(
        self,
        request: SFCRequest,
        connecting_node: int,
        plane: int,
        current_time: float,
        context: dict,
    ) -> float:
        anchor = connecting_node
        total = 0.0
        for service_id in request.services:
            replicas = [
                node_id for node_id in context["microservices"][service_id].replicas
                if orbit_plane(node_id, self.config) == plane
            ]
            if not replicas:
                total += self.missing_service_penalty
                continue
            node_id = min(
                replicas,
                key=lambda item: (
                    self._orbit_distance(anchor, item),
                    self._node_load_pressure(item, current_time, context),
                    item,
                ),
            )
            compute = compute_service_execution(service_id, node_id, current_time, context)
            total += (
                compute["queue_delay_s"]
                + compute["compute_delay_s"]
                + self.orbit_distance_weight * self._orbit_distance(anchor, node_id)
            )
            current_time = compute["compute_finish_s"]
            anchor = node_id
        return total

    def _candidate_pool(
        self,
        service_replicas: list[int],
        in_orbit: list[int],
        selected_plane: int,
        state: dict,
        service_index: int,
        current_node: int,
        current_time: float,
        context: dict,
    ) -> list[int]:
        max_candidates = max(1, int(self.config.max_candidate_replicas))
        anchor = self._stage_anchor(state, service_index, current_node)

        def rank(node_id: int) -> tuple[float, float, int]:
            return (
                self._orbit_distance(anchor, node_id),
                self._node_load_pressure(node_id, current_time, context),
                int(node_id),
            )

        pool: list[int] = []
        if in_orbit:
            pool.extend(sorted(in_orbit, key=rank)[:max_candidates])
            fallback = [
                node_id for node_id in service_replicas
                if orbit_plane(node_id, self.config) != selected_plane
            ]
            pool.extend(sorted(fallback, key=rank)[:max(1, max_candidates // 2)])
        else:
            pool.extend(sorted(service_replicas, key=rank)[:max_candidates])
        seen = set()
        unique_pool = []
        for node_id in pool:
            if node_id in seen:
                continue
            seen.add(node_id)
            unique_pool.append(node_id)
        return unique_pool

    def _stage_anchor(self, state: dict, service_index: int, current_node: int) -> int:
        if service_index > 0:
            previous = state.get("stage_nodes", {}).get(int(service_index) - 1)
            if previous is not None:
                return int(previous)
        return int(state.get("connecting_node", current_node))

    def _next_service_missing_penalty(
        self,
        request: SFCRequest,
        service_index: int,
        selected_plane: int,
        context: dict,
    ) -> float:
        if service_index + 1 >= len(request.services):
            return 0.0
        next_service_id = request.services[service_index + 1]
        has_next = any(
            orbit_plane(node_id, self.config) == selected_plane
            for node_id in context["microservices"][next_service_id].replicas
        )
        return 0.0 if has_next else 0.25 * self.missing_service_penalty

    def _orbit_distance(self, node_a: int, node_b: int) -> float:
        if orbit_plane(node_a, self.config) != orbit_plane(node_b, self.config):
            return float(self.config.sats_per_plane)
        pos_a = sat_position(node_a, self.config)
        pos_b = sat_position(node_b, self.config)
        direct = abs(pos_a - pos_b)
        return float(min(direct, self.config.sats_per_plane - direct))

    def _node_load_pressure(self, node_id: int, current_time: float, context: dict) -> float:
        _, slot_mod = slot_from_time(
            current_time, context["slot_duration"], context["slot_count"]
        )
        slot_duration = max(1.0e-9, float(context["slot_duration"]))
        queue_delay = float(
            context.get("queue_delay_table", {}).get(slot_mod, {}).get(node_id, 0.0)
        )
        utilization = float(
            context.get("compute_utilization_table", {}).get(slot_mod, {}).get(node_id, 0.0)
        )
        discount = float(
            context.get("discount_table", {}).get(slot_mod, {}).get(node_id, 1.0)
        )
        deployed = context.get("deployment_by_node", {}).get(node_id, set())
        deployment_load = len(deployed) / max(1, self.config.max_services_per_satellite)
        return (
            utilization
            + queue_delay / slot_duration
            + 0.50 * max(0.0, 1.0 - discount)
            + 0.25 * deployment_load
        )

    def _route_cost(self, route: dict, data_gb: float) -> float:
        if not route.get("reachable", False):
            return 1.0e6
        return (
            float(route.get("delay_s", math.inf))
            + self.config.route_failure_risk_weight
            * float(route.get("route_failure_risk", 0.0))
            + self.config.candidate_bottleneck_shortage_penalty_weight
            * self._route_bottleneck_shortage(route, data_gb)
            + float(route.get("slot_crossings", 0.0)) * self.config.switch_penalty_s
        )
