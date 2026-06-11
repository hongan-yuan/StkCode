from __future__ import annotations

import math
import random
from dataclasses import asdict, replace

from ..agents.execution_agent import ServiceExecutionAgent, CandidateDecision
from ..agents.migration import ReplicaPlacementMigrationAgent
from ..agents.ppo_gnn_agent import PPOGNNExecutionAgent
from ..config import SimulationConfig
from ..domain.request import SFCRequest, generate_sfc_requests
from ..domain.service import (
    Microservice,
    SatelliteResource,
    compute_service_execution,
    deployment_matrix,
    generate_microservice_catalog,
    generate_satellite_resources,
)
from ..network.queue import (
    generate_link_background_table,
    generate_markov_compute_load_tables,
)
from ..network.routing import build_routing_cache, route_data
from ..network.topology import load_temporal_graphs, slot_from_time


class SimulationEnvironment:
    def __init__(
            self,
            config: SimulationConfig | None = None,
            migration_agent: ReplicaPlacementMigrationAgent | None = None,
            initial_microservices: dict[int, Microservice] | None = None,
            initial_satellite_resources: dict[int, SatelliteResource] | None = None,
            auto_generate_requests: bool = True,
            auto_apply_migration: bool = True,
    ):
        self.config = config or SimulationConfig()
        self.rng = random.Random(self.config.random_seed)
        self.migration_agent = migration_agent or ReplicaPlacementMigrationAgent(self.config)
        self.initial_microservices = initial_microservices
        self.initial_satellite_resources = initial_satellite_resources
        self.auto_generate_requests = auto_generate_requests
        self.auto_apply_migration = auto_apply_migration
        self.context: dict = {}
        self.request_pool: list[SFCRequest] = []
        self.requests: list[SFCRequest] = []
        self.migration_actions = []

    def build(self) -> "SimulationEnvironment":
        snapshots, ordered_times, slot_duration, time_utc = load_temporal_graphs(
            self.config.isl_log_csv, self.config
        )
        slot_count = len(ordered_times)
        satellite_resources = self.initial_satellite_resources or generate_satellite_resources(
            self.rng, self.config
        )
        microservices = self.initial_microservices or generate_microservice_catalog(
            self.rng, self.config
        )
        compute_load_tables = generate_markov_compute_load_tables(
            self.rng, slot_count, slot_duration, satellite_resources, self.config
        )
        self.context = {
            "config": self.config,
            "snapshots": snapshots,
            "ordered_times": ordered_times,
            "slot_duration": slot_duration,
            "slot_count": slot_count,
            "time_utc_by_slot": time_utc,
            "satellite_resources": satellite_resources,
            **compute_load_tables,
            "link_background_table": generate_link_background_table(
                self.rng, snapshots, slot_duration, self.config
            ),
            "microservices": microservices,
            "deployment_by_node": deployment_matrix(microservices),
            "routing_cache": build_routing_cache(snapshots, self.config),
            "route_estimate_cache": {},
            "route_estimate_cache_stats": {"hits": 0, "misses": 0},
        }
        if self.auto_generate_requests:
            self.request_pool = generate_sfc_requests(self.rng, self.context)
            self.requests = self._select_requests_to_process(self.request_pool)
            if self.auto_apply_migration:
                self.migration_actions = self.apply_migration(self.request_pool)
        return self

    def _select_requests_to_process(self, request_pool: list[SFCRequest]) -> list[SFCRequest]:
        if not self.config.process_single_request:
            return request_pool
        if not request_pool:
            return []
        if self.config.selected_request_id is None:
            selected = self.rng.choice(request_pool)
        else:
            matches = [
                request
                for request in request_pool
                if request.request_id == self.config.selected_request_id
            ]
            if not matches:
                raise ValueError(
                    f"selected_request_id={self.config.selected_request_id} "
                    "does not exist in the generated request pool."
                )
            selected = matches[0]
        return [
            replace(
                selected,
                start_time=self.config.single_request_start_time,
                start_slot=0,
            )
        ]

    def run(self, agent: ServiceExecutionAgent | None = None) -> dict:
        if not self.context:
            self.build()
        agent = agent or PPOGNNExecutionAgent(self.config)
        results = self.execute_requests(self.requests, agent)
        self.migration_agent.observe_failed_replicas(results)
        if self.auto_apply_migration:
            self.migration_agent.observe_execution_feedback(self.migration_actions, results)
        return {
            "config": asdict(self.config),
            "fast_layer_agent": agent.__class__.__name__,
            "ppo_training_available": bool(getattr(agent, "training_available", False)),
            "slot_count": self.context["slot_count"],
            "slot_duration": self.context["slot_duration"],
            "request_pool_count": len(self.request_pool),
            "selected_request_ids": [request.request_id for request in self.requests],
            "request_count": len(self.requests),
            "migration_actions": [asdict(action) for action in self.migration_actions],
            "bandit_summary": self.migration_agent.summary(),
            "routing_cache_summary": dict(self.context["routing_cache"]["stats"]),
            "route_estimate_cache_summary": dict(
                self.context.get("route_estimate_cache_stats", {})
            ),
            "results": results,
        }

    def execute_requests(
            self,
            requests: list[SFCRequest],
            agent: ServiceExecutionAgent
    ) -> list[dict]:

        return [self.execute_request(request, agent) for request in requests]

    def apply_migration(self, requests: list[SFCRequest]):
        self.migration_actions = self.migration_agent.apply(
            self.context["microservices"], requests, self.context
        )
        self.context["deployment_by_node"] = deployment_matrix(self.context["microservices"])
        return self.migration_actions

    def payload_for_results(
            self,
            requests: list[SFCRequest],
            results: list[dict],
            agent: ServiceExecutionAgent,
            migration_actions=None,
    ) -> dict:
        migration_actions = self.migration_actions if migration_actions is None else migration_actions
        return {
            "config": asdict(self.config),
            "fast_layer_agent": agent.__class__.__name__,
            "ppo_training_available": bool(getattr(agent, "training_available", False)),
            "slot_count": self.context["slot_count"],
            "slot_duration": self.context["slot_duration"],
            "request_pool_count": len(requests),
            "selected_request_ids": [request.request_id for request in requests],
            "request_count": len(requests),
            "migration_actions": [asdict(action) for action in migration_actions],
            "bandit_summary": self.migration_agent.summary(),
            "routing_cache_summary": dict(self.context["routing_cache"]["stats"]),
            "route_estimate_cache_summary": dict(
                self.context.get("route_estimate_cache_stats", {})
            ),
            "results": results,
        }

    def execute_request(self, request: SFCRequest, agent: ServiceExecutionAgent) -> dict:
        current_node = request.source_node
        current_time = request.start_time
        accumulated_delay = 0.0
        accumulated_energy = 0.0
        execution_plan = []
        route_details = []

        # print(f"SimulationEnv ... executing request: {request}")
        for service_index, service_id in enumerate(request.services):
            data_gb = (
                request.input_data_gb if service_index == 0 else request.data_gb_between_services[service_index - 1]
            )
            decision = agent.select_replica(
                request,
                service_index,
                current_node,
                current_time,
                data_gb,
                self.context,
            )
            if decision.selected_node is None:
                if hasattr(agent, "record_failure"):
                    agent.record_failure(self.config.failure_penalty)
                self._finalize_agent_episode(
                    agent, self._reward(accumulated_delay, accumulated_energy, False),
                    len(request.services), False
                )
                return self._failure_result(
                    request,
                    service_index,
                    accumulated_delay,
                    accumulated_energy,
                    execution_plan,
                    route_details,
                    self._decision_failure_reason(decision),
                    self._candidate_failure_routes(
                        decision, current_node, current_time, data_gb
                    ),
                )

            route = decision.route_estimate or route_data(
                current_node, decision.selected_node, data_gb, current_time, self.context
            )
            if not route["reachable"]:
                if hasattr(agent, "record_failure"):
                    agent.record_failure(self.config.failure_penalty)
                self._finalize_agent_episode(
                    agent, self._reward(accumulated_delay, accumulated_energy, False),
                    len(request.services), False
                )
                return self._failure_result(
                    request,
                    service_index,
                    accumulated_delay,
                    accumulated_energy,
                    execution_plan,
                    route_details,
                    route.get("failure_reason", "route_failed"),
                    route,
                )

            compute = decision.compute_estimate or compute_service_execution(
                service_id, decision.selected_node, route["arrival_time"], self.context
            )
            route_details.append(
                self._route_record(
                    request,
                    service_index,
                    current_node,
                    decision.selected_node,
                    data_gb,
                    route,
                )
            )
            step_delay = route["delay_s"] + compute["queue_delay_s"] + compute["compute_delay_s"]
            step_energy = route["communication_energy_j"] + compute["compute_energy_j"]
            # Previous per-step PPO reward design:
            # step_reward = -(
            #         self.config.delay_weight * step_delay
            #         + self.config.energy_weight * step_energy / 1000.0
            #         + self.config.slot_switch_penalty_weight * route["slot_crossings"]
            #         + self.config.route_failure_risk_weight
            #         * float(route.get("route_failure_risk", 0.0))
            # )
            step_reward = -(
                    self.config.delay_weight * step_delay
                    + self.config.energy_weight * step_energy / 1000.0
                    + self.config.slot_switch_penalty_weight * route["slot_crossings"]
            )
            if hasattr(agent, "record_step_outcome"):
                agent.record_step_outcome(
                    decision,
                    step_reward,
                    done=service_index == len(request.services) - 1,
                )
            execution_plan.append(
                {
                    "request_id": request.request_id,
                    "stage": service_index,
                    "service_id": service_id,
                    "satellite_node": decision.selected_node,
                    "arrival_time_s": compute["arrival_time_s"],
                    "queue_delay_s": compute["queue_delay_s"],
                    "compute_start_s": compute["compute_start_s"],
                    "compute_finish_s": compute["compute_finish_s"],
                    "compute_delay_s": compute["compute_delay_s"],
                    "compute_energy_j": compute["compute_energy_j"],
                    "candidate_score": decision.score,
                }
            )
            accumulated_energy += step_energy
            current_time = compute["compute_finish_s"]
            accumulated_delay = current_time - request.start_time
            current_node = decision.selected_node

        route_to_destination = getattr(agent, "route_to_destination", None)
        if callable(route_to_destination):
            final_route = route_to_destination(
                request,
                current_node,
                current_time,
                request.output_data_gb,
                self.context,
            )
        else:
            final_route = route_data(
                current_node,
                request.destination_node,
                request.output_data_gb,
                current_time,
                self.context,
            )
        if not final_route["reachable"]:
            self._finalize_agent_episode(
                agent, self._reward(accumulated_delay, accumulated_energy, False),
                len(request.services), False
            )
            return self._failure_result(
                request,
                len(request.services),
                accumulated_delay,
                accumulated_energy,
                execution_plan,
                route_details,
                final_route.get("failure_reason", "destination_route_failed"),
                final_route,
            )

        route_details.append(
            self._route_record(
                request,
                "destination",
                current_node,
                request.destination_node,
                request.output_data_gb,
                final_route,
            )
        )
        accumulated_energy += final_route["communication_energy_j"]
        finish_time = final_route["arrival_time"]
        finish_abs_slot, finish_slot = slot_from_time(
            finish_time, self.context["slot_duration"], self.context["slot_count"]
        )
        terminal_reward = self._reward(
            finish_time - request.start_time, accumulated_energy, True
        )
        self._finalize_agent_episode(
            agent, terminal_reward, len(request.services), True
        )
        return {
            "feasible": True,
            "request": asdict(request),
            "failed_stage": None,
            "finish_time_s": finish_time,
            "finish_abs_slot": finish_abs_slot,
            "finish_slot": finish_slot,
            "total_delay_s": finish_time - request.start_time,
            "total_energy_j": accumulated_energy,
            "reward": terminal_reward,
            "execution_plan": execution_plan,
            "route_details": route_details,
        }

    def _failure_result(
            self,
            request,
            failed_stage,
            accumulated_delay,
            accumulated_energy,
            execution_plan,
            route_details,
            failure_reason="unknown_failure",
            failed_route=None,
    ) -> dict:
        return {
            "feasible": False,
            "request": asdict(request),
            "failed_stage": failed_stage,
            "failure_reason": failure_reason,
            "failed_route": failed_route,
            "finish_time_s": math.inf,
            "finish_abs_slot": "",
            "finish_slot": "",
            "total_delay_s": math.inf,
            "total_energy_j": math.inf,
            "reward": self._reward(accumulated_delay, accumulated_energy, False),
            "execution_plan": execution_plan,
            "route_details": route_details,
        }

    def _finalize_agent_episode(
            self,
            agent: ServiceExecutionAgent,
            terminal_reward: float,
            chain_length: int,
            success: bool,
    ) -> None:
        if hasattr(agent, "record_terminal_outcome"):
            agent.record_terminal_outcome(
                terminal_reward,
                chain_length=chain_length,
                success=success,
            )

    def _decision_failure_reason(self, decision: CandidateDecision) -> str:
        if not decision.candidate_scores:
            return "no_candidate_replicas"
        reasons = {}
        for candidate in decision.candidate_scores:
            reason = candidate.get("failure_reason", "unreachable_candidate")
            reasons[reason] = reasons.get(reason, 0) + 1
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items())
        )
        return f"all_candidate_replicas_unreachable({detail})"

    def _candidate_failure_routes(
            self,
            decision: CandidateDecision,
            current_node: int,
            current_time: float,
            data_gb: float,
    ) -> dict:
        return {
            "type": "candidate_routes",
            "service_id": decision.service_id,
            "current_node": current_node,
            "current_time": current_time,
            "data_gb": data_gb,
            "candidates": [
                {
                    "node_id": candidate.get("node_id"),
                    "reachable": candidate.get("reachable", False),
                    "failure_reason": candidate.get("failure_reason"),
                    "route": candidate.get("route"),
                }
                for candidate in decision.candidate_scores
            ],
        }

    def _reward(self, delay_s: float, energy_j: float, success: bool) -> float:
        if not success:
            return -self.config.failure_penalty
        return -(
                self.config.delay_weight * delay_s
                + self.config.energy_weight * energy_j / 1000.0
        )

    def _route_record(self, request, stage, source, target, data_gb, route) -> dict:
        return {
            "request_id": request.request_id,
            "stage": stage,
            "source_node": source,
            "target_node": target,
            "data_gb": data_gb,
            "route_mode": route.get("route_mode", ""),
            "communication_delay_s": route["delay_s"],
            "transmission_delay_s": route["transmission_delay_s"],
            "propagation_delay_s": route["propagation_delay_s"],
            "communication_energy_j": route["communication_energy_j"],
            "slot_crossings": route["slot_crossings"],
            "arrival_time_s": route["arrival_time"],
            "path": route["path"],
            "slot_paths": route["slot_paths"],
        }
