from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import ELARAConfig
from .bandit import BanditReplicaAdapter, StageExecutionRecord
from .connector import ConnectorBuilder, RequestSubgraph
from .domain import Microservice, SatelliteResource, ServiceRequest
from .routing import CrossSlotMinCostRouter
from .state import (
    CANDIDATE_FEATURES,
    EDGE_FEATURES,
    NODE_FEATURES,
    REQUEST_FEATURES,
    ServingSelectionState,
    SparseTopologyState,
)
from .topology import ISLEdge, SparseGraph, TemporalTopology


@dataclass
class EpisodeRuntime:
    request: ServiceRequest
    stage: int
    current_node: int
    current_time_s: float
    accumulated_latency_s: float
    accumulated_energy_j: float
    serving_history: list[int]
    subgraph: RequestSubgraph
    done: bool = False


class ELARAEnvironment:
    """Independent request-level environment for serving-satellite PPO.

    A full request is one episode and every microservice choice is one action.
    Routing and computation update link/compute availability before the next
    observation is built.
    """

    def __init__(self, config: ELARAConfig | None = None, topology: TemporalTopology | None = None):
        self.config = config or ELARAConfig()
        self.rng = random.Random(self.config.seed)
        self.topology = topology or TemporalTopology.from_csv(
            self.config.trace_csv,
            self.config.sats_per_plane,
            self.config.total_satellites,
            self.config.max_trace_slots,
            self.config.slot_duration_s,
        )
        self.connector = ConnectorBuilder(self.config.connector_edge_weight)
        self.router = CrossSlotMinCostRouter(self.topology, self.config)
        self.replica_adapter = BanditReplicaAdapter(self.config)
        self.services = self._generate_services()
        self.resources = self._generate_resources()
        self.compute_available_at = {node: 0.0 for node in range(self.config.total_satellites)}
        self.link_available_at: dict[tuple[int, int], float] = {}
        self.runtime: EpisodeRuntime | None = None
        self.request_counter = 0
        self.last_state: ServingSelectionState | None = None
        self.last_migration_actions = []

    def _generate_resources(self) -> dict[int, SatelliteResource]:
        resources = {}
        for node in range(self.config.total_satellites):
            resources[node] = SatelliteResource(
                node_id=node,
                capacity_gflops=self.rng.uniform(
                    self.config.compute_capacity_gflops_min,
                    self.config.compute_capacity_gflops_max,
                ),
                compute_power_w=self.rng.uniform(
                    self.config.compute_power_w_min,
                    self.config.compute_power_w_max,
                ),
                efficiency=self.rng.uniform(0.70, 1.0),
                memory_capacity_gb=self.config.satellite_memory_capacity_gb,
            )
        return resources

    def _generate_services(self) -> dict[int, Microservice]:
        services = {}
        nodes = list(range(self.config.total_satellites))
        replica_count = min(self.config.replicas_per_service, len(nodes))
        for service_id in range(self.config.num_services):
            services[service_id] = Microservice(
                service_id=service_id,
                workload_cycles=self.rng.uniform(
                    self.config.service_cycles_min,
                    self.config.service_cycles_max,
                ),
                replicas=sorted(self.rng.sample(nodes, replica_count)),
                memory_requirement_gb=self.rng.uniform(
                    self.config.service_memory_gb_min,
                    self.config.service_memory_gb_max,
                ),
                activation_delay_s=self.config.replica_activation_delay_s,
            )
        return services

    def sample_request(self) -> ServiceRequest:
        chain_length = min(self.config.chain_length, len(self.services))
        service_ids = tuple(self.rng.sample(list(self.services), chain_length))
        source, destination = self.rng.sample(range(self.config.total_satellites), 2)
        data = self.rng.uniform(self.config.input_data_gb_min, self.config.input_data_gb_max)
        volumes = [data]
        for _ in range(chain_length):
            data *= self.rng.uniform(self.config.data_shrink_min, self.config.data_shrink_max)
            volumes.append(max(1.0e-5, data))
        request = ServiceRequest(
            request_id=self.request_counter,
            source=source,
            destination=destination,
            services=service_ids,
            data_volumes_gb=tuple(volumes),
            arrival_time_s=(
                self.rng.randrange(self.topology.slot_count) * self.topology.slot_duration_s
            ),
        )
        self.request_counter += 1
        return request

    def reset(self, request: ServiceRequest | None = None) -> ServingSelectionState:
        request = request or self.sample_request()
        # Each request is an independent PPO episode.  Background state is a
        # deterministic function of the observed slot; reservations made by a
        # previous episode must not leak into a new episode whose clock starts
        # at a different trace position.
        self.compute_available_at = {
            node: request.arrival_time_s for node in range(self.config.total_satellites)
        }
        self.link_available_at.clear()
        terminals = {request.source, request.destination}
        for service_id in request.services:
            terminals.update(self.services[service_id].replicas)
        slot = self.topology.absolute_slot(request.arrival_time_s)
        subgraph = self.connector.build(
            self.topology.graph_at_slot(slot), terminals, request.source, slot
        )
        self.runtime = EpisodeRuntime(
            request=request,
            stage=0,
            current_node=request.source,
            current_time_s=request.arrival_time_s,
            accumulated_latency_s=0.0,
            accumulated_energy_j=0.0,
            serving_history=[],
            subgraph=subgraph,
        )
        self.last_state = self._build_state()
        self.last_state.validate()
        return self.last_state

    def _background_compute_queue(self, node: int, slot: int) -> float:
        # Fully observable current background state; future values are never put
        # in the observation.  Integer mixing keeps experiments reproducible.
        mixed = (node * 1_103_515_245 + slot * 12_345 + self.config.seed) & 0xFFFF
        return (mixed / 0xFFFF) * 0.20 * self.config.slot_duration_s

    def _compute_efficiency(self, node: int, slot: int) -> float:
        base = self.resources[node].efficiency
        modifier = 0.90 + 0.10 * math.cos(0.37 * slot + 0.11 * node)
        return max(0.2, min(1.0, base * modifier))

    def _compute_queue(self, node: int, time_s: float) -> float:
        slot = self.topology.absolute_slot(time_s)
        reserved = max(0.0, self.compute_available_at.get(node, 0.0) - time_s)
        return reserved + self._background_compute_queue(node, slot)

    def _link_efficiency(self, edge: ISLEdge, slot: int) -> float:
        phase = 0.19 * slot + 0.07 * edge.u + 0.13 * edge.v
        return max(0.55, min(1.0, 0.80 + 0.20 * math.sin(phase)))

    def _background_link_queue(self, edge: ISLEdge, slot: int) -> float:
        mixed = (edge.u * 2_654_435_761 + edge.v * 97_531 + slot * 31) & 0xFFFF
        return (mixed / 0xFFFF) * 0.05 * self.config.slot_duration_s

    def _link_queue(self, edge: ISLEdge, time_s: float) -> float:
        slot = self.topology.absolute_slot(time_s)
        reserved = max(0.0, self.link_available_at.get(edge.key, 0.0) - time_s)
        return reserved + self._background_link_queue(edge, slot)

    def _ensure_connector_at(self, time_s: float) -> SparseGraph:
        assert self.runtime is not None
        slot = self.topology.absolute_slot(time_s)
        graph = self.topology.graph_at_slot(slot)
        if slot != self.runtime.subgraph.built_slot:
            if self.config.connector_repair_on_slot_change and self.connector.needs_repair(
                self.runtime.subgraph, graph
            ):
                self.runtime.subgraph = self.connector.repair(
                    self.runtime.subgraph, graph, slot
                )
            else:
                # Record that this slot was checked so ordinary stages in the
                # same slot only refresh measurements and do no graph rebuild.
                self.runtime.subgraph = replace(self.runtime.subgraph, built_slot=slot)
        return graph.induced(set(self.runtime.subgraph.nodes))

    def _ensure_connector(self) -> SparseGraph:
        assert self.runtime is not None
        return self._ensure_connector_at(self.runtime.current_time_s)

    @staticmethod
    def _directed_edges(graph: SparseGraph, node_to_local: dict[int, int]):
        edge_pairs: list[tuple[int, int]] = []
        physical_edges: list[ISLEdge] = []
        for edge in graph.edges():
            if edge.u not in node_to_local or edge.v not in node_to_local:
                continue
            edge_pairs.extend(
                ((node_to_local[edge.u], node_to_local[edge.v]),
                 (node_to_local[edge.v], node_to_local[edge.u]))
            )
            physical_edges.extend((edge, edge))
        if not edge_pairs:
            return np.empty((2, 0), dtype=np.int64), []
        return np.asarray(edge_pairs, dtype=np.int64).T, physical_edges

    def _path_metrics(self, graph: SparseGraph, source: int, target: int, time_s: float):
        path = graph.shortest_path(source, target, "hop")
        if path is None:
            return None, 0.0
        if len(path) == 1:
            return path, self.config.compute_capacity_gflops_max * 1000.0
        slot = self.topology.absolute_slot(time_s)
        bottleneck = math.inf
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edge(u, v)
            assert edge is not None
            rate = edge.rate_mbps * self._link_efficiency(edge, slot)
            bottleneck = min(bottleneck, rate)
        return path, float(bottleneck)

    def _build_state(self) -> ServingSelectionState:
        assert self.runtime is not None and not self.runtime.done
        runtime = self.runtime
        request = runtime.request
        current_graph = self._ensure_connector()
        nodes = sorted(runtime.subgraph.nodes)
        node_to_local = {node: index for index, node in enumerate(nodes)}
        slot = self.topology.absolute_slot(runtime.current_time_s)
        current_service = request.services[runtime.stage]
        service = self.services[current_service]

        remaining = self.topology.slot_duration_s - (
            runtime.current_time_s % self.topology.slot_duration_s
        )
        request_features = np.asarray(
            [
                request.arrival_time_s / max(self.topology.slot_duration_s, 1.0),
                runtime.current_time_s / max(self.topology.slot_duration_s * self.topology.slot_count, 1.0),
                remaining / self.topology.slot_duration_s,
                runtime.stage / max(1, len(request.services)),
                request.data_volumes_gb[runtime.stage] / max(self.config.input_data_gb_max, 1.0e-9),
                runtime.accumulated_latency_s / max(self.config.latency_scale_s, 1.0e-9),
                runtime.accumulated_energy_j / max(self.config.energy_scale_j, 1.0e-9),
            ],
            dtype=np.float32,
        )

        node_features = np.zeros((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
        selected = set(runtime.serving_history)
        for local, node in enumerate(nodes):
            plane = node // self.config.sats_per_plane
            position = node % self.config.sats_per_plane
            resource = self.resources[node]
            node_features[local] = (
                math.sin(2 * math.pi * plane / self.config.num_planes),
                math.cos(2 * math.pi * plane / self.config.num_planes),
                math.sin(2 * math.pi * position / self.config.sats_per_plane),
                math.cos(2 * math.pi * position / self.config.sats_per_plane),
                resource.capacity_gflops / self.config.compute_capacity_gflops_max,
                self._compute_efficiency(node, slot),
                min(1.0, self._compute_queue(node, runtime.current_time_s) / self.topology.slot_duration_s),
                resource.compute_power_w / self.config.compute_power_w_max,
                float(node == request.source),
                float(node == request.destination),
                float(node == runtime.current_node),
                float(node in runtime.subgraph.relay_nodes),
                float(node in service.replicas),
                float(node in selected),
            )

        edge_index, physical_edges = self._directed_edges(current_graph, node_to_local)
        edge_features = np.zeros((len(physical_edges), len(EDGE_FEATURES)), dtype=np.float32)
        for index, edge in enumerate(physical_edges):
            efficiency = self._link_efficiency(edge, slot)
            edge_features[index] = (
                edge.rate_mbps * efficiency / 10_000.0,
                efficiency,
                min(1.0, self._link_queue(edge, runtime.current_time_s) / self.topology.slot_duration_s),
                min(1.0, edge.tx_power_w / 5.0),
                min(1.0, edge.distance_km / 10_000.0),
            )

        future = []
        for offset in range(1, self.config.future_topology_horizon + 1):
            future_graph = self.topology.graph_at_slot(slot + offset).induced(set(nodes))
            future_edge_index, _ = self._directed_edges(future_graph, node_to_local)
            future.append(SparseTopologyState(future_edge_index, offset))

        candidates: list[int] = []
        candidate_features: list[tuple[float, float, float]] = []
        for node in service.replicas:
            if node not in node_to_local:
                continue
            path, bottleneck = self._path_metrics(
                current_graph, runtime.current_node, node, runtime.current_time_s
            )
            if path is None:
                continue
            candidates.append(node_to_local[node])
            candidate_features.append(
                (
                    (len(path) - 1) / max(1, self.config.max_route_hops),
                    min(1.0, bottleneck / 10_000.0),
                    min(1.0, self._compute_queue(node, runtime.current_time_s) / self.topology.slot_duration_s),
                )
            )
        if not candidates:
            raise RuntimeError("connector graph has no reachable replica for the current service")

        history = np.full(len(request.services), -1, dtype=np.int64)
        history[: len(runtime.serving_history)] = runtime.serving_history
        state = ServingSelectionState(
            request_continuous=request_features,
            service_ids=np.asarray(request.services, dtype=np.int64),
            data_volumes=np.asarray(request.data_volumes_gb, dtype=np.float32),
            serving_history=history,
            service_mask=np.ones(len(request.services), dtype=bool),
            node_ids=np.asarray(nodes, dtype=np.int64),
            node_features=node_features,
            node_mask=np.ones(len(nodes), dtype=bool),
            current_edge_index=edge_index,
            current_edge_features=edge_features,
            future_topologies=tuple(future),
            candidate_indices=np.asarray(candidates, dtype=np.int64),
            candidate_features=np.asarray(candidate_features, dtype=np.float32).reshape(-1, len(CANDIDATE_FEATURES)),
            action_mask=np.ones(len(candidates), dtype=bool),
        )
        return state

    def _route(self, source: int, target: int, data_gb: float):
        assert self.runtime is not None
        return self.router.route(
            source=source,
            target=target,
            data_gb=data_gb,
            start_time=self.runtime.current_time_s,
            graph_provider=self._ensure_connector_at,
            rate_provider=self._effective_link_rate,
            queue_provider=self._link_queue,
            reserve=self._reserve_link,
            commit=True,
        )

    def _effective_link_rate(self, edge: ISLEdge, absolute_slot: int) -> float:
        return edge.rate_mbps * self._link_efficiency(edge, absolute_slot)

    def _reserve_link(
        self, edge: ISLEdge, start_time: float, data_gb: float, rate_mbps: float
    ) -> None:
        queue_s = self._link_queue(edge, start_time)
        transmission_s = data_gb * 8_000.0 / max(rate_mbps, 1.0e-9)
        self.link_available_at[edge.key] = max(
            self.link_available_at.get(edge.key, 0.0),
            start_time + queue_s + transmission_s,
        )

    def _estimate_full_network_cost(
        self, source: int, target: int, data_gb: float, start_time: float
    ) -> float:
        result = self.router.route(
            source=source,
            target=target,
            data_gb=data_gb,
            start_time=start_time,
            graph_provider=lambda time_s: self.topology.graph_at_time(time_s),
            rate_provider=self._effective_link_rate,
            queue_provider=self._link_queue,
            reserve=None,
            commit=False,
        )
        if not result["reachable"]:
            return math.inf
        return (
            self.config.delay_weight
            * result["delay_s"]
            / max(self.config.latency_scale_s, 1.0e-9)
            + self.config.energy_weight
            * result["energy_j"]
            / max(self.config.energy_scale_j, 1.0e-9)
        )

    def _normalized_compute_load(self, node: int, time_s: float) -> float:
        return min(1.0, self._compute_queue(node, time_s) / self.topology.slot_duration_s)

    def control_state_dict(self) -> dict:
        return {
            "service_replicas": {
                int(service_id): list(service.replicas)
                for service_id, service in self.services.items()
            },
            "replica_adapter": self.replica_adapter.state_dict(),
        }

    def load_control_state_dict(self, state: dict) -> None:
        for service_id, replicas in state.get("service_replicas", {}).items():
            self.services[int(service_id)].replicas = sorted(map(int, replicas))
        if "replica_adapter" in state:
            self.replica_adapter.load_state_dict(state["replica_adapter"])

    def _compute(self, service_id: int, node: int, arrival_time_s: float):
        service = self.services[service_id]
        resource = self.resources[node]
        slot = self.topology.absolute_slot(arrival_time_s)
        queue_s = self._compute_queue(node, arrival_time_s)
        efficiency = self._compute_efficiency(node, slot)
        compute_s = service.workload_cycles / (
            resource.capacity_gflops * 1.0e9 * efficiency
        )
        finish = arrival_time_s + queue_s + compute_s
        self.compute_available_at[node] = max(self.compute_available_at.get(node, 0.0), finish)
        return {
            "finish_time_s": finish,
            "queue_s": queue_s,
            "compute_s": compute_s,
            "energy_j": resource.compute_power_w * compute_s,
        }

    def step(self, action: int):
        if self.runtime is None or self.last_state is None:
            raise RuntimeError("reset() must be called before step()")
        if self.runtime.done:
            raise RuntimeError("episode is already done")
        if action < 0 or action >= len(self.last_state.candidate_indices):
            raise ValueError("action is outside the current candidate set")
        if not self.last_state.action_mask[action]:
            raise ValueError("action is masked")

        runtime = self.runtime
        stage_start = runtime.current_time_s
        energy = 0.0
        local_index = int(self.last_state.candidate_indices[action])
        selected_node = int(self.last_state.node_ids[local_index])
        service_id = runtime.request.services[runtime.stage]
        data_gb = runtime.request.data_volumes_gb[runtime.stage]

        route = self._route(runtime.current_node, selected_node, data_gb)
        if not route["reachable"]:
            runtime.done = True
            reward = -self.config.failure_penalty
            return None, reward, True, False, {
                "success": False,
                "reason": route.get("failure_reason", "route_failed"),
                "route": route,
            }
        energy += route["energy_j"]
        computation = self._compute(service_id, selected_node, route["arrival_time"])
        energy += computation["energy_j"]
        runtime.current_time_s = computation["finish_time_s"]
        runtime.current_node = selected_node
        runtime.serving_history.append(selected_node)
        runtime.stage += 1

        final_route = None
        if runtime.stage == len(runtime.request.services):
            final_route = self._route(
                runtime.current_node,
                runtime.request.destination,
                runtime.request.data_volumes_gb[-1],
            )
            if not final_route["reachable"]:
                runtime.done = True
                reward = -self.config.failure_penalty
                return None, reward, True, False, {
                    "success": False,
                    "reason": final_route.get("failure_reason", "egress_failed"),
                    "final_route": final_route,
                }
            runtime.current_time_s = final_route["arrival_time"]
            energy += final_route["energy_j"]

        stage_latency = runtime.current_time_s - stage_start
        runtime.accumulated_latency_s += stage_latency
        runtime.accumulated_energy_j += energy
        observed_route_delay = route["delay_s"] + (
            final_route["delay_s"] if final_route is not None else 0.0
        )
        observed_route_energy = route["energy_j"] + (
            final_route["energy_j"] if final_route is not None else 0.0
        )
        normalized_stage_cost = (
            self.config.delay_weight
            * (observed_route_delay + computation["queue_s"] + computation["compute_s"])
            / max(self.config.latency_scale_s, 1.0e-9)
            + self.config.energy_weight
            * (observed_route_energy + computation["energy_j"])
            / max(self.config.energy_scale_j, 1.0e-9)
        )
        self.replica_adapter.observe_stage(
            StageExecutionRecord(
                request_id=runtime.request.request_id,
                service_id=service_id,
                source_node=route["source"],
                serving_node=selected_node,
                data_gb=data_gb,
                route_delay_s=observed_route_delay,
                route_energy_j=observed_route_energy,
                compute_queue_s=computation["queue_s"],
                compute_delay_s=computation["compute_s"],
                compute_energy_j=computation["energy_j"],
                normalized_cost=normalized_stage_cost,
            )
        )
        reward = -(
            self.config.delay_weight * stage_latency / self.config.latency_scale_s
            + self.config.energy_weight * energy / self.config.energy_scale_j
        )
        terminated = runtime.stage == len(runtime.request.services)
        truncated = runtime.stage >= self.config.max_episode_steps and not terminated
        runtime.done = terminated or truncated
        info = {
            "success": terminated,
            "stage": runtime.stage - 1,
            "selected_node": selected_node,
            "stage_latency_s": stage_latency,
            "stage_energy_j": energy,
            "route": route,
            "compute": computation,
            "final_route": final_route,
            "relay_count": len(runtime.subgraph.relay_nodes),
            "subgraph_node_count": len(runtime.subgraph.nodes),
        }
        if runtime.done:
            migration_actions = []
            if terminated and self.config.adaptation_enabled:
                migration_actions = self.replica_adapter.close_request(
                    self.services,
                    self.resources,
                    runtime.current_time_s,
                    self._estimate_full_network_cost,
                    self._normalized_compute_load,
                )
            self.last_migration_actions = migration_actions
            info.update(
                total_latency_s=runtime.accumulated_latency_s,
                total_energy_j=runtime.accumulated_energy_j,
                serving_history=tuple(runtime.serving_history),
                migration_actions=[asdict(item) for item in migration_actions],
                bandit_summary=self.replica_adapter.summary(),
            )
            self.last_state = None
            return None, reward, terminated, truncated, info
        self.last_state = self._build_state()
        self.last_state.validate()
        return self.last_state, reward, False, False, info
