from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace

import numpy as np

from .config import ELARAConfig
from .background import (
    ComputeBackground,
    LinkBackground,
    MarkovBackgroundProcess,
)
from .bandit import BanditReplicaAdapter, StageExecutionRecord
from .connector import ConnectorBuilder, RequestSubgraph
from .domain import (
    Microservice,
    SatelliteResource,
    ServiceRequest,
    ServiceRequestTemplate,
)
from .routing import CrossSlotMinCostRouter
from .request_templates import load_templates
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


@dataclass
class RequestSession:
    """Mutable state owned by one independent target request."""

    runtime: EpisodeRuntime
    last_state: ServingSelectionState | None
    compute_available_at: dict[int, float]
    link_available_at: dict[tuple[int, int], float]
    graph_cache: dict[tuple[int, tuple[int, ...]], SparseGraph]
    edge_cache: dict[
        tuple[int, tuple[int, ...]],
        tuple[np.ndarray, list[ISLEdge]],
    ]
    compute_background_cache: dict[tuple[int, int], ComputeBackground]
    link_background_cache: dict[
        tuple[int, tuple[int, int]], LinkBackground
    ]


class ELARAEnvironment:
    """Independent request-level environment for serving-satellite PPO.

    A full request is one episode and every microservice choice is one action.
    Routing and computation reservations persist only inside that request.
    Other concurrent work is represented exclusively by the slot-correlated
    Markov background communication and computation processes.
    """

    def __init__(self, config: ELARAConfig | None = None, topology: TemporalTopology | None = None):
        self.config = config or ELARAConfig()
        self.rng = random.Random(self.config.seed)
        self.request_rng = random.Random(self.config.seed + 10_000)
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
        self.resources = self._generate_resources()
        self.services = self._generate_services()
        self.initial_service_replicas = {
            service_id: tuple(service.replicas)
            for service_id, service in self.services.items()
        }
        self.background = MarkovBackgroundProcess(
            self.topology, self.resources, self.config, self.config.seed + 20_000
        )
        self.request_templates = self._generate_request_templates()
        self.arrival_time_s = 0.0
        self.compute_available_at = {node: 0.0 for node in range(self.config.total_satellites)}
        self.link_available_at: dict[tuple[int, int], float] = {}
        self.runtime: EpisodeRuntime | None = None
        self.request_counter = 0
        self.last_state: ServingSelectionState | None = None
        self.last_migration_actions = []
        self._request_graph_cache: dict[
            tuple[int, tuple[int, ...]], SparseGraph
        ] = {}
        self._request_edge_cache: dict[
            tuple[int, tuple[int, ...]],
            tuple[np.ndarray, list[ISLEdge]],
        ] = {}
        self._request_compute_background_cache: dict[
            tuple[int, int], ComputeBackground
        ] = {}
        self._request_link_background_cache: dict[
            tuple[int, tuple[int, int]], LinkBackground
        ] = {}
        self._session_trace_start = 0
        self._session_request_order: dict[int, int] = {}
        self.replica_adapter.start_fresh_window(0.0)

    def _generate_resources(self) -> dict[int, SatelliteResource]:
        resources = {}
        for node in range(self.config.total_satellites):
            nominal_capacity = self.rng.choice(
                self.config.compute_capacity_choices_gflops
            )
            capacity = nominal_capacity * self.config.compute_capacity_scale
            resources[node] = SatelliteResource(
                node_id=node,
                capacity_gflops=capacity,
                compute_power_w=self.config.compute_power_by_capacity_w[
                    nominal_capacity
                ],
                efficiency=1.0,
                memory_capacity_gb=self.config.satellite_memory_capacity_gb,
            )
        return resources

    def _generate_services(self) -> dict[int, Microservice]:
        services = {}
        nodes = list(range(self.config.total_satellites))
        memory_used = {node: 0.0 for node in nodes}
        for service_id in range(self.config.num_services):
            if self.config.replicas_per_service is None:
                replica_count = self.rng.randint(*self.config.replica_count_range)
            else:
                replica_count = self.config.replicas_per_service
            replica_count = min(replica_count, len(nodes))
            memory_requirement = self.rng.uniform(
                self.config.service_memory_gb_min,
                self.config.service_memory_gb_max,
            )
            feasible = [
                node for node in nodes
                if memory_used[node] + memory_requirement
                <= self.resources[node].memory_capacity_gb + 1.0e-9
            ]
            if len(feasible) < replica_count:
                raise RuntimeError("not enough satellite memory for initial replicas")
            feasible.sort(key=lambda node: (memory_used[node], self.rng.random()))
            replicas = sorted(feasible[:replica_count])
            for node in replicas:
                memory_used[node] += memory_requirement
            services[service_id] = Microservice(
                service_id=service_id,
                workload_cycles=self.rng.uniform(
                    self.config.service_cycles_min,
                    self.config.service_cycles_max,
                ),
                replicas=replicas,
                memory_requirement_gb=memory_requirement,
                activation_delay_s=self.rng.uniform(
                    self.config.replica_activation_delay_s_min,
                    self.config.replica_activation_delay_s_max,
                ),
            )
        return services

    def _sample_data_gb(self) -> float:
        return min(
            self.config.input_data_gb_max,
            max(
                self.config.input_data_gb_min,
                self.rng.gauss(
                    self.config.request_data_mean_gb,
                    math.sqrt(self.config.request_data_variance_gb),
                ),
            ),
        )

    def _generate_request_templates(self) -> tuple[ServiceRequestTemplate, ...]:
        if self.config.request_template_file is not None:
            return load_templates(
                self.config.request_template_file,
                num_services=self.config.num_services,
                data_scale=self.config.request_data_scale,
            )
        lengths = (
            (self.config.chain_length,)
            if self.config.chain_length is not None
            else self.config.request_template_chain_lengths
        )
        templates = []
        service_ids = list(self.services)
        for template_id, requested_length in enumerate(lengths, start=1):
            length = max(1, int(requested_length))
            services = tuple(self.rng.choices(service_ids, k=length))
            volumes = tuple(
                self._sample_data_gb() * self.config.request_data_scale
                for _ in range(length + 1)
            )
            templates.append(ServiceRequestTemplate(template_id, services, volumes))
        return tuple(templates)

    def _nearby_nodes(self, anchors: list[int], graph: SparseGraph) -> list[int]:
        visited = set(map(int, anchors))
        frontier = set(visited)
        for _ in range(max(0, self.config.request_endpoint_near_hops)):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(map(int, graph.neighbors(node)))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        return sorted(visited)

    def sample_request(self) -> ServiceRequest:
        total_lambda = (
            self.config.request_arrival_lambda_per_template_per_slot
            * len(self.request_templates)
        )
        rate_per_second = total_lambda / self.topology.slot_duration_s
        if rate_per_second > 0.0:
            self.arrival_time_s += self.request_rng.expovariate(rate_per_second)
        template = self.request_rng.choice(self.request_templates)
        graph = self.topology.graph_at_time(self.arrival_time_s)
        # Request endpoints are anchored to the initial deployment rather than
        # the placement produced by a previous control action. This preserves
        # a common exogenous request stream across sensitivity conditions.
        first_replicas = self.initial_service_replicas[template.services[0]]
        last_replicas = self.initial_service_replicas[template.services[-1]]
        source_pool = self._nearby_nodes(first_replicas, graph)
        destination_pool = self._nearby_nodes(last_replicas, graph)
        source = self.request_rng.choice(source_pool)
        destination_choices = [node for node in destination_pool if node != source]
        if not destination_choices:
            destination_choices = [node for node in graph.nodes if node != source]
        destination = self.request_rng.choice(destination_choices)
        request = ServiceRequest(
            request_id=self.request_counter,
            source=source,
            destination=destination,
            services=template.services,
            data_volumes_gb=template.data_volumes_gb,
            arrival_time_s=self.arrival_time_s,
            template_id=template.template_id,
        )
        self.request_counter += 1
        return request

    def iter_request_batches(
        self,
        *,
        slot_count: int | None = None,
        request_count: int | None = None,
    ):
        """Yield Poisson arrivals grouped by absolute time slot.

        Exactly one limit must be supplied.  A slot limit admits every request
        whose arrival falls in ``[0, slot_count * slot_duration)`` and also
        yields empty slots.  A request limit samples exactly that many requests
        and groups them without changing their continuous arrival times.
        """
        if (slot_count is None) == (request_count is None):
            raise ValueError("supply exactly one of slot_count or request_count")
        if slot_count is not None:
            if slot_count < 0:
                raise ValueError("slot_count must be nonnegative")
            pending = self.sample_request()
            for absolute_slot in range(slot_count):
                slot_end = (absolute_slot + 1) * self.topology.slot_duration_s
                batch = []
                while pending.arrival_time_s < slot_end:
                    batch.append(pending)
                    pending = self.sample_request()
                yield absolute_slot, tuple(batch)
            return

        if request_count is None or request_count < 0:
            raise ValueError("request_count must be nonnegative")
        grouped: dict[int, list[ServiceRequest]] = {}
        for _ in range(request_count):
            request = self.sample_request()
            slot = self.topology.absolute_slot(request.arrival_time_s)
            grouped.setdefault(slot, []).append(request)
        for absolute_slot in range(max(grouped, default=-1) + 1):
            yield absolute_slot, tuple(grouped.get(absolute_slot, ()))

    def _clear_request_reservations(self, reference_time_s: float) -> None:
        self.compute_available_at = {
            node: reference_time_s for node in range(self.config.total_satellites)
        }
        self.link_available_at.clear()

    def reset(self, request: ServiceRequest | None = None) -> ServingSelectionState:
        request = request or self.sample_request()
        # A target request is an independent episode. Reservations from every
        # preceding target request, including requests in the same slot, must
        # never enter this request's queues.
        self._clear_request_reservations(request.arrival_time_s)
        self._request_graph_cache.clear()
        self._request_edge_cache.clear()
        self._request_compute_background_cache.clear()
        self._request_link_background_cache.clear()
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

    def capture_request_session(self) -> RequestSession:
        if self.runtime is None:
            raise RuntimeError("no active request to capture")
        return RequestSession(
            runtime=self.runtime,
            last_state=self.last_state,
            compute_available_at=dict(self.compute_available_at),
            link_available_at=dict(self.link_available_at),
            graph_cache=dict(self._request_graph_cache),
            edge_cache=dict(self._request_edge_cache),
            compute_background_cache=dict(
                self._request_compute_background_cache
            ),
            link_background_cache=dict(self._request_link_background_cache),
        )

    def restore_request_session(self, session: RequestSession) -> None:
        self.runtime = session.runtime
        self.last_state = session.last_state
        self.compute_available_at = session.compute_available_at
        self.link_available_at = session.link_available_at
        self._request_graph_cache = session.graph_cache
        self._request_edge_cache = session.edge_cache
        self._request_compute_background_cache = (
            session.compute_background_cache
        )
        self._request_link_background_cache = session.link_background_cache

    def start_request_sessions(
        self, requests
    ) -> list[RequestSession]:
        requests = list(requests)
        self._session_trace_start = len(
            self.replica_adapter.window_records
        )
        self._session_request_order = {
            int(request.request_id): index
            for index, request in enumerate(requests)
        }
        sessions = []
        for request in requests:
            self.reset(request)
            sessions.append(self.capture_request_session())
        return sessions

    def finalize_request_sessions(self) -> None:
        """Restore sequential trace order before window sampling by Bandit."""
        records = self.replica_adapter.window_records
        start = min(self._session_trace_start, len(records))
        if start < len(records):
            records[start:] = sorted(
                records[start:],
                key=lambda record: (
                    self._session_request_order.get(
                        int(record.request_id), len(self._session_request_order)
                    ),
                    int(record.stage_index),
                ),
            )
        self._session_trace_start = len(records)
        self._session_request_order = {}

    def _compute_background(
        self, node: int, slot: int
    ) -> ComputeBackground:
        key = (int(slot), int(node))
        cached = self._request_compute_background_cache.get(key)
        if cached is None:
            cached = self.background.compute(node, slot)
            self._request_compute_background_cache[key] = cached
        return cached

    def _link_background(
        self, edge: ISLEdge, slot: int
    ) -> LinkBackground:
        key = (int(slot), edge.key)
        cached = self._request_link_background_cache.get(key)
        if cached is None:
            cached = self.background.link(edge, slot)
            self._request_link_background_cache[key] = cached
        return cached

    def _compute_efficiency(self, node: int, slot: int) -> float:
        return self._compute_background(node, slot).discount

    def _compute_queue(self, node: int, time_s: float) -> float:
        slot = self.topology.absolute_slot(time_s)
        reserved = max(0.0, self.compute_available_at.get(node, 0.0) - time_s)
        return reserved + self._compute_background(
            node, slot
        ).queue_delay_s

    def _link_efficiency(self, edge: ISLEdge, slot: int) -> float:
        return self._link_background(edge, slot).efficiency

    def _link_queue(self, edge: ISLEdge, time_s: float) -> float:
        slot = self.topology.absolute_slot(time_s)
        reserved = max(0.0, self.link_available_at.get(edge.key, 0.0) - time_s)
        return reserved + self._link_background(
            edge, slot
        ).queue_delay_s

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
        return self._cached_request_graph(slot, self.runtime.subgraph.nodes)

    def _cached_request_graph(
        self, absolute_slot: int, nodes
    ) -> SparseGraph:
        node_key = tuple(sorted(map(int, nodes)))
        key = (absolute_slot % self.topology.slot_count, node_key)
        cached = self._request_graph_cache.get(key)
        if cached is None:
            cached = self.topology.graph_at_slot(absolute_slot).induced(
                set(node_key)
            )
            self._request_graph_cache[key] = cached
        return cached

    def _cached_directed_edges(
        self,
        absolute_slot: int,
        nodes: tuple[int, ...],
        graph: SparseGraph,
        node_to_local: dict[int, int],
    ):
        key = (absolute_slot % self.topology.slot_count, nodes)
        cached = self._request_edge_cache.get(key)
        if cached is None:
            cached = self._directed_edges(graph, node_to_local)
            self._request_edge_cache[key] = cached
        return cached

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
            return (
                path,
                self.config.compute_capacity_gflops_max
                * self.config.compute_capacity_scale
                * 1000.0,
            )
        slot = self.topology.absolute_slot(time_s)
        bottleneck = math.inf
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edge(u, v)
            assert edge is not None
            rate = (
                edge.rate_mbps
                * self.config.link_capacity_scale
                * self._link_efficiency(edge, slot)
            )
            bottleneck = min(bottleneck, rate)
        return path, float(bottleneck)

    def _build_state(self) -> ServingSelectionState:
        assert self.runtime is not None and not self.runtime.done
        runtime = self.runtime
        request = runtime.request
        current_graph = self._ensure_connector()
        nodes = tuple(sorted(runtime.subgraph.nodes))
        node_to_local = {node: index for index, node in enumerate(nodes)}
        slot = self.topology.absolute_slot(runtime.current_time_s)
        current_service = request.services[runtime.stage]
        service = self.services[current_service]

        remaining = self.topology.slot_duration_s - (
            runtime.current_time_s % self.topology.slot_duration_s
        )
        request_features = np.asarray(
            [
                (request.arrival_time_s % (self.topology.slot_duration_s * self.topology.slot_count))
                / max(self.topology.slot_duration_s * self.topology.slot_count, 1.0),
                (runtime.current_time_s % (self.topology.slot_duration_s * self.topology.slot_count))
                / max(self.topology.slot_duration_s * self.topology.slot_count, 1.0),
                remaining / self.topology.slot_duration_s,
                runtime.stage / max(1, len(request.services)),
                request.data_volumes_gb[runtime.stage]
                / max(
                    self.config.input_data_gb_max
                    * self.config.request_data_scale,
                    1.0e-9,
                ),
                runtime.accumulated_latency_s / max(self.config.latency_scale_s, 1.0e-9),
                runtime.accumulated_energy_j / max(self.config.energy_scale_j, 1.0e-9),
            ],
            dtype=np.float32,
        )

        node_features = np.zeros((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
        selected = set(runtime.serving_history)
        compute_background = {
            node: self._compute_background(node, slot) for node in nodes
        }
        for local, node in enumerate(nodes):
            plane = node // self.config.sats_per_plane
            position = node % self.config.sats_per_plane
            resource = self.resources[node]
            node_features[local] = (
                math.sin(2 * math.pi * plane / self.config.num_planes),
                math.cos(2 * math.pi * plane / self.config.num_planes),
                math.sin(2 * math.pi * position / self.config.sats_per_plane),
                math.cos(2 * math.pi * position / self.config.sats_per_plane),
                resource.capacity_gflops
                / (
                    self.config.compute_capacity_gflops_max
                    * self.config.compute_capacity_scale
                ),
                compute_background[node].discount,
                min(
                    1.0,
                    (
                        max(
                            0.0,
                            self.compute_available_at.get(node, 0.0)
                            - runtime.current_time_s,
                        )
                        + compute_background[node].queue_delay_s
                    )
                    / self.topology.slot_duration_s,
                ),
                resource.compute_power_w / self.config.compute_power_w_max,
                float(node == request.source),
                float(node == request.destination),
                float(node == runtime.current_node),
                float(node in runtime.subgraph.relay_nodes),
                float(node in service.replicas),
                float(node in selected),
            )

        edge_index, physical_edges = self._cached_directed_edges(
            slot, nodes, current_graph, node_to_local
        )
        edge_features = np.zeros((len(physical_edges), len(EDGE_FEATURES)), dtype=np.float32)
        link_background = {
            edge.key: self._link_background(edge, slot)
            for edge in current_graph.edges()
        }
        for index, edge in enumerate(physical_edges):
            background = link_background[edge.key]
            efficiency = background.efficiency
            edge_features[index] = (
                edge.rate_mbps
                * self.config.link_capacity_scale
                * efficiency
                / 10_000.0,
                efficiency,
                min(
                    1.0,
                    (
                        max(
                            0.0,
                            self.link_available_at.get(edge.key, 0.0)
                            - runtime.current_time_s,
                        )
                        + background.queue_delay_s
                    )
                    / self.topology.slot_duration_s,
                ),
                min(1.0, edge.tx_power_w / 5.0),
                min(1.0, edge.distance_km / 10_000.0),
            )

        future = []
        for offset in range(1, self.config.future_topology_horizon + 1):
            future_slot = slot + offset
            future_graph = self._cached_request_graph(future_slot, nodes)
            future_edge_index, _ = self._cached_directed_edges(
                future_slot, nodes, future_graph, node_to_local
            )
            future.append(SparseTopologyState(future_edge_index, offset))

        candidates: list[int] = []
        candidate_features: list[tuple[float, float, float]] = []
        distances, parents = current_graph.shortest_paths(
            runtime.current_node, "hop"
        )
        for node in service.replicas:
            if node not in node_to_local:
                continue
            if node not in distances:
                continue
            path = [int(node)]
            while path[-1] != runtime.current_node:
                path.append(parents[path[-1]])
            path.reverse()
            _, bottleneck = self._path_metrics_from_path(
                current_graph, path, slot, link_background
            )
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

    def _path_metrics_from_path(
        self,
        graph: SparseGraph,
        path: list[int],
        slot: int,
        background_by_edge: dict[tuple[int, int], LinkBackground] | None = None,
    ):
        if len(path) == 1:
            return (
                path,
                self.config.compute_capacity_gflops_max
                * self.config.compute_capacity_scale
                * 1000.0,
            )
        bottleneck = math.inf
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edge(u, v)
            assert edge is not None
            background = (
                background_by_edge[edge.key]
                if background_by_edge is not None
                else self._link_background(edge, slot)
            )
            bottleneck = min(
                bottleneck,
                edge.rate_mbps
                * self.config.link_capacity_scale
                * background.efficiency,
            )
        return path, float(bottleneck)

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
        return (
            edge.rate_mbps
            * self.config.link_capacity_scale
            * self._link_efficiency(edge, absolute_slot)
        )

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

    def _trace_candidates(self):
        assert self.last_state is not None
        return {
            "candidate_nodes": tuple(
                int(self.last_state.node_ids[int(index)])
                for index in self.last_state.candidate_indices
            ),
            "candidate_hop_distances": tuple(
                float(value) for value in self.last_state.candidate_features[:, 0]
            ),
            "candidate_bottleneck_rates": tuple(
                float(value) for value in self.last_state.candidate_features[:, 1]
            ),
            "candidate_compute_queues": tuple(
                float(value) for value in self.last_state.candidate_features[:, 2]
            ),
        }

    def _record_completed_request(self):
        if not self.config.adaptation_enabled:
            return
        self.replica_adapter.record_request()

    def finish_time_slot(self, absolute_slot: int):
        """Advance control after every arrival in ``absolute_slot`` is done."""
        boundary_time_s = (absolute_slot + 1) * self.topology.slot_duration_s
        self._clear_request_reservations(boundary_time_s)
        if not self.config.adaptation_enabled:
            self.last_migration_actions = []
            return []
        actions = self.replica_adapter.adapt_if_due(
            self.services,
            self.resources,
            boundary_time_s,
            self._estimate_full_network_cost,
            self._normalized_compute_load,
        )
        self.last_migration_actions = actions
        return actions

    def _observe_failed_stage(
        self,
        runtime: EpisodeRuntime,
        service_id: int,
        stage_index: int,
        stage_start: float,
        selected_node: int,
        reason: str,
    ) -> None:
        if not self.config.adaptation_enabled:
            return
        self.replica_adapter.observe_stage(
            StageExecutionRecord(
                request_id=runtime.request.request_id,
                service_id=service_id,
                source_node=runtime.current_node,
                serving_node=selected_node,
                data_gb=runtime.request.data_volumes_gb[stage_index],
                route_delay_s=self.config.failure_penalty * self.config.latency_scale_s,
                route_energy_j=0.0,
                compute_queue_s=0.0,
                compute_delay_s=0.0,
                compute_energy_j=0.0,
                normalized_cost=self.config.failure_penalty,
                template_id=runtime.request.template_id,
                request_arrival_time_s=runtime.request.arrival_time_s,
                stage_index=stage_index,
                chain_length=len(runtime.request.services),
                stage_start_time_s=stage_start,
                stage_finish_time_s=stage_start,
                topology_slot=self.topology.absolute_slot(stage_start),
                destination_node=runtime.request.destination,
                success=False,
                failure_reason=reason,
                **self._trace_candidates(),
            )
        )

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
        stage_index = runtime.stage
        energy = 0.0
        local_index = int(self.last_state.candidate_indices[action])
        selected_node = int(self.last_state.node_ids[local_index])
        service_id = runtime.request.services[runtime.stage]
        data_gb = runtime.request.data_volumes_gb[runtime.stage]

        route = self._route(runtime.current_node, selected_node, data_gb)
        if not route["reachable"]:
            runtime.done = True
            reason = route.get("failure_reason", "route_failed")
            self._observe_failed_stage(
                runtime, service_id, stage_index, stage_start, selected_node, reason
            )
            self._record_completed_request()
            migration_actions = []
            self.last_state = None
            reward = -self.config.failure_penalty
            return None, reward, True, False, {
                "success": False,
                "request_id": runtime.request.request_id,
                "template_id": runtime.request.template_id,
                "arrival_time_s": runtime.request.arrival_time_s,
                "chain_length": len(runtime.request.services),
                "reason": reason,
                "route": route,
                "total_latency_s": runtime.accumulated_latency_s,
                "total_energy_j": runtime.accumulated_energy_j,
                "migration_actions": [asdict(item) for item in migration_actions],
                "bandit_summary": self.replica_adapter.summary(),
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
                reason = final_route.get("failure_reason", "egress_failed")
                self._observe_failed_stage(
                    runtime, service_id, stage_index, stage_start, selected_node, reason
                )
                self._record_completed_request()
                migration_actions = []
                self.last_state = None
                reward = -self.config.failure_penalty
                return None, reward, True, False, {
                    "success": False,
                    "request_id": runtime.request.request_id,
                    "template_id": runtime.request.template_id,
                    "arrival_time_s": runtime.request.arrival_time_s,
                    "chain_length": len(runtime.request.services),
                    "reason": reason,
                    "final_route": final_route,
                    "total_latency_s": runtime.accumulated_latency_s,
                    "total_energy_j": runtime.accumulated_energy_j,
                    "migration_actions": [asdict(item) for item in migration_actions],
                    "bandit_summary": self.replica_adapter.summary(),
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
        if self.config.adaptation_enabled:
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
                    template_id=runtime.request.template_id,
                    request_arrival_time_s=runtime.request.arrival_time_s,
                    stage_index=stage_index,
                    chain_length=len(runtime.request.services),
                    stage_start_time_s=stage_start,
                    stage_finish_time_s=runtime.current_time_s,
                    topology_slot=self.topology.absolute_slot(stage_start),
                    destination_node=runtime.request.destination,
                    route_slot_crossings=int(route.get("slot_crossings", 0)),
                    **self._trace_candidates(),
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
            self._record_completed_request()
            info.update(
                request_id=runtime.request.request_id,
                template_id=runtime.request.template_id,
                arrival_time_s=runtime.request.arrival_time_s,
                chain_length=len(runtime.request.services),
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
