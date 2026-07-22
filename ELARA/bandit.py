from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .domain import Microservice, SatelliteResource


@dataclass(frozen=True)
class StageExecutionRecord:
    request_id: int
    service_id: int
    source_node: int
    serving_node: int
    data_gb: float
    route_delay_s: float
    route_energy_j: float
    compute_queue_s: float
    compute_delay_s: float
    compute_energy_j: float
    normalized_cost: float


@dataclass
class ServicePressure:
    service_id: int
    invocation_count: int
    impact: float
    route_pressure: float
    queue_pressure: float
    replica_imbalance: float
    pressure: float
    mean_stage_cost: float
    replica_mean_cost: dict[int, float] = field(default_factory=dict)
    replica_counts: Counter = field(default_factory=Counter)
    routing_samples: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class MigrationAction:
    action: str
    service_id: int
    source_node: int
    target_node: int
    source_plane: int
    target_plane: int
    features: tuple[float, float, float]
    pressure: float
    expected_network_cost: float
    target_compute_load: float
    ucb_score: float
    baseline_service_cost: float
    activation_delay_s: float = 0.0
    feedback_reward: float = 0.0


class BanditReplicaAdapter:
    """Slow-timescale service-pressure ranking plus shared linear UCB."""

    feature_dim = 3

    def __init__(self, config):
        self.config = config
        self.matrix_a = np.eye(self.feature_dim, dtype=np.float64)
        self.vector_b = np.zeros(self.feature_dim, dtype=np.float64)
        self.window_records: list[StageExecutionRecord] = []
        self.requests_in_window = 0
        self.pending_actions: list[MigrationAction] = []
        self.pressure_ewma: dict[int, float] = {}
        self.total_windows = 0
        self.total_pulls = 0
        self.total_relocations = 0
        self.total_feedback_updates = 0
        self.action_counts = Counter()
        self.last_pressures: dict[int, ServicePressure] = {}

    def observe_stage(self, record: StageExecutionRecord) -> None:
        self.window_records.append(record)

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.quantile(np.asarray(values, dtype=float), q))

    @classmethod
    def _tail_pressure(cls, values: list[float]) -> float:
        if not values:
            return 0.0
        median = cls._percentile(values, 0.5)
        p95 = cls._percentile(values, 0.95)
        return max(0.0, p95 / max(1.0e-9, median) - 1.0)

    @staticmethod
    def _saturate(value: float) -> float:
        value = max(0.0, value)
        return value / (1.0 + value)

    def _pressures(self, services: dict[int, Microservice]) -> dict[int, ServicePressure]:
        grouped = defaultdict(list)
        for record in self.window_records:
            grouped[record.service_id].append(record)
        total_cost = sum(record.normalized_cost for record in self.window_records)
        result = {}
        for service_id, records in grouped.items():
            costs_by_node = defaultdict(list)
            counts = Counter(record.serving_node for record in records)
            for record in records:
                costs_by_node[record.serving_node].append(record.normalized_cost)
            service = services[service_id]
            all_counts = [counts.get(node, 0) for node in service.replicas]
            mean_count = sum(all_counts) / max(1, len(all_counts))
            imbalance = (
                max(all_counts) / max(1.0e-9, mean_count) - 1.0
                if sum(all_counts) else 0.0
            )
            impact = sum(record.normalized_cost for record in records) / max(total_cost, 1.0e-9)
            route_pressure = self._tail_pressure([record.route_delay_s for record in records])
            queue_pressure = self._tail_pressure([record.compute_queue_s for record in records])
            actionability = max(
                self._saturate(route_pressure),
                self._saturate(queue_pressure),
                self._saturate(imbalance),
            )
            raw_pressure = impact * actionability
            previous = self.pressure_ewma.get(service_id, raw_pressure)
            smoothed = (
                self.config.pressure_ewma * raw_pressure
                + (1.0 - self.config.pressure_ewma) * previous
            )
            self.pressure_ewma[service_id] = smoothed
            result[service_id] = ServicePressure(
                service_id=service_id,
                invocation_count=len(records),
                impact=impact,
                route_pressure=route_pressure,
                queue_pressure=queue_pressure,
                replica_imbalance=imbalance,
                pressure=smoothed,
                mean_stage_cost=sum(record.normalized_cost for record in records) / len(records),
                replica_mean_cost={
                    node: sum(values) / len(values) for node, values in costs_by_node.items()
                },
                replica_counts=counts,
                routing_samples=[(record.source_node, record.data_gb) for record in records[:4]],
            )
        return result

    def _update_feedback(self, pressures: dict[int, ServicePressure]) -> None:
        for action in self.pending_actions:
            after = pressures.get(action.service_id)
            after_cost = after.mean_stage_cost if after else action.baseline_service_cost
            baseline = max(action.baseline_service_cost, 1.0e-9)
            reward = (action.baseline_service_cost - after_cost) / baseline
            action.feedback_reward = reward
            features = np.asarray(action.features, dtype=float)
            self.matrix_a += np.outer(features, features)
            self.vector_b += reward * features
            self.total_feedback_updates += 1

    @staticmethod
    def _minmax(value: float, values: list[float]) -> float:
        low, high = min(values), max(values)
        return 0.0 if high <= low + 1.0e-12 else (value - low) / (high - low)

    def _ucb(self, features: tuple[float, float, float]) -> float:
        x = np.asarray(features, dtype=float)
        inverse = np.linalg.inv(self.matrix_a)
        theta = inverse @ self.vector_b
        uncertainty = math.sqrt(max(0.0, float(x @ inverse @ x)))
        return float(theta @ x + self.config.bandit_exploration * uncertainty)

    def _memory_usage(self, services: dict[int, Microservice]) -> dict[int, float]:
        usage = defaultdict(float)
        for service in services.values():
            for node in service.replicas:
                usage[node] += service.memory_requirement_gb
        return usage

    def _source_replica(self, service: Microservice, signal: ServicePressure) -> int:
        return max(
            service.replicas,
            key=lambda node: (
                signal.replica_mean_cost.get(node, 0.0),
                -signal.replica_counts.get(node, 0),
                -node,
            ),
        )

    def close_request(
        self,
        services: dict[int, Microservice],
        resources: dict[int, SatelliteResource],
        current_time: float,
        route_cost: Callable[[int, int, float, float], float],
        compute_load: Callable[[int, float], float],
    ) -> list[MigrationAction]:
        self.requests_in_window += 1
        if self.requests_in_window < self.config.deployment_window_requests:
            return []
        self.requests_in_window = 0
        self.total_windows += 1
        pressures = self._pressures(services)
        self._update_feedback(pressures)
        self.last_pressures = pressures
        ranked = sorted(
            pressures.values(), key=lambda item: (-item.pressure, item.service_id)
        )[: self.config.adaptation_top_k_services]
        if not ranked:
            self.window_records.clear()
            self.pending_actions = []
            return []

        memory = self._memory_usage(services)
        pressure_values = [signal.pressure for signal in ranked]
        decisions: list[MigrationAction] = []
        for signal in ranked:
            service = services[signal.service_id]
            source = self._source_replica(service, signal)
            source_plane = source // self.config.sats_per_plane
            raw_candidates = []

            def expected_cost(target: int) -> float:
                values = [
                    route_cost(origin, target, data_gb, current_time)
                    for origin, data_gb in signal.routing_samples
                ]
                finite = [value for value in values if math.isfinite(value)]
                return sum(finite) / len(finite) if finite else signal.mean_stage_cost

            raw_candidates.append(
                ("stay", source, source_plane, expected_cost(source), compute_load(source, current_time))
            )
            for plane in range(self.config.num_planes):
                if plane == source_plane:
                    continue
                nodes = range(
                    plane * self.config.sats_per_plane,
                    (plane + 1) * self.config.sats_per_plane,
                )
                feasible = [
                    node for node in nodes
                    if node not in service.replicas
                    and memory[node] + service.memory_requirement_gb
                    <= resources[node].memory_capacity_gb + 1.0e-9
                ]
                if not feasible:
                    continue
                target = min(feasible, key=lambda node: (compute_load(node, current_time), node))
                network = expected_cost(target)
                network += (
                    self.config.delay_weight
                    * service.activation_delay_s
                    / max(self.config.latency_scale_s, 1.0e-9)
                )
                if math.isfinite(network):
                    raw_candidates.append(
                        ("relocate", target, plane, network, compute_load(target, current_time))
                    )

            network_values = [item[3] for item in raw_candidates]
            load_values = [item[4] for item in raw_candidates]
            scored = []
            for action, target, plane, network, load in raw_candidates:
                features = (
                    self._minmax(signal.pressure, pressure_values),
                    self._minmax(network, network_values),
                    self._minmax(load, load_values),
                )
                score = self._ucb(features)
                scored.append((score, action == "stay", -target, action, target, plane, features, network, load))
            score, _, _, action, target, plane, features, network, load = max(scored)
            if action == "relocate":
                service.replicas.remove(source)
                service.replicas.append(target)
                service.replicas.sort()
                memory[source] -= service.memory_requirement_gb
                memory[target] += service.memory_requirement_gb
                self.total_relocations += 1
            self.action_counts[action] += 1
            self.total_pulls += 1
            decisions.append(
                MigrationAction(
                    action=action,
                    service_id=service.service_id,
                    source_node=source,
                    target_node=target,
                    source_plane=source_plane,
                    target_plane=plane,
                    features=features,
                    pressure=signal.pressure,
                    expected_network_cost=network,
                    target_compute_load=load,
                    ucb_score=score,
                    baseline_service_cost=max(signal.mean_stage_cost, 1.0e-9),
                    activation_delay_s=(
                        service.activation_delay_s if action == "relocate" else 0.0
                    ),
                )
            )

        self.pending_actions = decisions
        self.window_records.clear()
        return decisions

    def summary(self) -> dict:
        return {
            "policy": "service_pressure_shared_linear_ucb",
            "windows": self.total_windows,
            "pulls": self.total_pulls,
            "relocations": self.total_relocations,
            "feedback_updates": self.total_feedback_updates,
            "stay_actions": self.action_counts["stay"],
            "relocate_actions": self.action_counts["relocate"],
            "tracked_pressures": len(self.last_pressures),
        }

    def state_dict(self) -> dict:
        return {
            "matrix_a": self.matrix_a.tolist(),
            "vector_b": self.vector_b.tolist(),
            "window_records": [vars(record) for record in self.window_records],
            "requests_in_window": self.requests_in_window,
            "pending_actions": [vars(action) for action in self.pending_actions],
            "pressure_ewma": dict(self.pressure_ewma),
            "total_windows": self.total_windows,
            "total_pulls": self.total_pulls,
            "total_relocations": self.total_relocations,
            "total_feedback_updates": self.total_feedback_updates,
            "action_counts": dict(self.action_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        self.matrix_a = np.asarray(state["matrix_a"], dtype=np.float64)
        self.vector_b = np.asarray(state["vector_b"], dtype=np.float64)
        self.window_records = [
            StageExecutionRecord(**record) for record in state.get("window_records", [])
        ]
        self.requests_in_window = int(state.get("requests_in_window", 0))
        self.pending_actions = [
            MigrationAction(**action) for action in state.get("pending_actions", [])
        ]
        self.pressure_ewma = {
            int(key): float(value) for key, value in state.get("pressure_ewma", {}).items()
        }
        self.total_windows = int(state.get("total_windows", 0))
        self.total_pulls = int(state.get("total_pulls", 0))
        self.total_relocations = int(state.get("total_relocations", 0))
        self.total_feedback_updates = int(state.get("total_feedback_updates", 0))
        self.action_counts = Counter(state.get("action_counts", {}))
