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
    template_id: int | None = None
    request_arrival_time_s: float = 0.0
    stage_index: int = 0
    chain_length: int = 1
    stage_start_time_s: float = 0.0
    stage_finish_time_s: float = 0.0
    topology_slot: int = 0
    destination_node: int = -1
    route_slot_crossings: int = 0
    candidate_nodes: tuple[int, ...] = ()
    candidate_hop_distances: tuple[float, ...] = ()
    candidate_bottleneck_rates: tuple[float, ...] = ()
    candidate_compute_queues: tuple[float, ...] = ()
    success: bool = True
    failure_reason: str = ""


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
    replica_cost_contribution: dict[int, float] = field(default_factory=dict)
    replica_counts: Counter = field(default_factory=Counter)
    records: list[StageExecutionRecord] = field(default_factory=list)


@dataclass
class MigrationAction:
    action: str
    service_id: int
    source_node: int | None
    target_node: int | None
    source_plane: int | None
    target_plane: int | None
    features: tuple[float, ...]
    pressure: float
    expected_service_cost: float
    target_compute_load: float
    predicted_gain: float
    ucb_score: float
    baseline_service_cost: float
    replica_count_before: int
    replica_count_after: int
    activation_delay_s: float = 0.0
    feedback_reward: float = 0.0


class BanditReplicaAdapter:
    """Trace-driven four-action replica number and placement adaptation."""

    ACTIONS = ("no_op", "relocate", "scale_out", "scale_in")
    feature_dim = 8  # four shared context values plus four-action one-hot

    def __init__(self, config):
        self.config = config
        self.matrix_a = np.eye(self.feature_dim, dtype=np.float64)
        self.vector_b = np.zeros(self.feature_dim, dtype=np.float64)
        self.window_records: list[StageExecutionRecord] = []
        self.requests_in_window = 0
        self.window_start_slot: int | None = None
        self.pending_actions: list[MigrationAction] = []
        self.pressure_ewma: dict[int, float] = {}
        self.total_windows = 0
        self.total_pulls = 0
        self.total_feedback_updates = 0
        self.action_counts = Counter()
        self.last_pressures: dict[int, ServicePressure] = {}

    def observe_stage(self, record: StageExecutionRecord) -> None:
        self.window_records.append(record)

    def start_fresh_window(self, current_time: float) -> None:
        """Start joint training without treating PPO pretraining as one window."""
        self.window_records.clear()
        self.requests_in_window = 0
        self.window_start_slot = int(current_time // self.config.slot_duration_s)
        self.pending_actions.clear()
        self.last_pressures.clear()

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        return float(np.quantile(np.asarray(values, dtype=float), q)) if values else 0.0

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
        grouped: dict[int, list[StageExecutionRecord]] = defaultdict(list)
        for record in self.window_records:
            grouped[record.service_id].append(record)
        total_cost = sum(record.normalized_cost for record in self.window_records)
        result = {}
        for service_id, records in grouped.items():
            service = services[service_id]
            counts = Counter(record.serving_node for record in records)
            all_counts = [counts.get(node, 0) for node in service.replicas]
            mean_count = sum(all_counts) / max(1, len(all_counts))
            imbalance = (
                max(all_counts) / max(1.0e-9, mean_count) - 1.0
                if sum(all_counts) else 0.0
            )
            service_cost = sum(record.normalized_cost for record in records)
            cost_by_replica = defaultdict(float)
            for record in records:
                cost_by_replica[record.serving_node] += record.normalized_cost
            contribution = {
                node: cost_by_replica[node] / max(service_cost, 1.0e-9)
                for node in service.replicas
            }
            impact = service_cost / max(total_cost, 1.0e-9)
            route_pressure = self._tail_pressure([item.route_delay_s for item in records])
            queue_pressure = self._tail_pressure([item.compute_queue_s for item in records])
            opportunity = max(
                self._saturate(route_pressure),
                self._saturate(queue_pressure),
                self._saturate(imbalance),
            )
            raw_pressure = impact * opportunity
            previous = self.pressure_ewma.get(service_id, raw_pressure)
            decay = min(1.0, max(0.0, self.config.pressure_ewma))
            pressure = decay * previous + (1.0 - decay) * raw_pressure
            self.pressure_ewma[service_id] = pressure
            result[service_id] = ServicePressure(
                service_id=service_id,
                invocation_count=len(records),
                impact=impact,
                route_pressure=route_pressure,
                queue_pressure=queue_pressure,
                replica_imbalance=imbalance,
                pressure=pressure,
                mean_stage_cost=service_cost / len(records),
                replica_cost_contribution=contribution,
                replica_counts=counts,
                records=records,
            )
        return result

    def _update_feedback(self, pressures: dict[int, ServicePressure]) -> None:
        for action in self.pending_actions:
            after = pressures.get(action.service_id)
            after_cost = (
                after.mean_stage_cost
                if after else action.baseline_service_cost
            )
            baseline = max(action.baseline_service_cost, 1.0e-9)
            reward = (action.baseline_service_cost - after_cost) / baseline
            action.feedback_reward = reward
            features = np.asarray(action.features, dtype=float)
            self.matrix_a += np.outer(features, features)
            self.vector_b += reward * features
            self.total_feedback_updates += 1

    def _ucb(
        self,
        features: tuple[float, ...],
        inverse: np.ndarray | None = None,
        theta: np.ndarray | None = None,
    ) -> float:
        x = np.asarray(features, dtype=float)
        if inverse is None:
            inverse = np.linalg.inv(self.matrix_a)
        if theta is None:
            theta = inverse @ self.vector_b
        uncertainty = math.sqrt(max(0.0, float(x @ inverse @ x)))
        return float(theta @ x + self.config.bandit_exploration * uncertainty)

    @staticmethod
    def _memory_usage(services: dict[int, Microservice]) -> dict[int, float]:
        usage = defaultdict(float)
        for service in services.values():
            for node in service.replicas:
                usage[node] += service.memory_requirement_gb
        return usage

    def record_request(self) -> None:
        """Count one independent target request in the current trace window."""
        self.requests_in_window += 1

    def _window_due(self, current_time: float) -> bool:
        if self.config.deployment_window_requests is not None:
            return self.requests_in_window >= self.config.deployment_window_requests
        slot = int(current_time // self.config.slot_duration_s)
        if self.window_start_slot is None:
            self.window_start_slot = slot
            return False
        return slot - self.window_start_slot >= self.config.adaptation_window_slots

    def _representative_targets(
        self,
        service: Microservice,
        signal: ServicePressure,
        services: dict[int, Microservice],
        resources: dict[int, SatelliteResource],
        compute_load: Callable[[int, float], float],
    ) -> list[int]:
        memory = self._memory_usage(services)
        times = [record.stage_start_time_s for record in signal.records]
        if not times:
            times = [0.0]
        result = []
        for plane in range(self.config.num_planes):
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
            if feasible:
                result.append(
                    min(
                        feasible,
                        key=lambda node: (
                            sum(compute_load(node, time_s) for time_s in times) / len(times),
                            node,
                        ),
                    )
                )
        return result

    def _target_cost(
        self,
        record: StageExecutionRecord,
        target: int,
        route_cost: Callable[[int, int, float, float], float],
        compute_load: Callable[[int, float], float],
        replay_cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        key = (id(record), int(target))
        if replay_cache is not None and key in replay_cache:
            return replay_cache[key]
        network = route_cost(
            record.source_node, target, record.data_gb, record.stage_start_time_s
        )
        if not math.isfinite(network):
            result = math.inf
        else:
            result = network + compute_load(target, record.stage_start_time_s)
        if replay_cache is not None:
            replay_cache[key] = result
        return result

    def _best_target_action(
        self,
        action: str,
        source: int | None,
        service: Microservice,
        signal: ServicePressure,
        targets: list[int],
        route_cost,
        compute_load,
        replay_cache,
    ) -> tuple[int, float, float] | None:
        records = signal.records[: self.config.adaptation_trace_sample_limit]
        best = None
        for target in targets:
            costs = []
            for record in records:
                candidate = self._target_cost(
                    record,
                    target,
                    route_cost,
                    compute_load,
                    replay_cache,
                )
                if action == "scale_out":
                    costs.append(min(record.normalized_cost, candidate))
                elif record.serving_node == source:
                    costs.append(candidate)
                else:
                    costs.append(record.normalized_cost)
            if not costs or not all(math.isfinite(value) for value in costs):
                continue
            expected = sum(costs) / len(costs)
            load = sum(
                compute_load(target, record.stage_start_time_s) for record in records
            ) / max(1, len(records))
            item = (expected, load, target)
            if best is None or item < best:
                best = item
        if best is None:
            return None
        expected, load, target = best
        return target, expected, load

    def _scale_in_cost(
        self,
        source,
        service,
        signal,
        route_cost,
        compute_load,
        replay_cache,
    ) -> float:
        remaining = [node for node in service.replicas if node != source]
        records = signal.records[: self.config.adaptation_trace_sample_limit]
        values = []
        for record in records:
            if record.serving_node != source:
                values.append(record.normalized_cost)
                continue
            alternatives = [
                self._target_cost(
                    record,
                    target,
                    route_cost,
                    compute_load,
                    replay_cache,
                )
                for target in remaining
            ]
            finite = [value for value in alternatives if math.isfinite(value)]
            if not finite:
                return math.inf
            values.append(min(finite))
        return sum(values) / max(1, len(values))

    def adapt_if_due(
        self,
        services: dict[int, Microservice],
        resources: dict[int, SatelliteResource],
        current_time: float,
        route_cost: Callable[[int, int, float, float], float],
        compute_load: Callable[[int, float], float],
    ) -> list[MigrationAction]:
        if not self._window_due(current_time):
            return []
        self.requests_in_window = 0
        self.window_start_slot = int(current_time // self.config.slot_duration_s)
        self.total_windows += 1
        pressures = self._pressures(services)
        self._update_feedback(pressures)
        self.last_pressures = pressures
        ranked = sorted(
            pressures.values(), key=lambda item: (-item.pressure, item.service_id)
        )[: self.config.adaptation_top_k_services]
        decisions = []
        pressure_max = max((item.pressure for item in ranked), default=1.0)
        min_replicas, max_replicas = self.config.replica_count_range
        replay_cache: dict[tuple[int, int], float] = {}
        compute_cache: dict[tuple[int, float], float] = {}

        def cached_compute_load(node: int, time_s: float) -> float:
            key = (int(node), float(time_s))
            if key not in compute_cache:
                compute_cache[key] = compute_load(node, time_s)
            return compute_cache[key]

        inverse = np.linalg.inv(self.matrix_a)
        theta = inverse @ self.vector_b

        for signal in ranked:
            service = services[signal.service_id]
            count_before = len(service.replicas)
            baseline_mean = max(signal.mean_stage_cost, 1.0e-9)
            baseline_total = baseline_mean
            bottleneck = max(
                service.replicas,
                key=lambda node: (signal.replica_cost_contribution.get(node, 0.0), -node),
            )
            redundant = min(
                service.replicas,
                key=lambda node: (signal.replica_cost_contribution.get(node, 0.0), node),
            )
            targets = self._representative_targets(
                service, signal, services, resources, cached_compute_load
            )
            candidates = [
                ("no_op", None, None, baseline_mean, 0.0, count_before)
            ]
            relocate = self._best_target_action(
                "relocate",
                bottleneck,
                service,
                signal,
                targets,
                route_cost,
                cached_compute_load,
                replay_cache,
            )
            if relocate is not None:
                target, expected, load = relocate
                candidates.append(
                    ("relocate", bottleneck, target, expected, load, count_before)
                )
            if count_before < max_replicas:
                scale_out = self._best_target_action(
                    "scale_out",
                    None,
                    service,
                    signal,
                    targets,
                    route_cost,
                    cached_compute_load,
                    replay_cache,
                )
                if scale_out is not None:
                    target, expected, load = scale_out
                    candidates.append(
                        ("scale_out", None, target, expected, load, count_before + 1)
                    )
            if count_before > min_replicas:
                expected = self._scale_in_cost(
                    redundant,
                    service,
                    signal,
                    route_cost,
                    cached_compute_load,
                    replay_cache,
                )
                if math.isfinite(expected):
                    candidates.append(
                        ("scale_in", redundant, None, expected, 0.0, count_before - 1)
                    )

            scored = []
            for action, source, target, expected, load, count_after in candidates:
                expected_total = expected
                gain = (baseline_total - expected_total) / max(baseline_total, 1.0e-9)
                one_hot = tuple(float(action == name) for name in self.ACTIONS)
                features = (
                    signal.pressure / max(pressure_max, 1.0e-9),
                    max(-1.0, min(1.0, gain)),
                    count_before / max(1, max_replicas),
                    min(1.0, max(0.0, load)),
                    *one_hot,
                )
                score = self._ucb(features, inverse, theta)
                scored.append(
                    (score, action == "no_op", action, source, target, expected_total,
                     load, gain, features, count_after)
                )
            (
                score, _, action, source, target, expected_total, load, gain,
                features, count_after,
            ) = max(scored)

            if action == "relocate":
                service.replicas.remove(source)
                service.replicas.append(target)
            elif action == "scale_out":
                service.replicas.append(target)
            elif action == "scale_in":
                service.replicas.remove(source)
            service.replicas.sort()
            self.action_counts[action] += 1
            self.total_pulls += 1
            decisions.append(
                MigrationAction(
                    action=action,
                    service_id=service.service_id,
                    source_node=source,
                    target_node=target,
                    source_plane=(source // self.config.sats_per_plane if source is not None else None),
                    target_plane=(target // self.config.sats_per_plane if target is not None else None),
                    features=features,
                    pressure=signal.pressure,
                    expected_service_cost=expected_total,
                    target_compute_load=load,
                    predicted_gain=gain,
                    ucb_score=score,
                    baseline_service_cost=baseline_total,
                    replica_count_before=count_before,
                    replica_count_after=count_after,
                    activation_delay_s=(service.activation_delay_s if action in {"relocate", "scale_out"} else 0.0),
                )
            )

        self.pending_actions = decisions
        self.window_records.clear()
        return decisions

    def close_request(
        self,
        services: dict[int, Microservice],
        resources: dict[int, SatelliteResource],
        current_time: float,
        route_cost: Callable[[int, int, float, float], float],
        compute_load: Callable[[int, float], float],
    ) -> list[MigrationAction]:
        """Compatibility wrapper for request-count based callers and tests."""
        self.record_request()
        return self.adapt_if_due(
            services, resources, current_time, route_cost, compute_load
        )

    def summary(self) -> dict:
        return {
            "policy": "trace_pressure_four_action_shared_linear_ucb",
            "windows": self.total_windows,
            "pulls": self.total_pulls,
            "feedback_updates": self.total_feedback_updates,
            "no_op_actions": self.action_counts["no_op"],
            "relocate_actions": self.action_counts["relocate"],
            "scale_out_actions": self.action_counts["scale_out"],
            "scale_in_actions": self.action_counts["scale_in"],
            "tracked_pressures": len(self.last_pressures),
        }

    def state_dict(self) -> dict:
        return {
            "matrix_a": self.matrix_a.tolist(),
            "vector_b": self.vector_b.tolist(),
            "window_records": [vars(record) for record in self.window_records],
            "requests_in_window": self.requests_in_window,
            "window_start_slot": self.window_start_slot,
            "pending_actions": [vars(action) for action in self.pending_actions],
            "pressure_ewma": dict(self.pressure_ewma),
            "total_windows": self.total_windows,
            "total_pulls": self.total_pulls,
            "total_feedback_updates": self.total_feedback_updates,
            "action_counts": dict(self.action_counts),
        }

    def load_state_dict(self, state: dict) -> None:
        matrix = np.asarray(state.get("matrix_a", []), dtype=np.float64)
        vector = np.asarray(state.get("vector_b", []), dtype=np.float64)
        if matrix.shape == (self.feature_dim, self.feature_dim) and vector.shape == (self.feature_dim,):
            self.matrix_a, self.vector_b = matrix, vector
        self.window_records = [
            StageExecutionRecord(**record) for record in state.get("window_records", [])
        ]
        self.requests_in_window = int(state.get("requests_in_window", 0))
        self.window_start_slot = state.get("window_start_slot")
        self.pending_actions = []
        for action in state.get("pending_actions", []):
            try:
                restored = MigrationAction(**action)
                if len(restored.features) == self.feature_dim:
                    self.pending_actions.append(restored)
            except TypeError:
                # Old two-action checkpoints do not contain the four-action
                # context fields and cannot provide valid delayed feedback.
                continue
        self.pressure_ewma = {
            int(key): float(value) for key, value in state.get("pressure_ewma", {}).items()
        }
        self.total_windows = int(state.get("total_windows", 0))
        self.total_pulls = int(state.get("total_pulls", 0))
        self.total_feedback_updates = int(state.get("total_feedback_updates", 0))
        self.action_counts = Counter(state.get("action_counts", {}))
