from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path

import numpy as np

from ..config import SimulationConfig
from ..domain.constellation import orbit_plane
from ..domain.service import Microservice, deployment_matrix, memory_usage_by_node
from ..network.routing import route_data
from ..network.topology import slot_from_time


FEATURE_DIM = 3
PRESSURE_EWMA_FACTOR = 0.5


@dataclass
class MigrationAction:
    """One slow-timescale decision for a pressured service."""

    action: str  # stay or relocate
    service_id: int
    arm_key: tuple
    source_node: int
    target_node: int
    source_plane: int
    target_plane: int
    context_features: tuple[float, float, float]
    service_pressure: float
    expected_network_cost: float
    target_compute_load: float
    estimated_reward: float
    baseline_service_cost: float
    execution_feedback_reward: float = 0.0
    activation_delay_s: float = 0.0
    memory_requirement_gb: float = 0.0


@dataclass
class ServicePressureSignal:
    service_id: int
    invocation_count: int = 0
    global_impact: float = 0.0
    route_pressure: float = 0.0
    waiting_pressure: float = 0.0
    replica_imbalance: float = 0.0
    actionability: float = 0.0
    pressure: float = 0.0
    mean_stage_cost: float = 0.0
    mean_network_cost: float = 0.0
    replica_stage_costs: dict[int, float] = field(default_factory=dict)
    replica_execution_counts: Counter[int] = field(default_factory=Counter)
    routing_samples: list[tuple[int, float]] = field(default_factory=list)


class ReplicaPlacementMigrationAgent:
    """Shared linear-contextual-UCB policy for local replica activation.

    All service images are assumed to be pre-provisioned on every satellite.
    A relocation therefore activates a local image, redirects new requests,
    drains the old replica, and deactivates it.  No image is transferred over
    an inter-satellite link and the active replica count remains unchanged.
    """

    def __init__(self, config: SimulationConfig, exploration_c: float = 1.25):
        self.config = config
        self.exploration_c = float(exploration_c)
        self.matrix_a = np.eye(FEATURE_DIM, dtype=float)
        self.vector_b = np.zeros(FEATURE_DIM, dtype=float)
        self.total_pulls = 0
        self.total_applied_actions = 0
        self.total_execution_feedback_updates = 0
        self.applied_action_type_counts = Counter()
        self.decision_stats: dict[tuple, dict[str, float]] = {}
        self.service_feedback_metrics: dict[int, ServicePressureSignal] = {}
        self._pressure_ewma: dict[int, float] = {}
        self.last_apply_summary: dict[str, float] = {}

    def observe_failed_replicas(self, results: list[dict]) -> None:
        """Compatibility hook; routing failures are not a pressure or reward term."""

    def observe_service_pressure_feedback(self, results: list[dict]) -> None:
        if not results:
            return
        current = self._collect_service_feedback_metrics(results)
        for service_id, signal in current.items():
            previous = self._pressure_ewma.get(service_id, signal.pressure)
            smoothed = (
                PRESSURE_EWMA_FACTOR * signal.pressure
                + (1.0 - PRESSURE_EWMA_FACTOR) * previous
            )
            self._pressure_ewma[service_id] = smoothed
            signal.pressure = smoothed
        self.service_feedback_metrics = current

    def observe_execution_feedback(
        self, actions: list[MigrationAction], results: list[dict]
    ) -> None:
        current = self._collect_service_feedback_metrics(results) if results else {}
        for action in actions:
            after = current.get(action.service_id)
            after_cost = after.mean_stage_cost if after else action.baseline_service_cost
            baseline = max(1.0e-9, action.baseline_service_cost)
            reward = (action.baseline_service_cost - after_cost) / baseline
            action.execution_feedback_reward = reward
            features = np.asarray(action.context_features, dtype=float)
            self.matrix_a += np.outer(features, features)
            self.vector_b += reward * features
            self.total_execution_feedback_updates += 1
            stats = self.decision_stats.setdefault(
                action.arm_key, {"count": 0.0, "reward_sum": 0.0}
            )
            stats["count"] += 1.0
            stats["reward_sum"] += reward
        self.observe_service_pressure_feedback(results)

    def estimate_service_pressure(
        self, requests, microservices: dict[int, Microservice] | None = None
    ) -> dict[int, ServicePressureSignal]:
        signals = {
            sid: ServicePressureSignal(**vars(signal))
            for sid, signal in self.service_feedback_metrics.items()
        }
        request_counts = Counter(
            int(service_id) for request in requests for service_id in request.services
        )
        total = sum(request_counts.values())
        for service_id, count in request_counts.items():
            if service_id not in signals:
                share = count / max(1, total)
                signals[service_id] = ServicePressureSignal(
                    service_id=service_id,
                    invocation_count=count,
                    global_impact=share,
                    pressure=share,
                )
        if microservices is not None:
            for service_id, signal in signals.items():
                service = microservices.get(service_id)
                if service is None or not service.replicas:
                    continue
                counts = [
                    float(signal.replica_execution_counts.get(node_id, 0))
                    for node_id in service.replicas
                ]
                mean_count = sum(counts) / len(counts)
                signal.replica_imbalance = (
                    max(counts) / max(1.0e-9, mean_count) - 1.0
                    if sum(counts) > 0.0 else 0.0
                )
                signal.actionability = max(
                    self._saturate(signal.route_pressure),
                    self._saturate(signal.waiting_pressure),
                    self._saturate(signal.replica_imbalance),
                )
                raw_pressure = signal.global_impact * signal.actionability
                signal.pressure = 0.5 * raw_pressure + 0.5 * signal.pressure
        return signals

    def apply(
        self,
        microservices: dict[int, Microservice],
        requests,
        context: dict,
        max_actions: int | None = None,
    ) -> list[MigrationAction]:
        del max_actions  # Top-K_m determines the number of per-service decisions.
        pressure = self.estimate_service_pressure(requests, microservices)
        ranked = sorted(
            pressure.values(), key=lambda item: (-item.pressure, item.service_id)
        )[: self.config.bandit_pressure_top_k_services]
        if not ranked:
            self.last_apply_summary = {"selected_service_count": 0}
            return []

        memory_used = memory_usage_by_node(microservices, self.config.total_sats)
        start_time = (
            max((request.start_time for request in requests), default=0.0)
            + float(context["slot_duration"])
        )
        decisions: list[MigrationAction] = []
        summary = Counter(selected_service_count=len(ranked))
        pressure_values = [item.pressure for item in ranked]

        for signal in ranked:
            service = microservices[signal.service_id]
            source = self._highest_cost_source(service, signal)
            candidates = self._candidate_actions(
                service, signal, source, memory_used, context, start_time
            )
            if not candidates:
                summary["no_candidate_count"] += 1
                continue

            network_values = [item[3] for item in candidates]
            load_values = [item[4] for item in candidates]
            scored = []
            for action, target, target_plane, network_cost, target_load in candidates:
                features = (
                    self._minmax(signal.pressure, pressure_values),
                    self._minmax(network_cost, network_values),
                    self._minmax(target_load, load_values),
                )
                score = self._ucb_score(features)
                # Deterministic tie-breaking prefers staying when evidence is equal.
                scored.append((score, action == "stay", -target, action, target, target_plane, features, network_cost, target_load))
            selected = max(scored)
            score, _, _, action, target, target_plane, features, network_cost, target_load = selected
            arm_key = (action, service.service_id, orbit_plane(source, self.config), target_plane)

            if action == "relocate":
                service.replicas.remove(source)
                service.replicas.append(target)
                service.replicas.sort()
                memory_used[source] -= service.memory_requirement_gb
                memory_used[target] += service.memory_requirement_gb
                self.total_applied_actions += 1
                summary["relocated_count"] += 1
            else:
                summary["stay_count"] += 1

            decision = MigrationAction(
                action=action,
                service_id=service.service_id,
                arm_key=arm_key,
                source_node=source,
                target_node=target,
                source_plane=orbit_plane(source, self.config),
                target_plane=target_plane,
                context_features=features,
                service_pressure=signal.pressure,
                expected_network_cost=network_cost,
                target_compute_load=target_load,
                estimated_reward=score,
                baseline_service_cost=max(1.0e-9, signal.mean_stage_cost),
                activation_delay_s=service.startup_delay_s if action == "relocate" else 0.0,
                memory_requirement_gb=service.memory_requirement_gb,
            )
            decisions.append(decision)
            self.total_pulls += 1
            self.applied_action_type_counts[action] += 1

        context["deployment_by_node"] = deployment_matrix(microservices)
        self.last_apply_summary = dict(summary)
        return decisions

    def _candidate_actions(
        self, service, signal, source, memory_used, context, start_time
    ) -> list[tuple[str, int, int, float, float]]:
        source_plane = orbit_plane(source, self.config)
        result = [(
            "stay",
            source,
            source_plane,
            self._expected_network_cost(signal, source, context, start_time),
            self._compute_load(source, context, start_time),
        )]
        for plane in range(self.config.num_planes):
            if plane == source_plane:
                continue
            target = self._best_feasible_node_in_plane(
                plane, service, memory_used, context, start_time
            )
            if target is None:
                continue
            network_cost = self._expected_network_cost(
                signal, target, context, start_time
            )
            if not math.isfinite(network_cost):
                continue
            result.append((
                "relocate",
                target,
                plane,
                network_cost,
                self._compute_load(target, context, start_time),
            ))
        return result

    def _best_feasible_node_in_plane(
        self, plane, service, memory_used, context, start_time
    ) -> int | None:
        resources = context["satellite_resources"]
        candidates = []
        for node_id in range(1, self.config.total_sats + 1):
            if orbit_plane(node_id, self.config) != plane or node_id in service.replicas:
                continue
            capacity = resources[node_id].memory_capacity_gb
            if memory_used[node_id] + service.memory_requirement_gb > capacity + 1.0e-9:
                continue
            candidates.append(node_id)
        return min(
            candidates,
            key=lambda node: (
                self._compute_load(node, context, start_time),
                memory_used[node] / max(1.0e-9, resources[node].memory_capacity_gb),
                node,
            ),
            default=None,
        )

    @staticmethod
    def _highest_cost_source(service, signal) -> int:
        return max(
            service.replicas,
            key=lambda node: (
                signal.replica_stage_costs.get(node, 0.0),
                -signal.replica_execution_counts.get(node, 0),
                -node,
            ),
        )

    def _expected_network_cost(
        self, signal, target: int, context: dict, start_time: float
    ) -> float:
        if not signal.routing_samples:
            return 0.0
        costs = []
        # Four representative arrivals keep candidate construction lightweight.
        for origin, data_gb in signal.routing_samples[:4]:
            route = route_data(origin, target, data_gb, start_time, context)
            if not route.get("reachable", False):
                continue
            costs.append(
                self.config.delay_weight * float(route["delay_s"])
                + self.config.energy_weight
                * float(route["communication_energy_j"]) / 1000.0
            )
        return sum(costs) / len(costs) if costs else signal.mean_network_cost

    @staticmethod
    def _compute_load(node_id: int, context: dict, start_time: float) -> float:
        _, slot_mod = slot_from_time(
            start_time, context["slot_duration"], context["slot_count"]
        )
        return float(
            context.get("compute_utilization_table", {}).get(slot_mod, {}).get(node_id, 0.0)
        )

    def _ucb_score(self, features: tuple[float, float, float]) -> float:
        x = np.asarray(features, dtype=float)
        inverse = np.linalg.inv(self.matrix_a)
        theta = inverse @ self.vector_b
        uncertainty = math.sqrt(max(0.0, float(x @ inverse @ x)))
        return float(theta @ x + self.exploration_c * uncertainty)

    def _collect_service_feedback_metrics(
        self, results: list[dict]
    ) -> dict[int, ServicePressureSignal]:
        raw = defaultdict(lambda: {
            "route_delays": [], "network_costs": [], "waits": [], "stage_costs": [],
            "costs_by_node": defaultdict(list), "counts": Counter(), "routing_samples": [],
        })
        total_request_cost = 0.0
        route_by_key = {}
        for result in results:
            request = result.get("request") or {}
            request_id = int(request.get("request_id", -1))
            for route in result.get("route_details") or []:
                stage = route.get("stage")
                if isinstance(stage, int):
                    route_by_key[(request_id, stage)] = route
            delay = float(result.get("total_delay_s", math.inf))
            energy = float(result.get("total_energy_j", math.inf))
            if math.isfinite(delay) and math.isfinite(energy):
                total_request_cost += self.config.delay_weight * delay + self.config.energy_weight * energy / 1000.0

            for step in result.get("execution_plan") or []:
                service_id = int(step["service_id"])
                stage = int(step.get("stage", -1))
                node = int(step["satellite_node"])
                route = route_by_key.get((request_id, stage), {})
                route_delay = float(route.get("communication_delay_s", 0.0))
                route_energy = float(route.get("communication_energy_j", 0.0))
                wait = float(step.get("queue_delay_s", 0.0))
                compute_delay = float(step.get("compute_delay_s", 0.0))
                compute_energy = float(step.get("compute_energy_j", 0.0))
                cost = self.config.delay_weight * (route_delay + wait + compute_delay) + self.config.energy_weight * (route_energy + compute_energy) / 1000.0
                values = raw[service_id]
                values["route_delays"].append(route_delay)
                values["network_costs"].append(
                    self.config.delay_weight * route_delay
                    + self.config.energy_weight * route_energy / 1000.0
                )
                values["waits"].append(wait)
                values["stage_costs"].append(cost)
                values["costs_by_node"][node].append(cost)
                values["counts"][node] += 1
                origin = route.get("source_node")
                if origin is not None:
                    values["routing_samples"].append(
                        (int(origin), float(route.get("data_gb", 0.0)))
                    )

        signals = {}
        for service_id, values in raw.items():
            stage_cost_sum = sum(values["stage_costs"])
            impact = stage_cost_sum / max(1.0e-9, total_request_cost)
            route_pressure = self._tail_pressure(values["route_delays"])
            waiting_pressure = self._tail_pressure(values["waits"])
            counts = list(values["counts"].values())
            imbalance = (max(counts) / max(1.0e-9, sum(counts) / len(counts)) - 1.0) if counts else 0.0
            actionability = max(self._saturate(route_pressure), self._saturate(waiting_pressure), self._saturate(imbalance))
            signals[service_id] = ServicePressureSignal(
                service_id=service_id,
                invocation_count=len(values["stage_costs"]),
                global_impact=impact,
                route_pressure=route_pressure,
                waiting_pressure=waiting_pressure,
                replica_imbalance=imbalance,
                actionability=actionability,
                pressure=impact * actionability,
                mean_stage_cost=stage_cost_sum / max(1, len(values["stage_costs"])),
                mean_network_cost=(
                    sum(values["network_costs"]) / len(values["network_costs"])
                    if values["network_costs"] else 0.0
                ),
                replica_stage_costs={node: sum(costs) / len(costs) for node, costs in values["costs_by_node"].items()},
                replica_execution_counts=values["counts"],
                routing_samples=values["routing_samples"],
            )
        return signals

    @classmethod
    def _tail_pressure(cls, values: list[float]) -> float:
        if not values:
            return 0.0
        median = cls._percentile(values, 0.5)
        p95 = cls._percentile(values, 0.95)
        return max(0.0, p95 / max(1.0e-9, median) - 1.0)

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
        return ordered[index]

    @staticmethod
    def _saturate(value: float) -> float:
        value = max(0.0, value)
        return value / (1.0 + value)

    @staticmethod
    def _minmax(value: float, values: list[float]) -> float:
        low, high = min(values), max(values)
        return 0.0 if high <= low + 1.0e-12 else (value - low) / (high - low)

    def export_arm_stats(self) -> list[dict]:
        model = {
            "record_type": "model",
            "model_type": "shared_linear_contextual_ucb",
            "matrix_a": json.dumps(self.matrix_a.tolist()),
            "vector_b": json.dumps(self.vector_b.tolist()),
            "pull_count": self.total_pulls,
            "execution_count": self.total_execution_feedback_updates,
        }
        rows = [model]
        for arm, stats in sorted(self.decision_stats.items(), key=lambda item: str(item[0])):
            count = stats["count"]
            rows.append({
                "record_type": "decision",
                "arm": json.dumps(arm),
                "action": arm[0],
                "service_id": arm[1],
                "source_plane": arm[2],
                "target_plane": arm[3],
                "pull_count": count,
                "reward_sum": stats["reward_sum"],
                "mean_reward": stats["reward_sum"] / count if count else 0.0,
            })
        return rows

    def load_arm_stats(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        model_rows = [row for row in rows if row.get("model_type") == "shared_linear_contextual_ucb"]
        if not model_rows:
            return 0  # Old UCB1 checkpoints are intentionally incompatible.
        model = model_rows[0]
        self.matrix_a = np.asarray(json.loads(model["matrix_a"]), dtype=float)
        self.vector_b = np.asarray(json.loads(model["vector_b"]), dtype=float)
        self.total_pulls = int(float(model.get("pull_count") or 0))
        self.total_execution_feedback_updates = int(float(model.get("execution_count") or 0))
        self.decision_stats.clear()
        for row in rows:
            if row.get("record_type") != "decision" or not row.get("arm"):
                continue
            arm = tuple(json.loads(row["arm"]))
            self.decision_stats[arm] = {
                "count": float(row.get("pull_count") or 0.0),
                "reward_sum": float(row.get("reward_sum") or 0.0),
            }
        return max(1, len(self.decision_stats))

    def summary(self) -> dict:
        means = [
            stats["reward_sum"] / stats["count"]
            for stats in self.decision_stats.values() if stats["count"]
        ]
        return {
            "policy": "shared_linear_contextual_ucb",
            "feature_dimension": FEATURE_DIM,
            "total_pulls": self.total_pulls,
            "known_arm_count": len(self.decision_stats),
            "positive_arm_count": sum(value > 0.0 for value in means),
            "average_arm_reward": sum(means) / len(means) if means else 0.0,
            "total_applied_actions": self.total_applied_actions,
            "total_execution_feedback_updates": self.total_execution_feedback_updates,
            "known_service_pressure_metric_count": len(self.service_feedback_metrics),
            **{f"total_selected_{action}_count": count for action, count in self.applied_action_type_counts.items()},
            **{f"last_{key}": value for key, value in self.last_apply_summary.items()},
        }
