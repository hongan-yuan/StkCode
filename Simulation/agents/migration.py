from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
import csv
import math
import json
from pathlib import Path

from ..config import SimulationConfig
from ..domain.constellation import orbit_plane, same_orbit
from ..domain.service import Microservice, deployment_matrix
from ..network.routing import route_data
from ..network.topology import slot_from_time


@dataclass
class MigrationAction:
    action: str
    service_id: int | None = None
    arm_key: tuple | None = None
    source_node: int | None = None
    target_node: int | None = None
    source_plane: int | None = None
    target_plane: int | None = None
    selected_source_node: int | None = None
    expected_saving: float = 0.0
    migration_cost: float = 0.0
    estimated_reward: float = 0.0
    execution_feedback_reward: float = 0.0
    migration_delay_s: float = 0.0
    migration_energy_j: float = 0.0
    migration_route_mode: str = ""
    cross_orbit_transfer: bool = False


@dataclass
class ServicePressureSignal:
    service_id: int
    request_count: float = 0.0
    average_route_delay_to_m: float = 0.0
    average_compute_waiting_time_of_m: float = 0.0
    route_failure_count_related_to_m: float = 0.0
    p95_delay_of_m: float = 0.0
    replica_utilization_imbalance: float = 0.0
    replica_execution_counts: Counter[int] = field(default_factory=Counter)


class ReplicaPlacementMigrationAgent:
    """
    Slow-layer multi-armed bandit migration agent.

    Each legal migration/placement candidate is treated as an arm.
    The default selection rule is UCB1 over estimated net benefit:

    reward = ExpectedSaving - MigrationCost

    This keeps the slow layer lightweight while still explicitly balancing
    exploration of uncertain placement actions and exploitation of historically
    profitable actions.
    """

    def __init__(self, config: SimulationConfig, exploration_c: float = 1.25):
        self.config = config
        self.exploration_c = exploration_c
        self.arm_stats: dict[tuple, dict[str, float]] = {}
        self.total_pulls = 0
        self.total_applied_actions = 0
        self.applied_action_type_counts = Counter()
        self.total_execution_feedback_updates = 0
        self.last_apply_summary: dict[str, float] = {}
        self.failed_replica_counts: Counter[tuple[int, int]] = Counter()
        self.total_failed_replica_observations = 0
        self.service_feedback_metrics: dict[int, ServicePressureSignal] = {}

    def estimate_service_pressure(
        self,
        requests,
        microservices: dict[int, Microservice] | None = None,
    ) -> dict[int, ServicePressureSignal]:
        service_pressure = {
            service_id: replace(signal, request_count=0.0)
            for service_id, signal in self.service_feedback_metrics.items()
        }
        for request in requests:
            for service_id in request.services:
                signal = service_pressure.setdefault(
                    service_id,
                    ServicePressureSignal(service_id=service_id),
                )
                signal.request_count += 1.0
        if microservices is not None:
            for service_id, signal in service_pressure.items():
                service = microservices.get(service_id)
                if service is None:
                    continue
                signal.replica_utilization_imbalance = self._replica_utilization_imbalance(
                    service, signal.replica_execution_counts
                )
        return service_pressure

    def apply(
            self,
            microservices: dict[int, Microservice],
            requests,
            context: dict,
            max_actions: int = 4,
    ) -> list[MigrationAction]:
        service_pressure = self.estimate_service_pressure(requests, microservices)
        service_count_by_node = self._service_count_by_node(microservices)
        actions: list[MigrationAction] = []
        migration_start_time = min((request.start_time for request in requests), default=0.0)
        candidates = self._candidate_arms(microservices, service_pressure, service_count_by_node)
        summary = Counter()

        for candidate in self._rank_candidates(candidates, microservices, service_pressure):
            if len(actions) >= max_actions:
                break
            action, service_id, source_plane, target_plane = candidate
            if action == "no-op":
                self._update_arm(candidate, 0.0, "estimated")
                continue
            service = microservices[service_id]
            concrete = self._resolve_arm(
                candidate, service, service_count_by_node, context, migration_start_time
            )
            if concrete is None:
                summary["unresolved_arm_count"] += 1
                continue
            source_node, target_node = concrete

            if not self._is_legal(action, source_node, target_node, service, service_count_by_node):
                summary["illegal_arm_count"] += 1
                continue

            migration_cost, migration_route = self._migration_cost(
                action, service, source_node, target_node, context, migration_start_time
            )
            if not math.isfinite(migration_cost):
                summary["failed_cost_count"] += 1
                self._update_arm(candidate, -1.0, "estimated")
                continue
            expected_saving = self._expected_saving(action, service_id, service_pressure)
            failure_relief = self._failure_relieving_bonus(action, service_id, source_node)
            placement_penalty = self._low_speed_placement_penalty(
                action, source_node, target_node, context, migration_start_time
            )
            reward = expected_saving + failure_relief - migration_cost - placement_penalty
            self._update_arm(candidate, reward, "estimated")
            if reward <= self.config.migration_safety_margin:
                summary["rejected_nonpositive_count"] += 1
                continue

            if action == "add":
                service.replicas.append(target_node)
                service.replicas.sort()
                service_count_by_node[target_node] += 1
            elif action == "remove":
                service.replicas.remove(source_node)
                service_count_by_node[source_node] -= 1
            elif action == "move":
                service.replicas.remove(source_node)
                service.replicas.append(target_node)
                service.replicas.sort()
                service_count_by_node[source_node] -= 1
                service_count_by_node[target_node] += 1

            actions.append(
                MigrationAction(
                    action=action,
                    service_id=service_id,
                    arm_key=candidate,
                    source_node=source_node,
                    target_node=target_node,
                    source_plane=source_plane,
                    target_plane=target_plane,
                    selected_source_node=migration_route.get("selected_source_node"),
                    expected_saving=expected_saving,
                    migration_cost=migration_cost,
                    estimated_reward=reward,
                    migration_delay_s=migration_route.get("delay_s", 0.0),
                    migration_energy_j=migration_route.get("communication_energy_j", 0.0),
                    migration_route_mode=migration_route.get("route_mode", ""),
                    cross_orbit_transfer=(
                            migration_route.get("selected_source_node") is not None
                            and target_node is not None
                            and not same_orbit(
                        migration_route["selected_source_node"],
                        target_node,
                        self.config,
                    )
                    ),
                )
            )
            self.total_applied_actions += 1
            self.applied_action_type_counts[action] += 1
            summary["applied_action_count"] += 1
            summary[f"applied_{action}_count"] += 1

        context["deployment_by_node"] = deployment_matrix(microservices)
        self.last_apply_summary = dict(summary)
        return actions

    def _service_count_by_node(
            self, microservices: dict[int, Microservice]
    ) -> dict[int, int]:
        deployment = deployment_matrix(microservices)
        return {
            node_id: len(deployment.get(node_id, set()))
            for node_id in range(1, self.config.total_sats + 1)
        }

    def _candidate_arms(
            self,
            microservices: dict[int, Microservice],
            service_pressure: dict[int, ServicePressureSignal],
            service_count_by_node: dict[int, int],
    ) -> list[tuple]:
        spare_nodes = [
            node_id
            for node_id, count in service_count_by_node.items()
            if count < self.config.max_services_per_satellite
        ]
        spare_planes = sorted({orbit_plane(node_id, self.config) for node_id in spare_nodes})
        candidates: list[tuple] = [("no-op", 0, None, None)]
        ranked_services = sorted(
            service_pressure,
            key=lambda service_id: self._service_pressure_score(service_pressure[service_id]),
            reverse=True,
        )
        for service_id in ranked_services[: self.config.num_microservices]:
            if service_id not in microservices:
                continue
            service = microservices[service_id]
            replica_planes = sorted(
                {orbit_plane(node_id, self.config) for node_id in service.replicas}
            )
            for target_plane in spare_planes:
                if (
                    len(service.replicas) < self.config.replica_count_range[1]
                    and self._plane_has_target(service, target_plane, service_count_by_node)
                ):
                    candidates.append(("add", service_id, None, target_plane))
            for source_plane in replica_planes:
                if len(service.replicas) > self.config.replica_count_range[0]:
                    candidates.append(("remove", service_id, source_plane, None))
                for target_plane in spare_planes:
                    if target_plane != source_plane and self._plane_has_target(
                        service, target_plane, service_count_by_node
                    ):
                        candidates.append(("move", service_id, source_plane, target_plane))
        return candidates

    def _rank_candidates(
        self,
        candidates: list[tuple],
        microservices: dict[int, Microservice],
        service_pressure: dict[int, ServicePressureSignal],
    ) -> list[tuple]:
        return sorted(
            candidates,
            key=lambda arm: (
                self._arm_failure_priority(arm, microservices),
                self._arm_service_pressure_priority(arm, service_pressure),
                self._ucb_score(arm),
            ),
            reverse=True,
        )

    def _service_pressure_score(self, signal: ServicePressureSignal) -> float:
        delay_scale = max(1.0e-9, self.config.service_pressure_delay_scale_s)
        return (
            signal.request_count
            + self.config.service_pressure_route_delay_weight
            * signal.average_route_delay_to_m
            / delay_scale
            + self.config.service_pressure_compute_wait_weight
            * signal.average_compute_waiting_time_of_m
            / delay_scale
            + self.config.service_pressure_route_failure_weight
            * signal.route_failure_count_related_to_m
            + self.config.service_pressure_p95_delay_weight
            * signal.p95_delay_of_m
            / delay_scale
            + self.config.service_pressure_replica_imbalance_weight
            * signal.replica_utilization_imbalance
        )

    def _arm_service_pressure_priority(
        self,
        arm: tuple,
        service_pressure: dict[int, ServicePressureSignal],
    ) -> float:
        action, service_id, _, _ = arm
        signal = service_pressure.get(service_id)
        if signal is None:
            return 0.0
        score = self._service_pressure_score(signal)
        if action == "move":
            return 0.75 * score
        if action == "remove":
            return max(0.0, 0.25 - 0.1 * score)
        if action == "add":
            return score
        return 0.0

    def _arm_failure_priority(
        self,
        arm: tuple,
        microservices: dict[int, Microservice],
    ) -> float:
        action, service_id, source_plane, _ = arm
        if action not in {"move", "remove"} or service_id not in microservices:
            return 0.0
        service = microservices[service_id]
        priority = 0.0
        for node_id in service.replicas:
            if source_plane is not None and orbit_plane(node_id, self.config) != source_plane:
                continue
            priority = max(priority, float(self.failed_replica_counts[(service_id, node_id)]))
        return priority

    def _ucb_score(self, arm: tuple) -> float:
        stats = self.arm_stats.get(arm)
        if stats is None or stats["count"] <= 0:
            return math.inf
        mean_reward = stats["reward_sum"] / stats["count"]
        exploration = self.exploration_c * math.sqrt(
            math.log(max(2, self.total_pulls + 1)) / stats["count"]
        )
        return mean_reward + exploration

    def _update_arm(self, arm: tuple, reward: float, reward_type: str = "estimated") -> None:
        stats = self.arm_stats.setdefault(
            arm,
            {
                "count": 0.0,
                "reward_sum": 0.0,
                "estimated_count": 0.0,
                "estimated_reward_sum": 0.0,
                "execution_count": 0.0,
                "execution_reward_sum": 0.0,
            },
        )
        stats["count"] += 1.0
        stats["reward_sum"] += reward
        if reward_type == "execution":
            stats["execution_count"] += 1.0
            stats["execution_reward_sum"] += reward
            self.total_execution_feedback_updates += 1
        else:
            stats["estimated_count"] += 1.0
            stats["estimated_reward_sum"] += reward
        self.total_pulls += 1

    def export_arm_stats(self) -> list[dict]:
        rows = []
        for arm, stats in sorted(self.arm_stats.items(), key=lambda item: str(item[0])):
            count = stats["count"]
            reward_sum = stats["reward_sum"]
            rows.append(
                {
                    "arm": json.dumps(arm, ensure_ascii=False),
                    "action": arm[0],
                    "service_id": arm[1],
                    "source_plane": arm[2],
                    "target_plane": arm[3],
                    "pull_count": count,
                    "reward_sum": reward_sum,
                    "mean_reward": reward_sum / count if count else 0.0,
                    "estimated_count": stats.get("estimated_count", 0.0),
                    "estimated_mean_reward": (
                        stats.get("estimated_reward_sum", 0.0)
                        / stats.get("estimated_count", 1.0)
                        if stats.get("estimated_count", 0.0)
                        else 0.0
                    ),
                    "execution_count": stats.get("execution_count", 0.0),
                    "execution_mean_reward": (
                        stats.get("execution_reward_sum", 0.0)
                        / stats.get("execution_count", 1.0)
                        if stats.get("execution_count", 0.0)
                        else 0.0
                    ),
                }
            )
        return rows

    def load_arm_stats(self, path: str | Path) -> int:
        path = Path(path)
        loaded_count = 0
        if not path.exists():
            return loaded_count

        self.arm_stats.clear()
        self.total_pulls = 0
        self.total_execution_feedback_updates = 0
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                arm = tuple(json.loads(row["arm"]))
                count = float(row.get("pull_count") or 0.0)
                reward_sum = float(row.get("reward_sum") or 0.0)
                estimated_count = float(row.get("estimated_count") or 0.0)
                estimated_mean = float(row.get("estimated_mean_reward") or 0.0)
                execution_count = float(row.get("execution_count") or 0.0)
                execution_mean = float(row.get("execution_mean_reward") or 0.0)
                self.arm_stats[arm] = {
                    "count": count,
                    "reward_sum": reward_sum,
                    "estimated_count": estimated_count,
                    "estimated_reward_sum": estimated_mean * estimated_count,
                    "execution_count": execution_count,
                    "execution_reward_sum": execution_mean * execution_count,
                }
                self.total_pulls += int(count)
                self.total_execution_feedback_updates += int(execution_count)
                loaded_count += 1
        return loaded_count

    def summary(self) -> dict:
        rows = self.export_arm_stats()
        positive = [row for row in rows if row["mean_reward"] > 0.0]
        return {
            "total_pulls": self.total_pulls,
            "known_arm_count": len(rows),
            "positive_arm_count": len(positive),
            "average_arm_reward": (
                sum(row["mean_reward"] for row in rows) / len(rows) if rows else 0.0
            ),
            "total_applied_actions": self.total_applied_actions,
            "total_execution_feedback_updates": self.total_execution_feedback_updates,
            "total_failed_replica_observations": self.total_failed_replica_observations,
            "known_failed_replica_count": len(self.failed_replica_counts),
            "known_service_pressure_metric_count": len(self.service_feedback_metrics),
            **{
                f"total_applied_{action}_count": count
                for action, count in self.applied_action_type_counts.items()
            },
            **{f"last_{key}": value for key, value in self.last_apply_summary.items()},
        }

    def _is_legal(
            self,
            action: str,
            source_node: int | None,
            target_node: int | None,
            service: Microservice,
            service_count_by_node: dict[int, int],
    ) -> bool:
        if action == "no-op":
            return False
        if action == "add":
            return (
                    target_node not in service.replicas
                    and len(service.replicas) < self.config.replica_count_range[1]
                    and service_count_by_node[target_node] < self.config.max_services_per_satellite
            )
        if action == "remove":
            return (
                    source_node in service.replicas
                    and len(service.replicas) > self.config.replica_count_range[0]
            )
        if action == "move":
            return (
                    source_node in service.replicas
                    and target_node not in service.replicas
                    and service_count_by_node[target_node] < self.config.max_services_per_satellite
            )
        return False

    def _resolve_arm(
        self,
        arm: tuple,
        service: Microservice,
        service_count_by_node: dict[int, int],
        context: dict,
        start_time: float,
    ) -> tuple[int | None, int | None] | None:
        action, _, source_plane, target_plane = arm
        if action == "add":
            target_node = self._select_target_node(
                service, target_plane, service_count_by_node, context, start_time
            )
            if target_node is None:
                return None
            return None, target_node
        if action == "remove":
            source_node = self._select_source_node(service, source_plane, context, start_time)
            if source_node is None:
                return None
            return source_node, None
        if action == "move":
            source_node = self._select_source_node(service, source_plane, context, start_time)
            target_node = self._select_target_node(
                service, target_plane, service_count_by_node, context, start_time
            )
            if source_node is None or target_node is None:
                return None
            return source_node, target_node
        return None

    def _plane_has_target(
        self,
        service: Microservice,
        target_plane: int | None,
        service_count_by_node: dict[int, int],
    ) -> bool:
        return any(
            (target_plane is None or orbit_plane(node_id, self.config) == target_plane)
            and node_id not in service.replicas
            and count < self.config.max_services_per_satellite
            for node_id, count in service_count_by_node.items()
        )

    def _select_source_node(
        self,
        service: Microservice,
        source_plane: int | None,
        context: dict,
        start_time: float,
    ) -> int | None:
        candidates = [
            node_id
            for node_id in service.replicas
            if source_plane is None or orbit_plane(node_id, self.config) == source_plane
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda node_id: (
                -self.failed_replica_counts[(service.service_id, node_id)],
                -self._low_speed_neighbor_score(node_id, context, start_time),
                node_id,
            )
        )
        return candidates[0]

    def _select_target_node(
        self,
        service: Microservice,
        target_plane: int | None,
        service_count_by_node: dict[int, int],
        context: dict,
        start_time: float,
    ) -> int | None:
        candidates = [
            node_id
            for node_id, count in service_count_by_node.items()
            if (target_plane is None or orbit_plane(node_id, self.config) == target_plane)
            and node_id not in service.replicas
            and count < self.config.max_services_per_satellite
        ]
        candidates.sort(
            key=lambda node_id: (
                self._low_speed_neighbor_score(node_id, context, start_time),
                service_count_by_node[node_id],
                node_id,
            )
        )
        return candidates[0] if candidates else None

    def _low_speed_neighbor_score(
        self,
        node_id: int | None,
        context: dict,
        start_time: float,
    ) -> float:
        if node_id is None:
            return 0.0
        slot_duration = context["slot_duration"]
        slot_count = context["slot_count"]
        _, slot_mod = slot_from_time(start_time, slot_duration, slot_count)
        graph = context["snapshots"][slot_mod]
        if node_id not in graph:
            return 0.0
        threshold = max(1.0e-9, self.config.low_speed_neighbor_rate_threshold_mbps)
        shortage_sum = 0.0
        degree = 0
        for neighbor in graph.neighbors(node_id):
            degree += 1
            rate_mbps = float(graph[node_id][neighbor].get("rate_mbps", 0.0))
            if rate_mbps < threshold:
                shortage_sum += 1.0 - max(0.0, rate_mbps) / threshold
        return shortage_sum / degree if degree else 0.0

    def _low_speed_placement_penalty(
        self,
        action: str,
        source_node: int | None,
        target_node: int | None,
        context: dict,
        start_time: float,
    ) -> float:
        source_score = self._low_speed_neighbor_score(source_node, context, start_time)
        target_score = self._low_speed_neighbor_score(target_node, context, start_time)
        if action == "add":
            raw_penalty = target_score
        elif action == "move":
            raw_penalty = target_score - source_score
        elif action == "remove":
            raw_penalty = -source_score
        else:
            raw_penalty = 0.0
        return self.config.low_speed_neighbor_penalty_weight * raw_penalty

    def _failure_relieving_bonus(
        self,
        action: str,
        service_id: int,
        source_node: int | None,
    ) -> float:
        if action not in {"move", "remove"} or source_node is None:
            return 0.0
        count = self.failed_replica_counts[(service_id, source_node)]
        return self.config.migration_failure_relief_bonus * min(1.0, float(count))

    def observe_failed_replicas(self, results: list[dict]) -> None:
        for result in results:
            if result.get("feasible", True):
                continue
            execution_plan = result.get("execution_plan") or []
            if not execution_plan:
                continue
            failed_stage = result.get("failed_stage")
            if not isinstance(failed_stage, int) or failed_stage <= 0:
                continue
            blame_index = min(failed_stage, len(execution_plan)) - 1
            blame_step = execution_plan[blame_index]
            service_id = blame_step.get("service_id")
            node_id = blame_step.get("satellite_node")
            if service_id is None or node_id is None:
                continue
            self.failed_replica_counts[(int(service_id), int(node_id))] += 1
            self.total_failed_replica_observations += 1

    def observe_execution_feedback(
        self, actions: list[MigrationAction], results: list[dict]
    ) -> None:
        self.observe_service_pressure_feedback(results)
        if not actions or not results:
            return
        feasible_count = sum(1 for result in results if result.get("feasible"))
        success_rate = feasible_count / len(results)
        finite_delays = [
            float(result["total_delay_s"])
            for result in results
            if math.isfinite(float(result.get("total_delay_s", math.inf)))
        ]
        finite_energy = [
            float(result["total_energy_j"])
            for result in results
            if math.isfinite(float(result.get("total_energy_j", math.inf)))
        ]
        avg_delay = sum(finite_delays) / len(finite_delays) if finite_delays else self.config.failure_penalty
        avg_energy = sum(finite_energy) / len(finite_energy) if finite_energy else self.config.failure_penalty
        execution_quality = (
            success_rate
            - self.config.delay_weight * avg_delay / 100.0
            - self.config.energy_weight * avg_energy / 10_000.0
        )
        for action in actions:
            if action.arm_key is None:
                continue
            feedback_reward = execution_quality - 0.1 * action.migration_cost
            action.execution_feedback_reward = feedback_reward
            self._update_arm(action.arm_key, feedback_reward, "execution")

    def observe_service_pressure_feedback(self, results: list[dict]) -> None:
        if not results:
            return
        self.service_feedback_metrics = self._collect_service_feedback_metrics(results)

    def _collect_service_feedback_metrics(
        self, results: list[dict]
    ) -> dict[int, ServicePressureSignal]:
        raw_stats = defaultdict(
            lambda: {
                "route_delays": [],
                "compute_waits": [],
                "step_delays": [],
                "route_failures": 0.0,
                "replica_counts": Counter(),
            }
        )
        route_delay_by_stage: dict[tuple[int, int], float] = {}

        for result in results:
            request_info = result.get("request") or {}
            services = [int(service_id) for service_id in request_info.get("services", [])]
            for route_record in result.get("route_details") or []:
                stage = route_record.get("stage")
                if not isinstance(stage, int) or stage < 0 or stage >= len(services):
                    continue
                service_id = services[stage]
                route_delay = float(route_record.get("communication_delay_s", math.inf))
                if math.isfinite(route_delay):
                    raw_stats[service_id]["route_delays"].append(route_delay)
                    request_id = int(request_info.get("request_id", -1))
                    route_delay_by_stage[(request_id, stage)] = route_delay

            for step in result.get("execution_plan") or []:
                service_id = step.get("service_id")
                if service_id is None:
                    continue
                service_id = int(service_id)
                queue_delay = float(step.get("queue_delay_s", math.inf))
                compute_delay = float(step.get("compute_delay_s", math.inf))
                if math.isfinite(queue_delay):
                    raw_stats[service_id]["compute_waits"].append(queue_delay)
                if math.isfinite(queue_delay) and math.isfinite(compute_delay):
                    request_id = int(step.get("request_id", -1))
                    stage = int(step.get("stage", -1))
                    route_delay = route_delay_by_stage.get((request_id, stage), 0.0)
                    raw_stats[service_id]["step_delays"].append(
                        route_delay + queue_delay + compute_delay
                    )
                node_id = step.get("satellite_node")
                if node_id is not None:
                    raw_stats[service_id]["replica_counts"][int(node_id)] += 1

            if not result.get("feasible", True):
                failed_service_id = self._failed_service_id(result, services)
                if failed_service_id is not None:
                    raw_stats[failed_service_id]["route_failures"] += 1.0

        return {
            service_id: ServicePressureSignal(
                service_id=service_id,
                average_route_delay_to_m=self._mean(values["route_delays"]),
                average_compute_waiting_time_of_m=self._mean(values["compute_waits"]),
                route_failure_count_related_to_m=values["route_failures"],
                p95_delay_of_m=self._percentile(values["step_delays"], 0.95),
                replica_execution_counts=values["replica_counts"],
            )
            for service_id, values in raw_stats.items()
        }

    def _failed_service_id(
        self,
        result: dict,
        services: list[int],
    ) -> int | None:
        failed_route = result.get("failed_route") or {}
        if failed_route.get("type") == "candidate_routes":
            service_id = failed_route.get("service_id")
            return int(service_id) if service_id is not None else None
        failed_stage = result.get("failed_stage")
        if not isinstance(failed_stage, int) or not services:
            return None
        if 0 <= failed_stage < len(services):
            return services[failed_stage]
        if failed_stage == len(services):
            return services[-1]
        return None

    def _replica_utilization_imbalance(
        self,
        service: Microservice,
        replica_counts: Counter[int],
    ) -> float:
        if not service.replicas:
            return 0.0
        counts = [float(replica_counts.get(node_id, 0)) for node_id in service.replicas]
        total = sum(counts)
        if total <= 0.0:
            return 0.0
        mean_count = total / len(counts)
        return sum(abs(count - mean_count) for count in counts) / (2.0 * total)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
        return ordered[index]

    def _expected_saving(
        self,
        action: str,
        service_id: int,
        service_pressure: dict[int, ServicePressureSignal],
    ) -> float:
        signal = service_pressure.get(service_id, ServicePressureSignal(service_id=service_id))
        base = self._service_pressure_score(signal) * 0.1
        if action == "add":
            return base
        if action == "move":
            return base * 0.75
        if action == "remove":
            return max(0.0, 0.25 - base * 0.1)
        return 0.0

    def _migration_cost(
            self,
            action: str,
            service: Microservice,
            source_node: int | None,
            target_node: int | None,
            context: dict,
            start_time: float,
    ) -> tuple[float, dict]:
        if action == "remove":
            return 0.01, {"delay_s": 0.0, "communication_energy_j": 0.0, "route_mode": "remove_only"}
        if target_node is None:
            return math.inf, {"reachable": False, "failure_reason": "missing_migration_endpoint"}

        selected_source_node = source_node
        if action == "add":
            selected_source_node = self._select_add_source(
                service, target_node, context, start_time
            )
        if selected_source_node is None:
            return math.inf, {"reachable": False, "failure_reason": "missing_add_source"}

        route = route_data(
            selected_source_node,
            target_node,
            service.image_size_gb,
            start_time,
            context,
        )
        if not route["reachable"]:
            return math.inf, route
        route["selected_source_node"] = selected_source_node

        # The route layer decides the actual image-transfer path. Therefore,
        # cross-orbit migrations use the same orbit-aware dual-path routing
        # mechanism as ordinary microservice intermediate-data transfers.
        route_cost = (
                self.config.delay_weight * route["delay_s"]
                + self.config.energy_weight * route["communication_energy_j"] / 1000.0
                + self.config.slot_switch_penalty_weight * route["slot_crossings"]
                + self.config.route_failure_risk_weight
                * float(route.get("route_failure_risk", 0.0))
        )
        image_cost = self.config.migration_weight * service.image_size_gb
        startup_cost = 0.01 * service.startup_delay_s
        if action == "move":
            return route_cost + image_cost + startup_cost + 0.02, route
        if action == "add":
            return route_cost + image_cost + startup_cost, route
        return 0.0, {"delay_s": 0.0, "communication_energy_j": 0.0, "route_mode": "no-op"}

    def _select_add_source(
            self,
            service: Microservice,
            target_node: int,
            context: dict,
            start_time: float,
    ) -> int | None:
        best_source = None
        best_cost = math.inf
        for source_node in service.replicas:
            route = route_data(
                source_node,
                target_node,
                service.image_size_gb,
                start_time,
                context,
            )
            if not route["reachable"]:
                continue
            cost = (
                    self.config.delay_weight * route["delay_s"]
                    + self.config.energy_weight * route["communication_energy_j"] / 1000.0
                    + self.config.slot_switch_penalty_weight * route["slot_crossings"]
                    + self.config.route_failure_risk_weight
                    * float(route.get("route_failure_risk", 0.0))
            )
            if cost < best_cost:
                best_cost = cost
                best_source = source_node
        return best_source
