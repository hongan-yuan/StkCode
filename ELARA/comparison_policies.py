from __future__ import annotations

from collections import defaultdict

import numpy as np

from .state import NODE_FEATURES, ServingSelectionState


BASELINES = (
    "ELARA",
    "ELARA-NB",
    "ELARA-NR",
    "ELARA-SH",
    "SECO",
    "SP-Routing",
    "SC-NFV",
)
PPO_BASELINES = frozenset(("ELARA", "ELARA-NB", "ELARA-SH"))
ADAPTATION_BASELINES = frozenset(
    ("ELARA", "ELARA-NR", "ELARA-SH", "SP-Routing", "SC-NFV")
)
ROUTE_STRATEGIES = {
    "ELARA": "min_cost_flow",
    "ELARA-NB": "min_cost_flow",
    "ELARA-NR": "min_cost_flow",
    "ELARA-SH": "shortest_hop",
    "SECO": "shortest_hop",
    "SP-Routing": "service_pressure",
    "SC-NFV": "min_cost_flow",
}
BASELINE_DESCRIPTIONS = {
    "ELARA": "trained PPO selection, min-cost flow routing, replica adaptation",
    "ELARA-NB": "trained PPO selection, min-cost flow routing, no replica adaptation",
    "ELARA-NR": "nearest-replica selection, min-cost flow routing, replica adaptation",
    "ELARA-SH": "trained PPO selection, shortest-hop routing, replica adaptation",
    "SECO": "queue-aware greedy selection, shortest-hop routing, no replica adaptation",
    "SP-Routing": "virtual-backlog selection and service-pressure routing, replica adaptation",
    "SC-NFV": "orbital-plane-aware chaining, min-cost flow routing, replica adaptation",
}


class BaselineServingPolicy:
    """Serving-node policies evaluated inside the common ELARA environment.

    The three ELARA variants use the trained PPO network where appropriate.
    The remaining policies are lightweight adaptations of their published
    decision principles to the candidate features exposed by ELARA.
    """

    def __init__(self, baseline: str, config, ppo_agent=None):
        if baseline not in BASELINES:
            raise ValueError(f"unknown comparison baseline: {baseline}")
        if baseline in PPO_BASELINES and ppo_agent is None:
            raise ValueError(f"{baseline} requires a trained PPO agent")
        self.baseline = baseline
        self.config = config
        self.ppo_agent = ppo_agent
        self.virtual_backlog: defaultdict[tuple[int, int], float] = defaultdict(
            float
        )

    @staticmethod
    def _stage_index(state: ServingSelectionState) -> int:
        return int(np.count_nonzero(state.serving_history >= 0))

    @staticmethod
    def _argmin_valid(
        state: ServingSelectionState, scores: np.ndarray
    ) -> int:
        masked = np.where(state.action_mask, scores, np.inf)
        return int(np.argmin(masked))

    def act_batch(
        self, states: list[ServingSelectionState]
    ) -> list[int]:
        if self.baseline in PPO_BASELINES:
            return [
                action
                for action, _, _ in self.ppo_agent.act_batch(
                    states, deterministic=True
                )
            ]
        return [self._heuristic_action(state) for state in states]

    def _heuristic_action(self, state: ServingSelectionState) -> int:
        features = state.candidate_features
        hops = features[:, 0]
        rates = features[:, 1]
        queues = features[:, 2]

        if self.baseline == "ELARA-NR":
            scores = hops + 1.0e-3 * queues - 1.0e-4 * rates
            return self._argmin_valid(state, scores)

        if self.baseline == "SECO":
            # Queue-aware processing with a joint communication and computing
            # estimate. Lower hop count, queue, and bottleneck shortage win.
            scores = hops + 1.5 * queues + 0.75 * (1.0 - rates)
            return self._argmin_valid(state, scores)

        if self.baseline == "SP-Routing":
            stage = min(self._stage_index(state), len(state.service_ids) - 1)
            service_id = int(state.service_ids[stage])
            nodes = state.node_ids[state.candidate_indices]
            backlog = np.asarray(
                [
                    self.virtual_backlog[(service_id, int(node))]
                    for node in nodes
                ],
                dtype=np.float32,
            )
            if backlog.size and float(backlog.max()) > 0.0:
                backlog = backlog / float(backlog.max())
            scores = (
                0.50 * hops
                + 1.25 * queues
                + 0.75 * backlog
                + 0.75 * (1.0 - rates)
            )
            action = self._argmin_valid(state, scores)
            data_index = min(stage, len(state.data_volumes) - 1)
            selected_node = int(nodes[action])
            self.virtual_backlog[(service_id, selected_node)] += max(
                0.01, float(state.data_volumes[data_index])
            )
            return action

        if self.baseline == "SC-NFV":
            nodes = state.node_ids[state.candidate_indices]
            current_feature = NODE_FEATURES.index("is_current")
            destination_feature = NODE_FEATURES.index("is_destination")
            current_local = int(
                np.argmax(state.node_features[:, current_feature])
            )
            destination_local = int(
                np.argmax(state.node_features[:, destination_feature])
            )
            current_plane = (
                int(state.node_ids[current_local]) // self.config.sats_per_plane
            )
            destination_plane = (
                int(state.node_ids[destination_local])
                // self.config.sats_per_plane
            )
            candidate_planes = nodes // self.config.sats_per_plane
            current_distance = self._plane_distance(
                candidate_planes, current_plane
            )
            destination_distance = self._plane_distance(
                candidate_planes, destination_plane
            )
            scores = (
                0.45 * hops
                + 0.25 * queues
                + 0.15 * current_distance
                + 0.15 * destination_distance
                + 0.25 * (1.0 - rates)
            )
            return self._argmin_valid(state, scores)

        raise RuntimeError(f"no serving policy implemented for {self.baseline}")

    def _plane_distance(
        self, candidate_planes: np.ndarray, reference_plane: int
    ) -> np.ndarray:
        difference = np.abs(candidate_planes - int(reference_plane))
        wrapped = np.minimum(
            difference, self.config.num_planes - difference
        )
        return wrapped / max(1, self.config.num_planes // 2)

    def finish_time_slot(self) -> None:
        if self.baseline != "SP-Routing":
            return
        stale = []
        for key, value in self.virtual_backlog.items():
            updated = value * 0.90
            if updated <= 1.0e-8:
                stale.append(key)
            else:
                self.virtual_backlog[key] = updated
        for key in stale:
            del self.virtual_backlog[key]
