from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .topology import ISLEdge, TemporalTopology


def poisson_sample(rng: random.Random, mean: float) -> int:
    if mean <= 0.0:
        return 0
    if mean > 50.0:
        return max(0, int(round(rng.gauss(mean, math.sqrt(mean)))))
    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _sample_distribution(
    rng: random.Random, distribution: dict[str, float], fallback: str
) -> str:
    total = sum(max(0.0, float(weight)) for weight in distribution.values())
    if total <= 0.0:
        return fallback
    threshold = rng.random() * total
    cumulative = 0.0
    for state, weight in distribution.items():
        cumulative += max(0.0, float(weight))
        if threshold <= cumulative:
            return state
    return fallback


@dataclass(frozen=True)
class ComputeBackground:
    state: str
    utilization: float
    discount: float
    queue_delay_s: float
    background_cycles: float


@dataclass(frozen=True)
class LinkBackground:
    state: str
    utilization: float
    efficiency: float
    queue_delay_s: float
    background_data_gb: float


class MarkovBackgroundProcess:
    """Lazily generate absolute-slot Markov compute and ISL background load."""

    def __init__(self, topology: TemporalTopology, resources: dict, config, seed: int):
        self.topology = topology
        self.resources = resources
        self.config = config
        self.rng = random.Random(seed)
        compute_fallback = config.compute_load_states[0]
        link_fallback = config.link_load_states[0]
        self.compute_states = {
            node: _sample_distribution(
                self.rng, config.compute_load_initial_distribution, compute_fallback
            )
            for node in resources
        }
        self.link_states: dict[tuple[int, int], str] = {}
        self.link_fallback = link_fallback
        self.compute_cache: dict[int, dict[int, ComputeBackground]] = {}
        self.link_cache: dict[int, dict[tuple[int, int], LinkBackground]] = {}
        self.generated_through = -1

    def _generate_compute(self, node: int, state: str) -> ComputeBackground:
        config = self.config
        resource = self.resources[node]
        count = poisson_sample(
            self.rng,
            config.compute_load_lambda_per_slot.get(state, 0.0)
            * config.background_load_scale,
        )
        cycles = sum(
            max(
                config.background_compute_cycles_min,
                self.rng.expovariate(1.0 / config.background_compute_cycles_mean),
            )
            for _ in range(count)
        )
        nominal = max(
            1.0e-9,
            resource.capacity_gflops * 1.0e9 * self.topology.slot_duration_s,
        )
        raw = min(cycles / nominal, config.background_compute_rho_max)
        low, high = config.compute_load_utilization_ranges[state]
        utilization = min(config.background_compute_rho_max, max(low, min(high, raw)))
        discount_low, discount_high = config.compute_load_discount_ranges[state]
        discount = self.rng.uniform(discount_low, discount_high)
        queue = (
            config.background_compute_queue_base_s
            * utilization
            / (1.0 - utilization + config.background_epsilon)
        )
        return ComputeBackground(state, utilization, discount, queue, cycles)

    def _generate_link(self, edge: ISLEdge, state: str) -> LinkBackground:
        config = self.config
        count = poisson_sample(
            self.rng,
            config.background_link_lambda_per_slot_by_state.get(state, 0.0)
            * config.background_load_scale,
        )
        data = sum(
            max(
                config.background_link_data_min_gb,
                self.rng.expovariate(1.0 / config.background_link_data_mean_gb),
            )
            for _ in range(count)
        )
        capacity = (
            edge.rate_mbps
            * config.link_capacity_scale
            * self.topology.slot_duration_s
            / 8_000.0
        )
        utilization = (
            min(data / capacity, config.background_link_rho_max)
            if capacity > 0.0
            else config.background_link_rho_max
        )
        efficiency = max(
            config.background_link_eta_min,
            math.exp(-config.background_link_kappa * utilization),
        )
        queue = (
            config.background_link_queue_base_s
            * utilization
            / (1.0 - utilization + config.background_epsilon)
        )
        return LinkBackground(state, utilization, efficiency, queue, data)

    def _ensure(self, absolute_slot: int) -> None:
        absolute_slot = max(0, int(absolute_slot))
        for slot in range(self.generated_through + 1, absolute_slot + 1):
            compute_row = {}
            for node, state in self.compute_states.items():
                compute_row[node] = self._generate_compute(node, state)
                transition = self.config.compute_load_transition_matrix.get(state, {})
                self.compute_states[node] = _sample_distribution(
                    self.rng, transition, state
                )
            self.compute_cache[slot] = compute_row

            graph = self.topology.graph_at_slot(slot)
            current_edges = {edge.key: edge for edge in graph.edges()}
            for key in current_edges:
                if key not in self.link_states:
                    self.link_states[key] = _sample_distribution(
                        self.rng,
                        self.config.link_load_initial_distribution,
                        self.link_fallback,
                    )
            link_row = {}
            for key, state in list(self.link_states.items()):
                edge = current_edges.get(key)
                if edge is not None:
                    link_row[key] = self._generate_link(edge, state)
                transition = self.config.link_load_transition_matrix.get(state, {})
                self.link_states[key] = _sample_distribution(
                    self.rng, transition, state
                )
            self.link_cache[slot] = link_row
            self.generated_through = slot

    def compute(self, node: int, absolute_slot: int) -> ComputeBackground:
        self._ensure(absolute_slot)
        return self.compute_cache[int(absolute_slot)][int(node)]

    def link(self, edge: ISLEdge, absolute_slot: int) -> LinkBackground:
        self._ensure(absolute_slot)
        value = self.link_cache[int(absolute_slot)].get(edge.key)
        if value is None:
            return LinkBackground("Unavailable", 1.0, 0.0, math.inf, 0.0)
        return value
