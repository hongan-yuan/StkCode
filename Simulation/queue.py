from __future__ import annotations

import math
import random

from .config import SimulationConfig


def _sample_from_distribution(
    rng: random.Random, distribution: dict[str, float], fallback: str
) -> str:
    total = sum(max(0.0, float(weight)) for weight in distribution.values())
    if total <= 0.0:
        return fallback
    threshold = rng.random() * total
    running = 0.0
    for key, weight in distribution.items():
        running += max(0.0, float(weight))
        if threshold <= running:
            return key
    return fallback


def _state_range(
    ranges: dict[str, tuple[float, float]], state: str, default: tuple[float, float]
) -> tuple[float, float]:
    low, high = ranges.get(state, default)
    return min(low, high), max(low, high)


def generate_markov_compute_load_tables(
    rng: random.Random,
    slot_count: int,
    slot_duration: float,
    satellite_resources: dict,
    config: SimulationConfig,
) -> dict[str, dict]:
    """Generate slot-correlated background compute load with a Markov state model.

    The returned derived tables keep the older execution interface intact:
    ``discount_table`` provides the per-slot effective CPU discount and
    ``queue_delay_table`` provides the pre-execution waiting time. Additional
    state/utilization tables are available for diagnostics and GNN features.
    """

    states = config.compute_load_states
    fallback_state = states[0] if states else "Idle"
    state_table: dict[int, dict[int, str]] = {}
    utilization_table: dict[int, dict[int, float]] = {}
    background_cycles_table: dict[int, dict[int, float]] = {}
    discount_table: dict[int, dict[int, float]] = {}
    queue_delay_table: dict[int, dict[int, float]] = {}

    current_state_by_node = {
        node_id: _sample_from_distribution(
            rng, config.compute_load_initial_distribution, fallback_state
        )
        for node_id in range(1, config.total_sats + 1)
    }

    for slot in range(slot_count):
        state_table[slot] = {}
        utilization_table[slot] = {}
        background_cycles_table[slot] = {}
        discount_table[slot] = {}
        queue_delay_table[slot] = {}

        for node_id in range(1, config.total_sats + 1):
            state = current_state_by_node[node_id]
            resource = satellite_resources[node_id]
            lambda_per_slot = config.compute_load_lambda_per_slot.get(state, 0.0)
            request_count = _poisson_sample(rng, lambda_per_slot)
            background_cycles = 0.0
            for _ in range(request_count):
                sample = rng.expovariate(
                    1.0 / max(config.background_compute_cycles_mean, 1.0e-9)
                )
                background_cycles += max(config.background_compute_cycles_min, sample)

            nominal_cycles = max(1.0e-9, resource.base_freq_hz * slot_duration)
            raw_rho = min(background_cycles / nominal_cycles, config.background_compute_rho_max)
            rho_low, rho_high = _state_range(
                config.compute_load_utilization_ranges, state, (0.0, config.background_compute_rho_max)
            )
            rho = min(
                config.background_compute_rho_max,
                max(rho_low, min(rho_high, raw_rho)),
            )
            discount_low, discount_high = _state_range(
                config.compute_load_discount_ranges, state, config.cpu_discount_range
            )
            discount = rng.uniform(discount_low, discount_high)
            queue_delay = (
                config.background_compute_queue_base_s
                * rho
                / (1.0 - rho + config.background_epsilon)
            )

            state_table[slot][node_id] = state
            utilization_table[slot][node_id] = rho
            background_cycles_table[slot][node_id] = background_cycles
            discount_table[slot][node_id] = discount
            queue_delay_table[slot][node_id] = queue_delay

            transition = config.compute_load_transition_matrix.get(state, {})
            current_state_by_node[node_id] = _sample_from_distribution(
                rng, transition, state
            )

    return {
        "compute_load_state_table": state_table,
        "compute_utilization_table": utilization_table,
        "compute_background_cycles_table": background_cycles_table,
        "discount_table": discount_table,
        "queue_delay_table": queue_delay_table,
    }


def generate_cpu_discount_table(
    rng: random.Random, slot_count: int, config: SimulationConfig
) -> dict[int, dict[int, float]]:
    low, high = config.cpu_discount_range
    return {
        slot: {
            node_id: rng.uniform(low, high)
            for node_id in range(1, config.total_sats + 1)
        }
        for slot in range(slot_count)
    }


def generate_queue_delay_table(
    rng: random.Random, slot_count: int, slot_duration: float, config: SimulationConfig
) -> dict[int, dict[int, float]]:
    max_delay = max(0.0, slot_duration * config.queue_max_fraction_of_slot)
    table: dict[int, dict[int, float]] = {}
    for slot in range(slot_count):
        table[slot] = {}
        for node_id in range(1, config.total_sats + 1):
            if rng.random() < config.queue_zero_probability:
                queue_delay = 0.0
            else:
                queue_delay = rng.uniform(0.0, max_delay)
            table[slot][node_id] = queue_delay
    return table


def _poisson_sample(rng: random.Random, mean: float) -> int:
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


def _edge_key(node_u: int, node_v: int) -> tuple[int, int]:
    return tuple(sorted((int(node_u), int(node_v))))


def generate_link_background_table(
    rng: random.Random,
    snapshots: dict,
    slot_duration: float,
    config: SimulationConfig,
) -> dict[int, dict[tuple[int, int], dict[str, float]]]:
    """Generate implicit background communication load for current-slot routing.

    The table is intentionally keyed by slot and edge only. PPO may inspect
    future deterministic ISL availability/rates from topology snapshots, but it
    should not use future entries from this stochastic background table.
    """

    table: dict[int, dict[tuple[int, int], dict[str, float]]] = {}
    for slot, graph in snapshots.items():
        table[slot] = {}
        for node_u, node_v, edge_data in graph.edges(data=True):
            rate_mbps = max(0.0, float(edge_data.get("rate_mbps", 0.0)))
            nominal_capacity_gb = rate_mbps * 1.0e6 * slot_duration / 1.0e9
            request_count = _poisson_sample(rng, config.background_link_lambda_per_slot)
            background_data_gb = 0.0
            for _ in range(request_count):
                sample = rng.expovariate(1.0 / max(config.background_link_data_mean_gb, 1.0e-9))
                background_data_gb += max(config.background_link_data_min_gb, sample)
            if nominal_capacity_gb > 0.0:
                rho = min(background_data_gb / nominal_capacity_gb, config.background_link_rho_max)
            else:
                rho = config.background_link_rho_max
            eta = max(
                config.background_link_eta_min,
                math.exp(-config.background_link_kappa * rho),
            )
            queue_delay_s = (
                config.background_link_queue_base_s
                * rho
                / (1.0 - rho + config.background_epsilon)
            )
            table[slot][_edge_key(node_u, node_v)] = {
                "background_data_gb": background_data_gb,
                "rho": rho,
                "eta": eta,
                "queue_delay_s": queue_delay_s,
                "effective_rate_mbps": rate_mbps * eta,
            }
    return table
