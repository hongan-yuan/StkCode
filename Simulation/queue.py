from __future__ import annotations

import math
import random

from .config import SimulationConfig


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
