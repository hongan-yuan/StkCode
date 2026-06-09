from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..config import SimulationConfig
from ..network.topology import slot_from_time
from .energy import compute_energy_j


@dataclass
class SatelliteResource:
    node_id: int
    base_freq_ghz: float
    base_freq_hz: float
    power_w: float
    storage_capacity_gb: float


@dataclass
class Microservice:
    service_id: int
    workload_cycles: float
    image_size_gb: float
    storage_requirement_gb: float
    startup_delay_s: float
    replicas: list[int]


def generate_satellite_resources(
    rng: random.Random, config: SimulationConfig
) -> dict[int, SatelliteResource]:
    resources: dict[int, SatelliteResource] = {}
    for node_id in range(1, config.total_sats + 1):
        freq_ghz = rng.choice(config.cpu_freq_choices_ghz)
        resources[node_id] = SatelliteResource(
            node_id=node_id,
            base_freq_ghz=freq_ghz,
            base_freq_hz=freq_ghz * 1.0e9,
            power_w=config.cpu_power_by_freq_w[freq_ghz],
            storage_capacity_gb=config.satellite_storage_capacity_gb,
        )
    return resources


def generate_microservice_catalog(
    rng: random.Random, config: SimulationConfig
) -> dict[int, Microservice]:
    catalog: dict[int, Microservice] = {}
    service_count_by_sat = {node_id: 0 for node_id in range(1, config.total_sats + 1)}
    min_workload, max_workload = config.microservice_workload_range_cycles

    for service_id in range(1, config.num_microservices + 1):
        replica_count = rng.randint(*config.replica_count_range)
        feasible_nodes = [
            node_id
            for node_id, count in service_count_by_sat.items()
            if count < config.max_services_per_satellite
        ]
        if len(feasible_nodes) < replica_count:
            raise RuntimeError("Not enough satellite capacity for microservice replicas.")

        feasible_nodes.sort(key=lambda node_id: (service_count_by_sat[node_id], rng.random()))
        replicas = sorted(feasible_nodes[:replica_count])
        for node_id in replicas:
            service_count_by_sat[node_id] += 1

        catalog[service_id] = Microservice(
            service_id=service_id,
            workload_cycles=rng.uniform(min_workload, max_workload),
            image_size_gb=rng.uniform(*config.microservice_image_size_range_gb),
            storage_requirement_gb=rng.uniform(*config.microservice_storage_range_gb),
            startup_delay_s=rng.uniform(0.2, 2.0),
            replicas=replicas,
        )
    return catalog


def deployment_matrix(
    microservices: dict[int, Microservice]
) -> dict[int, set[int]]:
    by_node: dict[int, set[int]] = {}
    for service in microservices.values():
        for node_id in service.replicas:
            by_node.setdefault(node_id, set()).add(service.service_id)
    return by_node


def compute_service_execution(
    service_id: int, node_id: int, arrival_time: float, context: dict
) -> dict:
    config: SimulationConfig = context["config"]
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    resources: dict[int, SatelliteResource] = context["satellite_resources"]
    discount_table = context["discount_table"]
    queue_delay_table = context["queue_delay_table"]
    state_table = context.get("compute_load_state_table", {})
    utilization_table = context.get("compute_utilization_table", {})
    microservices: dict[int, Microservice] = context["microservices"]

    service = microservices[service_id]
    resource = resources[node_id]
    _, queue_slot_mod = slot_from_time(arrival_time, slot_duration, slot_count)
    queue_delay = queue_delay_table[queue_slot_mod][node_id]
    queue_load_state = state_table.get(queue_slot_mod, {}).get(node_id, "")
    queue_compute_utilization = utilization_table.get(queue_slot_mod, {}).get(node_id, 0.0)
    compute_start = arrival_time + queue_delay

    current_time = compute_start
    remaining_cycles = service.workload_cycles
    compute_segments = []
    weighted_discount_sum = 0.0
    compute_delay = 0.0

    while remaining_cycles > 1.0e-6:
        abs_slot, slot_mod = slot_from_time(current_time, slot_duration, slot_count)
        next_slot_time = (abs_slot + 1) * slot_duration
        available_time = max(1.0e-9, next_slot_time - current_time)
        discount = discount_table[slot_mod][node_id]
        compute_load_state = state_table.get(slot_mod, {}).get(node_id, "")
        compute_utilization = utilization_table.get(slot_mod, {}).get(node_id, 0.0)
        effective_freq_hz = resource.base_freq_hz * discount
        cycles_until_next_slot = effective_freq_hz * available_time

        if cycles_until_next_slot >= remaining_cycles:
            segment_duration = remaining_cycles / effective_freq_hz
            segment_cycles = remaining_cycles
            remaining_cycles = 0.0
        else:
            segment_duration = available_time
            segment_cycles = cycles_until_next_slot
            remaining_cycles -= cycles_until_next_slot

        compute_segments.append(
            {
                "slot_mod": slot_mod,
                "start_time_s": current_time,
                "duration_s": segment_duration,
                "discount": discount,
                "compute_load_state": compute_load_state,
                "compute_utilization": compute_utilization,
                "effective_freq_ghz": effective_freq_hz / 1.0e9,
                "cycles": segment_cycles,
            }
        )
        weighted_discount_sum += discount * segment_duration
        compute_delay += segment_duration
        current_time += segment_duration

        if not math.isfinite(current_time):
            break

    avg_discount = weighted_discount_sum / compute_delay if compute_delay > 0 else 1.0
    compute_energy = compute_energy_j(resource.power_w, compute_delay)

    return {
        "arrival_time_s": arrival_time,
        "queue_slot_mod": queue_slot_mod,
        "queue_compute_load_state": queue_load_state,
        "queue_compute_utilization": queue_compute_utilization,
        "queue_delay_s": queue_delay,
        "compute_start_s": compute_start,
        "compute_finish_s": current_time,
        "compute_delay_s": compute_delay,
        "avg_cpu_discount": avg_discount,
        "base_freq_ghz": resource.base_freq_ghz,
        "avg_effective_freq_ghz": resource.base_freq_ghz * avg_discount,
        "compute_energy_j": compute_energy,
        "segments": compute_segments,
    }
