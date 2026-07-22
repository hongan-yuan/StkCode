from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Microservice:
    service_id: int
    workload_cycles: float
    replicas: list[int]
    memory_requirement_gb: float
    activation_delay_s: float = 0.0


@dataclass(frozen=True)
class SatelliteResource:
    node_id: int
    capacity_gflops: float
    compute_power_w: float
    efficiency: float
    memory_capacity_gb: float


@dataclass(frozen=True)
class ServiceRequest:
    request_id: int
    source: int
    destination: int
    services: tuple[int, ...]
    data_volumes_gb: tuple[float, ...]
    arrival_time_s: float = 0.0

    def __post_init__(self) -> None:
        if len(self.data_volumes_gb) != len(self.services) + 1:
            raise ValueError("data_volumes_gb must have chain_length + 1 entries")
