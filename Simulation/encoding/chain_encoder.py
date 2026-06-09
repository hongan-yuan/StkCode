from __future__ import annotations

from ..config import SimulationConfig
from ..domain.request import SFCRequest
from ..domain.service import Microservice


def encode_service_chain(
    request: SFCRequest,
    current_index: int,
    microservices: dict[int, Microservice],
    config: SimulationConfig,
) -> list[float]:
    current_service = microservices[request.services[current_index]]
    remaining = len(request.services) - current_index
    total_data = request.input_data_gb + request.output_data_gb + sum(
        request.data_gb_between_services
    )
    return [
        len(request.services) / 20.0,
        current_index / max(1, len(request.services) - 1),
        remaining / max(1, len(request.services)),
        current_service.workload_cycles / config.microservice_workload_range_cycles[1],
        current_service.image_size_gb / config.microservice_image_size_range_gb[1],
        total_data / max(1.0, len(request.services) * config.request_data_mean_gb),
    ]
