from __future__ import annotations

from .config import SimulationConfig


def compute_energy_j(power_w: float, duration_s: float) -> float:
    return max(0.0, float(power_w)) * max(0.0, float(duration_s))


def communication_energy_j(
    tx_power_w: float, transmission_delay_s: float, config: SimulationConfig
) -> float:
    return compute_energy_j(tx_power_w or config.default_tx_power_w, transmission_delay_s)

