from __future__ import annotations

import math
from statistics import mean


def finite_values(values):
    return [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def percentile(values, pct: float):
    values = sorted(finite_values(values))
    if not values:
        return math.inf
    index = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[index]


def summarize_results(results: list[dict]) -> dict:
    feasible = [result for result in results if result["feasible"]]
    delays = [result["total_delay_s"] for result in feasible]
    energies = [result["total_energy_j"] for result in feasible]
    route_records = [
        route
        for result in results
        for route in result.get("route_details", [])
        if math.isfinite(route.get("communication_delay_s", math.inf))
    ]
    route_delays = [route["communication_delay_s"] for route in route_records]
    slot_crossings = [route["slot_crossings"] for route in route_records]
    route_modes = {}
    for route in route_records:
        mode = route.get("route_mode", "")
        route_modes[mode] = route_modes.get(mode, 0) + 1

    return {
        "request_count": len(results),
        "feasible_count": len(feasible),
        "success_rate": len(feasible) / len(results) if results else 0.0,
        "average_end_to_end_delay_s": mean(delays) if delays else math.inf,
        "p95_end_to_end_delay_s": percentile(delays, 95),
        "average_energy_j": mean(energies) if energies else math.inf,
        "p95_energy_j": percentile(energies, 95),
        "average_communication_delay_s": mean(route_delays) if route_delays else math.inf,
        "average_slot_crossings": mean(slot_crossings) if slot_crossings else 0.0,
        "route_mode_counts": route_modes,
    }

