from __future__ import annotations

import math
import random
import ast
import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..config import SimulationConfig
from ..network.topology import TemporalGraph
from .service import Microservice


@dataclass
class SFCRequest:
    request_id: int
    start_time: float
    start_slot: int
    source_node: int
    destination_node: int
    services: list[int]
    input_data_gb: float
    data_gb_between_services: list[float]
    output_data_gb: float
    service_workload_cycles: list[float] = field(default_factory=list)
    template_id: int | None = None


def sample_data_gb(rng: random.Random, config: SimulationConfig) -> float:
    std = math.sqrt(config.request_data_variance_gb)
    return max(config.request_data_min_gb, rng.gauss(config.request_data_mean_gb, std))


def nodes_within_hops(graph: TemporalGraph, start_node: int, max_hops: int) -> list[int]:
    if start_node not in graph:
        return [start_node]

    visited = {start_node}
    queue = deque([(start_node, 0)])
    result: list[int] = []
    while queue:
        node, depth = queue.popleft()
        result.append(int(node))
        if depth >= max_hops:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((int(neighbor), depth + 1))
    return result


def low_speed_neighbor_score(
    graph: TemporalGraph,
    node_id: int,
    config: SimulationConfig,
) -> float:
    if node_id not in graph:
        return 1.0
    threshold = max(1.0e-9, config.low_speed_neighbor_rate_threshold_mbps)
    shortage_sum = 0.0
    degree = 0
    for neighbor in graph.neighbors(node_id):
        degree += 1
        rate_mbps = float(graph[node_id][neighbor].get("rate_mbps", 0.0))
        if rate_mbps < threshold:
            shortage_sum += 1.0 - max(0.0, rate_mbps) / threshold
    return shortage_sum / degree if degree else 1.0


def route_reachable(
    source_node: int,
    target_node: int,
    data_gb: float,
    start_time: float,
    context: dict,
) -> bool:
    from ..network.routing import route_data

    route_context = dict(context)
    route_context["routing_cache"] = None
    route = route_data(source_node, target_node, data_gb, start_time, route_context)
    return bool(route.get("reachable", False))


def select_request_endpoint(
    rng: random.Random,
    context: dict,
    graph: TemporalGraph,
    anchor_replicas: list[int],
    data_gb: float,
    start_time: float,
    direction: str,
    excluded_node: int | None = None,
) -> int:
    config: SimulationConfig = context["config"]
    candidates: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()

    shuffled_replicas = list(anchor_replicas)
    rng.shuffle(shuffled_replicas)
    for anchor_node in shuffled_replicas:
        for endpoint_node in nodes_within_hops(
            graph, anchor_node, config.source_dest_near_hops
        ):
            if endpoint_node == excluded_node:
                continue
            key = (endpoint_node, anchor_node)
            if key in seen:
                continue
            seen.add(key)
            score = low_speed_neighbor_score(graph, endpoint_node, config)
            candidates.append((score, endpoint_node, anchor_node))

    if not candidates:
        fallback_pool = [node for node in anchor_replicas if node != excluded_node]
        return rng.choice(fallback_pool or anchor_replicas)

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    preferred_candidates = [
        item
        for item in candidates
        if item[0] <= config.request_endpoint_max_low_speed_score
    ]
    if not preferred_candidates:
        preferred_candidates = candidates

    route_check_limit = max(
        1,
        min(config.request_endpoint_route_check_limit, len(preferred_candidates)),
    )
    check_pool = preferred_candidates[:route_check_limit]
    rng.shuffle(check_pool)
    reachable_candidates: list[tuple[float, int, int]] = []
    for score, endpoint_node, anchor_node in check_pool:
        if direction == "source":
            reachable = route_reachable(endpoint_node, anchor_node, data_gb, start_time, context)
        elif direction == "destination":
            reachable = route_reachable(anchor_node, endpoint_node, data_gb, start_time, context)
        else:
            raise ValueError(f"Unknown endpoint direction: {direction}")
        if reachable:
            reachable_candidates.append((score, endpoint_node, anchor_node))

    if not reachable_candidates and preferred_candidates is not candidates:
        fallback_pool = candidates[: max(route_check_limit, config.request_endpoint_sample_top_k)]
        rng.shuffle(fallback_pool)
        for score, endpoint_node, anchor_node in fallback_pool:
            if direction == "source":
                reachable = route_reachable(endpoint_node, anchor_node, data_gb, start_time, context)
            else:
                reachable = route_reachable(anchor_node, endpoint_node, data_gb, start_time, context)
            if reachable:
                reachable_candidates.append((score, endpoint_node, anchor_node))

    if not reachable_candidates:
        reachable_candidates = preferred_candidates
    reachable_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    top_k = max(1, min(config.request_endpoint_sample_top_k, len(reachable_candidates)))
    return rng.choice(reachable_candidates[:top_k])[1]


def expand_chain_lengths(config: SimulationConfig) -> list[int]:
    lengths: list[int] = []
    for count, length in config.request_chain_plan:
        lengths.extend([length] * count)
    return lengths


def generate_sfc_requests(
    rng: random.Random, context: dict, shuffle_lengths: bool = True
) -> list[SFCRequest]:
    config: SimulationConfig = context["config"]
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    microservices: dict[int, Microservice] = context["microservices"]

    chain_lengths = expand_chain_lengths(config)
    if shuffle_lengths:
        rng.shuffle(chain_lengths)

    lambda_per_second = config.request_arrival_lambda_per_slot / slot_duration
    current_time = 0.0
    requests: list[SFCRequest] = []

    for request_id, chain_length in enumerate(chain_lengths, start=1):
        if request_id > 1:
            current_time += rng.expovariate(lambda_per_second)
        start_slot = int(current_time // slot_duration) % slot_count
        start_time = current_time

        services = rng.choices(range(1, config.num_microservices + 1), k=chain_length)
        input_data_gb = sample_data_gb(rng, config)
        data_gb_between_services = [
            sample_data_gb(rng, config) for _ in range(chain_length - 1)
        ]
        output_data_gb = sample_data_gb(rng, config)
        reference_graph = context["snapshots"][start_slot]
        source_node = select_request_endpoint(
            rng,
            context,
            reference_graph,
            microservices[services[0]].replicas,
            input_data_gb,
            start_time,
            "source",
        )
        destination_node = select_request_endpoint(
            rng,
            context,
            reference_graph,
            microservices[services[-1]].replicas,
            output_data_gb,
            start_time,
            "destination",
            excluded_node=source_node,
        )

        requests.append(
            SFCRequest(
                request_id=request_id,
                start_time=start_time,
                start_slot=start_slot,
                source_node=source_node,
                destination_node=destination_node,
                services=services,
                input_data_gb=input_data_gb,
                data_gb_between_services=data_gb_between_services,
                output_data_gb=output_data_gb,
                service_workload_cycles=[
                    microservices[service_id].workload_cycles for service_id in services
                ],
                template_id=request_id,
            )
        )

    return requests


def poisson_sample(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    threshold = math.exp(-lam)
    value = 1.0
    count = 0
    while value > threshold:
        count += 1
        value *= rng.random()
    return count - 1


def generate_request_templates(
    rng: random.Random, context: dict, shuffle_lengths: bool = False
) -> list[SFCRequest]:
    config: SimulationConfig = context["config"]
    reference_graph = context["snapshots"][0]
    microservices: dict[int, Microservice] = context["microservices"]
    chain_lengths = expand_chain_lengths(config)
    if shuffle_lengths:
        rng.shuffle(chain_lengths)

    templates = []
    for template_id, chain_length in enumerate(chain_lengths, start=1):
        services = rng.choices(range(1, config.num_microservices + 1), k=chain_length)
        input_data_gb = sample_data_gb(rng, config)
        data_gb_between_services = [
            sample_data_gb(rng, config) for _ in range(chain_length - 1)
        ]
        output_data_gb = sample_data_gb(rng, config)
        source_node = select_request_endpoint(
            rng,
            context,
            reference_graph,
            microservices[services[0]].replicas,
            input_data_gb,
            0.0,
            "source",
        )
        destination_node = select_request_endpoint(
            rng,
            context,
            reference_graph,
            microservices[services[-1]].replicas,
            output_data_gb,
            0.0,
            "destination",
            excluded_node=source_node,
        )
        templates.append(
            SFCRequest(
                request_id=template_id,
                start_time=0.0,
                start_slot=0,
                source_node=source_node,
                destination_node=destination_node,
                services=services,
                input_data_gb=input_data_gb,
                data_gb_between_services=data_gb_between_services,
                output_data_gb=output_data_gb,
                service_workload_cycles=[
                    microservices[service_id].workload_cycles for service_id in services
                ],
                template_id=template_id,
            )
        )
    return templates


def instantiate_request_from_template(
    rng: random.Random,
    context: dict,
    template: SFCRequest,
    request_id: int,
    absolute_slot: int,
) -> SFCRequest:
    slot_duration = context["slot_duration"]
    slot_count = context["slot_count"]
    slot_mod = absolute_slot % slot_count
    slot_start_time = absolute_slot * slot_duration
    start_time = slot_start_time + rng.uniform(0.0, slot_duration)
    config: SimulationConfig = context["config"]
    reference_graph = context["snapshots"][slot_mod]
    microservices: dict[int, Microservice] = context["microservices"]

    first_service = template.services[0]
    last_service = template.services[-1]
    source_node = select_request_endpoint(
        rng,
        context,
        reference_graph,
        microservices[first_service].replicas,
        template.input_data_gb,
        start_time,
        "source",
    )
    destination_node = select_request_endpoint(
        rng,
        context,
        reference_graph,
        microservices[last_service].replicas,
        template.output_data_gb,
        start_time,
        "destination",
        excluded_node=source_node,
    )

    return SFCRequest(
        request_id=request_id,
        start_time=start_time,
        start_slot=slot_mod,
        source_node=source_node,
        destination_node=destination_node,
        services=list(template.services),
        input_data_gb=template.input_data_gb,
        data_gb_between_services=list(template.data_gb_between_services),
        output_data_gb=template.output_data_gb,
        service_workload_cycles=list(template.service_workload_cycles),
        template_id=template.request_id,
    )


def generate_slot_arrivals(
    rng: random.Random,
    context: dict,
    templates: list[SFCRequest],
    absolute_slot: int,
    next_request_id: int,
) -> tuple[list[SFCRequest], int, dict]:

    config: SimulationConfig = context["config"]
    arrivals: list[SFCRequest] = []
    counts_by_template: dict[int, int] = {}

    for template in templates:
        arrival_count = poisson_sample(
            rng, config.request_arrival_lambda_per_pattern_per_slot
        )
        counts_by_template[template.request_id] = arrival_count
        for _ in range(arrival_count):
            arrivals.append(
                instantiate_request_from_template(
                    rng, context, template, next_request_id, absolute_slot
                )
            )
            next_request_id += 1
    arrivals.sort(key=lambda request: request.start_time)

    return arrivals, next_request_id, {
        "arrival_count": len(arrivals),
        "arrival_lambda_per_pattern_per_slot": config.request_arrival_lambda_per_pattern_per_slot,
        "arrival_counts_by_template": counts_by_template,
    }


def generate_slot_arrivals_total_poisson(
    rng: random.Random,
    context: dict,
    templates: list[SFCRequest],
    absolute_slot: int,
    next_request_id: int,
    arrival_lambda_total_per_slot: float,
) -> tuple[list[SFCRequest], int, dict]:
    if not templates:
        raise ValueError("At least one request template is required.")

    arrivals: list[SFCRequest] = []
    counts_by_template = {template.request_id: 0 for template in templates}
    arrival_count = poisson_sample(rng, arrival_lambda_total_per_slot)
    for _ in range(arrival_count):
        template = rng.choice(templates)
        counts_by_template[template.request_id] += 1
        arrivals.append(
            instantiate_request_from_template(
                rng, context, template, next_request_id, absolute_slot
            )
        )
        next_request_id += 1
    arrivals.sort(key=lambda request: request.start_time)

    return arrivals, next_request_id, {
        "arrival_count": len(arrivals),
        "arrival_lambda_total_per_slot": arrival_lambda_total_per_slot,
        "arrival_counts_by_template": counts_by_template,
    }


def parse_list_field(value: str) -> list:
    if value in ("", "None", None):
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list field, got {type(parsed).__name__}: {value}")
    return parsed


def load_request_templates(path: str | Path) -> list[SFCRequest]:
    path = Path(path)
    templates: list[SFCRequest] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            template_id = int(row.get("template_id") or row.get("request_template_id"))
            services = [int(service_id) for service_id in parse_list_field(row["services"])]
            templates.append(
                SFCRequest(
                    request_id=template_id,
                    start_time=0.0,
                    start_slot=0,
                    source_node=int(row["source_node"]),
                    destination_node=int(row["destination_node"]),
                    services=services,
                    input_data_gb=float(row["input_data_gb"]),
                    data_gb_between_services=[
                        float(value) for value in parse_list_field(row["data_gb_between_services"])
                    ],
                    output_data_gb=float(row["output_data_gb"]),
                    service_workload_cycles=[
                        float(value) for value in parse_list_field(
                            row.get("service_workload_cycles", "")
                        )
                    ],
                    template_id=template_id,
                )
            )
    templates.sort(key=lambda request: request.request_id)
    return templates
