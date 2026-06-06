from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

from .config import SimulationConfig
from .constellation import sat_name_to_node_id

REQUIRED_ISL_COLUMNS = {
    "Time_EpSec",
    "Time_UTC",
    "Endpoint_A",
    "Endpoint_B",
    "Link_ID",
    "Link_Type",
    "Status",
    "Distance_km",
    "Effective_DataRate_Mbps",
}

_TEMPORAL_GRAPH_CACHE: dict[tuple, tuple[dict[int, "TemporalGraph"], list[float], float, dict[int, str]]] = {}


class DegreeView:
    def __init__(self, graph: "TemporalGraph"):
        self._graph = graph

    def __getitem__(self, node_id: int) -> int:
        return len(self._graph.adj.get(int(node_id), {}))


class TemporalGraph:
    def __init__(self):
        self.nodes: set[int] = set()
        self.adj: dict[int, dict[int, dict]] = {}
        self.graph: dict = {}
        self.degree = DegreeView(self)

    def add_nodes_from(self, nodes) -> None:
        for node in nodes:
            node = int(node)
            self.nodes.add(node)
            self.adj.setdefault(node, {})

    def add_edge(self, node_u: int, node_v: int, **data) -> None:
        node_u = int(node_u)
        node_v = int(node_v)
        self.add_nodes_from([node_u, node_v])
        self.adj[node_u][node_v] = dict(data)
        self.adj[node_v][node_u] = dict(data)

    def has_edge(self, node_u: int, node_v: int) -> bool:
        return int(node_v) in self.adj.get(int(node_u), {})

    def neighbors(self, node_id: int):
        return self.adj.get(int(node_id), {}).keys()

    def edges(self, data: bool = False):
        seen = set()
        for node_u, nbrs in self.adj.items():
            for node_v, edge_data in nbrs.items():
                edge = tuple(sorted((node_u, node_v)))
                if edge in seen:
                    continue
                seen.add(edge)
                yield (node_u, node_v, edge_data) if data else (node_u, node_v)

    def __contains__(self, node_id: int) -> bool:
        return int(node_id) in self.nodes

    def __getitem__(self, node_id: int) -> dict[int, dict]:
        return self.adj[int(node_id)]


def slot_from_time(
        absolute_time: float, slot_duration: float, slot_count: int
) -> tuple[int, int]:
    absolute_slot = int(math.floor(max(0.0, absolute_time) / slot_duration))
    return absolute_slot, absolute_slot % slot_count


def load_temporal_graphs(
        csv_path: str | Path, config: SimulationConfig
) -> tuple[dict[int, TemporalGraph], list[float], float, dict[int, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ISL log file not found: {csv_path}")
    cache_key = (
        str(csv_path.resolve()),
        csv_path.stat().st_mtime_ns,
        config.max_slots_to_load,
        config.slot_duration_override_s,
        config.total_sats,
        config.default_tx_power_w,
    )
    cached = _TEMPORAL_GRAPH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows_by_time: dict[float, list[dict]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("ISL log has no header.")
        missing = sorted(REQUIRED_ISL_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"ISL log is missing required columns: {missing}")
        for row in reader:
            try:
                time_epsec = float(row["Time_EpSec"])
            except (TypeError, ValueError):
                continue
            rows_by_time.setdefault(time_epsec, []).append(row)

    time_values = sorted(rows_by_time)
    if config.max_slots_to_load is not None:
        time_values = time_values[: config.max_slots_to_load]
    if not time_values:
        raise ValueError("ISL log does not contain any usable time slots.")

    slot_by_time = {time_epsec: slot for slot, time_epsec in enumerate(time_values)}
    if config.slot_duration_override_s is not None:
        slot_duration = float(config.slot_duration_override_s)
    elif len(time_values) >= 2:
        diffs = [round(b - a, 9) for a, b in zip(time_values[:-1], time_values[1:]) if b > a]
        slot_duration = Counter(diffs).most_common(1)[0][0] if diffs else 10.0
    else:
        slot_duration = 10.0

    snapshots: dict[int, TemporalGraph] = {}
    time_utc_by_slot: dict[int, str] = {}
    for time_epsec in time_values:
        group = rows_by_time[time_epsec]
        slot = slot_by_time[time_epsec]
        graph = TemporalGraph()
        graph.add_nodes_from(range(1, config.total_sats + 1))
        graph.graph["time_slot"] = slot
        graph.graph["time_epsec"] = float(time_epsec)
        graph.graph["time_utc"] = str(group[0]["Time_UTC"])
        time_utc_by_slot[slot] = graph.graph["time_utc"]

        for row in group:
            try:
                rate_mbps = float(row["Effective_DataRate_Mbps"])
                distance_km = float(row["Distance_km"])
            except (TypeError, ValueError):
                continue
            if str(row["Status"]).lower() != "alive" or rate_mbps <= 0.0:
                continue
            node_u = sat_name_to_node_id(str(row["Endpoint_A"]), config)
            node_v = sat_name_to_node_id(str(row["Endpoint_B"]), config)
            try:
                tx_power_w = float(row.get("Link_Tx_Power_W", config.default_tx_power_w))
            except (TypeError, ValueError):
                tx_power_w = config.default_tx_power_w
            graph.add_edge(
                node_u,
                node_v,
                rate_mbps=rate_mbps,
                distance_km=distance_km,
                link_type=str(row["Link_Type"]),
                link_id=str(row["Link_ID"]),
                tx_power_w=tx_power_w,
            )
        snapshots[slot] = graph

    result = (
        snapshots,
        list(map(float, time_values)),
        float(slot_duration),
        time_utc_by_slot,
    )
    _TEMPORAL_GRAPH_CACHE[cache_key] = result
    return result
