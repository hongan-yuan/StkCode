from __future__ import annotations

import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ISLEdge:
    u: int
    v: int
    rate_mbps: float
    distance_km: float
    tx_power_w: float
    link_type: str = "unknown"

    @property
    def key(self) -> tuple[int, int]:
        return (min(self.u, self.v), max(self.u, self.v))


class SparseGraph:
    def __init__(self, nodes=()):
        self.nodes: set[int] = {int(node) for node in nodes}
        self.adj: dict[int, dict[int, ISLEdge]] = {int(node): {} for node in self.nodes}

    def add_edge(self, edge: ISLEdge) -> None:
        self.nodes.update((edge.u, edge.v))
        self.adj.setdefault(edge.u, {})[edge.v] = edge
        self.adj.setdefault(edge.v, {})[edge.u] = edge

    def neighbors(self, node: int):
        return self.adj.get(int(node), {}).keys()

    def edge(self, u: int, v: int) -> ISLEdge | None:
        return self.adj.get(int(u), {}).get(int(v))

    def edges(self):
        seen: set[tuple[int, int]] = set()
        for u, neighbors in self.adj.items():
            for v, edge in neighbors.items():
                if edge.key not in seen:
                    seen.add(edge.key)
                    yield edge

    def induced(self, selected_nodes: set[int]) -> "SparseGraph":
        result = SparseGraph(selected_nodes)
        for edge in self.edges():
            if edge.u in selected_nodes and edge.v in selected_nodes:
                result.add_edge(edge)
        return result

    def connected(self, required: set[int] | None = None) -> bool:
        required = set(self.nodes if required is None else required)
        if not required:
            return True
        root = next(iter(required))
        visited = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return required <= visited

    def shortest_paths(self, source: int, weight: str = "hop") -> tuple[dict[int, float], dict[int, int]]:
        distance = {int(source): 0.0}
        parent: dict[int, int] = {}
        heap = [(0.0, int(source))]
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > distance.get(node, math.inf):
                continue
            for neighbor, edge in self.adj.get(node, {}).items():
                edge_cost = 1.0 if weight == "hop" else max(1.0e-9, edge.distance_km)
                new_cost = cost + edge_cost
                if new_cost < distance.get(neighbor, math.inf):
                    distance[neighbor] = new_cost
                    parent[neighbor] = node
                    heapq.heappush(heap, (new_cost, neighbor))
        return distance, parent

    def shortest_path(self, source: int, target: int, weight: str = "hop") -> list[int] | None:
        if source == target:
            return [int(source)]
        distance, parent = self.shortest_paths(source, weight)
        if target not in distance:
            return None
        path = [int(target)]
        while path[-1] != source:
            path.append(parent[path[-1]])
        path.reverse()
        return path


def satellite_name_to_id(name: str, sats_per_plane: int) -> int:
    parts = str(name).strip().split("_")
    if len(parts) < 3:
        raise ValueError(f"invalid satellite name: {name}")
    plane, position = int(parts[-2]), int(parts[-1])
    return plane * sats_per_plane + position


class TemporalTopology:
    def __init__(self, snapshots: dict[int, SparseGraph], slot_duration_s: float):
        if not snapshots:
            raise ValueError("at least one topology snapshot is required")
        self.snapshots = dict(sorted(snapshots.items()))
        self.slot_duration_s = float(slot_duration_s)
        self.slot_count = len(self.snapshots)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        sats_per_plane: int,
        total_satellites: int,
        max_slots: int | None = None,
        slot_duration_s: float | None = None,
    ) -> "TemporalTopology":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        graphs_by_time: dict[float, SparseGraph] = {}
        ordered_times: list[float] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                time_s = float(row["Time_EpSec"])
                if time_s not in graphs_by_time:
                    if max_slots is not None and len(ordered_times) >= max_slots:
                        break
                    ordered_times.append(time_s)
                    graphs_by_time[time_s] = SparseGraph(range(total_satellites))
                if str(row["Status"]).lower() != "alive":
                    continue
                rate = float(row["Effective_DataRate_Mbps"])
                if rate <= 0.0:
                    continue
                u = satellite_name_to_id(row["Endpoint_A"], sats_per_plane)
                v = satellite_name_to_id(row["Endpoint_B"], sats_per_plane)
                power = float(row.get("Link_Tx_Power_W") or 1.0)
                graphs_by_time[time_s].add_edge(
                    ISLEdge(u, v, rate, float(row["Distance_km"]), power, row["Link_Type"])
                )
        if slot_duration_s is None:
            slot_duration_s = ordered_times[1] - ordered_times[0] if len(ordered_times) > 1 else 10.0
        snapshots = {index: graphs_by_time[time_s] for index, time_s in enumerate(ordered_times)}
        return cls(snapshots, slot_duration_s)

    @classmethod
    def synthetic_ring(cls, node_count: int = 12, slots: int = 4, slot_duration_s: float = 10.0):
        snapshots: dict[int, SparseGraph] = {}
        for slot in range(slots):
            graph = SparseGraph(range(node_count))
            for node in range(node_count):
                graph.add_edge(ISLEdge(node, (node + 1) % node_count, 1000.0, 1000.0, 1.0, "ring"))
            if node_count >= 6 and slot % 2 == 0:
                graph.add_edge(ISLEdge(0, node_count // 2, 800.0, 1500.0, 1.0, "cross"))
            snapshots[slot] = graph
        return cls(snapshots, slot_duration_s)

    def absolute_slot(self, time_s: float) -> int:
        return int(math.floor(max(0.0, time_s) / self.slot_duration_s))

    def graph_at_slot(self, absolute_slot: int) -> SparseGraph:
        return self.snapshots[int(absolute_slot) % self.slot_count]

    def graph_at_time(self, time_s: float) -> SparseGraph:
        return self.graph_at_slot(self.absolute_slot(time_s))

