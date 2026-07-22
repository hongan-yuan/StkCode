from __future__ import annotations

from dataclasses import dataclass

from .topology import SparseGraph


@dataclass(frozen=True)
class RequestSubgraph:
    terminals: frozenset[int]
    nodes: frozenset[int]
    relay_nodes: frozenset[int]
    connector_edges: frozenset[tuple[int, int]]
    root: int
    built_slot: int

    def graph(self, full_graph: SparseGraph) -> SparseGraph:
        return full_graph.induced(set(self.nodes))


class ConnectorBuilder:
    """Build a low-overhead rooted shortest-path connector tree."""

    def __init__(self, weight: str = "hop"):
        self.weight = weight

    def build(
        self,
        graph: SparseGraph,
        terminals: set[int],
        root: int,
        slot: int,
    ) -> RequestSubgraph:
        terminals = {int(node) for node in terminals}
        terminals.add(int(root))
        distance, parent = graph.shortest_paths(root, self.weight)
        unreachable = terminals - distance.keys()
        if unreachable:
            raise ValueError(f"terminals are disconnected in the full topology: {sorted(unreachable)}")

        nodes = {int(root)}
        edges: set[tuple[int, int]] = set()
        for terminal in sorted(terminals):
            node = terminal
            while node != root and node not in nodes:
                nodes.add(node)
                previous = parent[node]
                edges.add((min(node, previous), max(node, previous)))
                node = previous
            nodes.add(node)

        # The loop above may stop at an existing connector node, which is enough
        # because that node already has a path to the root.
        relays = nodes - terminals
        return RequestSubgraph(
            terminals=frozenset(terminals),
            nodes=frozenset(nodes),
            relay_nodes=frozenset(relays),
            connector_edges=frozenset(edges),
            root=int(root),
            built_slot=int(slot),
        )

    def needs_repair(self, subgraph: RequestSubgraph, graph: SparseGraph) -> bool:
        induced = graph.induced(set(subgraph.nodes))
        return not induced.connected(set(subgraph.terminals))

    def repair(self, subgraph: RequestSubgraph, graph: SparseGraph, slot: int) -> RequestSubgraph:
        if not self.needs_repair(subgraph, graph):
            return subgraph
        return self.build(graph, set(subgraph.terminals), subgraph.root, slot)
