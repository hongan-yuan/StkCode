from __future__ import annotations

import unittest

from ELARA.connector import ConnectorBuilder
from ELARA.topology import TemporalTopology


class ConnectorTests(unittest.TestCase):
    def test_connector_contains_and_connects_all_terminals(self):
        graph = TemporalTopology.synthetic_ring(node_count=12, slots=2).graph_at_slot(1)
        terminals = {0, 3, 6, 9}
        connector = ConnectorBuilder("hop").build(graph, terminals, root=0, slot=0)
        self.assertTrue(terminals <= connector.nodes)
        self.assertEqual(connector.relay_nodes, connector.nodes - connector.terminals)
        self.assertTrue(connector.graph(graph).connected(terminals))

    def test_relay_is_added_for_nonadjacent_terminals(self):
        graph = TemporalTopology.synthetic_ring(node_count=8, slots=1).graph_at_slot(0)
        connector = ConnectorBuilder("hop").build(graph, {1, 3}, root=1, slot=0)
        self.assertTrue(connector.relay_nodes)


if __name__ == "__main__":
    unittest.main()

