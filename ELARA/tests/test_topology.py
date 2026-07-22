from __future__ import annotations

import unittest

from ELARA.topology import TemporalTopology


class TopologyTests(unittest.TestCase):
    def test_sparse_ring_and_shortest_path(self):
        topology = TemporalTopology.synthetic_ring(node_count=10, slots=3)
        graph = topology.graph_at_slot(1)
        path = graph.shortest_path(0, 5)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 5)
        self.assertTrue(graph.connected())

    def test_slot_wraparound(self):
        topology = TemporalTopology.synthetic_ring(node_count=8, slots=2)
        self.assertIs(topology.graph_at_slot(0), topology.graph_at_slot(2))


if __name__ == "__main__":
    unittest.main()

