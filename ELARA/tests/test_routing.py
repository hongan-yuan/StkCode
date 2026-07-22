from __future__ import annotations

import unittest
from unittest.mock import patch

from ELARA.config import ELARAConfig
from ELARA.routing import CrossSlotMinCostRouter
from ELARA.topology import ISLEdge, SparseGraph, TemporalTopology


class RoutingTests(unittest.TestCase):
    def test_future_link_risk_is_disabled_by_default(self):
        topology = TemporalTopology.synthetic_ring(4, slots=2, slot_duration_s=5.0)
        config = ELARAConfig(num_planes=1, sats_per_plane=4)
        router = CrossSlotMinCostRouter(topology, config)
        edge = topology.graph_at_slot(0).edge(0, 1)

        with patch.object(
            router,
            "_future_edge_risk",
            side_effect=AssertionError("disabled risk must not be evaluated"),
        ):
            cost = router._edge_unit_cost(
                edge,
                absolute_slot=0,
                current_time=0.0,
                rate_provider=lambda item, slot: item.rate_mbps,
                queue_provider=lambda item, time_s: 0.0,
            )

        self.assertGreater(cost, 0.0)

    def test_residual_data_is_rerouted_across_slots(self):
        topology = TemporalTopology.synthetic_ring(4, slots=4, slot_duration_s=5.0)
        config = ELARAConfig(
            num_planes=1,
            sats_per_plane=4,
            route_horizon_slots=3,
            route_max_paths_per_slot=1,
            future_topology_horizon=1,
        )
        router = CrossSlotMinCostRouter(topology, config)
        reservations = []
        result = router.route(
            source=0,
            target=1,
            data_gb=0.001,
            start_time=0.0,
            graph_provider=topology.graph_at_time,
            rate_provider=lambda edge, slot: 1.0,
            queue_provider=lambda edge, time_s: 0.0,
            reserve=lambda edge, start, data, rate: reservations.append((edge.key, data)),
        )
        self.assertTrue(result["reachable"])
        self.assertGreaterEqual(len(result["slot_phases"]), 2)
        self.assertGreaterEqual(result["slot_crossings"], 1)
        self.assertAlmostEqual(result["remaining_data_gb"], 0.0)
        self.assertTrue(all(phase["paths"][0]["path"] == [0, 1] for phase in result["slot_phases"]))
        self.assertGreaterEqual(len(reservations), 2)

    def test_edge_disjoint_paths_complete_in_parallel(self):
        graph = SparseGraph(range(4))
        for u, v, rate in (
            (0, 1, 1.0),
            (1, 3, 1.0e15),
            (0, 2, 1.0),
            (2, 3, 1.0e15),
        ):
            graph.add_edge(
                ISLEdge(u, v, rate_mbps=rate, distance_km=0.0, tx_power_w=1.0)
            )
        topology = TemporalTopology({0: graph}, slot_duration_s=20.0)
        config = ELARAConfig(
            num_planes=1,
            sats_per_plane=4,
            route_max_paths_per_slot=2,
            route_switch_delay_s=0.0,
            future_topology_horizon=1,
        )
        router = CrossSlotMinCostRouter(topology, config)
        result = router._slot_flow(
            graph=graph,
            source=0,
            target=3,
            remaining_data_gb=0.004,
            current_time=0.0,
            slot_end=20.0,
            absolute_slot=0,
            rate_provider=lambda edge, slot: edge.rate_mbps,
            queue_provider=lambda edge, time_s: 0.0,
            reserve=None,
            commit=False,
        )

        self.assertEqual(len(result["paths"]), 2)
        path_delays = [path["delay_s"] for path in result["paths"]]
        self.assertAlmostEqual(result["phase_delay_s"], max(path_delays))
        self.assertLess(result["phase_delay_s"], sum(path_delays))


if __name__ == "__main__":
    unittest.main()
