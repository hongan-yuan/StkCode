from __future__ import annotations

import unittest

import numpy as np

from ELARA.bandit import BanditReplicaAdapter, StageExecutionRecord
from ELARA.config import ELARAConfig
from ELARA.domain import Microservice, SatelliteResource


def record(node: int, cost: float) -> StageExecutionRecord:
    return StageExecutionRecord(
        request_id=node,
        service_id=0,
        source_node=3,
        serving_node=node,
        data_gb=0.01,
        route_delay_s=cost,
        route_energy_j=1.0,
        compute_queue_s=cost,
        compute_delay_s=1.0,
        compute_energy_j=1.0,
        normalized_cost=cost,
        stage_start_time_s=1.0,
    )


class BanditTests(unittest.TestCase):
    def make_config(self):
        return ELARAConfig(
            num_planes=2,
            sats_per_plane=4,
            num_services=1,
            replica_count_range=(1, 3),
            deployment_window_requests=1,
            adaptation_top_k_services=1,
            bandit_exploration=0.0,
        )

    @staticmethod
    def resources():
        return {
            node: SatelliteResource(node, 4.0, 50.0, 1.0, 12.0)
            for node in range(8)
        }

    def choose(self, wanted: str, replicas: list[int], records):
        config = self.make_config()
        adapter = BanditReplicaAdapter(config)
        adapter.vector_b[4 + adapter.ACTIONS.index(wanted)] = 100.0
        services = {0: Microservice(0, 1.0e9, list(replicas), 2.0, 0.2)}
        for item in records:
            adapter.observe_stage(item)
        actions = adapter.close_request(
            services,
            self.resources(),
            current_time=1.0,
            route_cost=lambda source, target, data, time: 0.1 + target * 0.01,
            compute_load=lambda node, time: node * 0.01,
        )
        return adapter, services, actions

    def test_relocate_uses_highest_cumulative_cost_replica_and_preserves_count(self):
        adapter, services, actions = self.choose(
            "relocate", [0, 1], [record(0, 10.0), record(1, 1.0)]
        )
        self.assertEqual(actions[0].action, "relocate")
        self.assertEqual(actions[0].source_node, 0)
        self.assertEqual(len(services[0].replicas), 2)
        self.assertNotIn(0, services[0].replicas)

        restored = BanditReplicaAdapter(self.make_config())
        restored.load_state_dict(adapter.state_dict())
        self.assertTrue(np.allclose(restored.matrix_a, adapter.matrix_a))

    def test_scale_out_adds_exactly_one_replica(self):
        _, services, actions = self.choose("scale_out", [0], [record(0, 2.0)])
        self.assertEqual(actions[0].action, "scale_out")
        self.assertEqual(len(services[0].replicas), 2)
        self.assertEqual(actions[0].replica_count_after, 2)

    def test_scale_in_removes_lowest_contribution_replica(self):
        _, services, actions = self.choose("scale_in", [0, 1], [record(0, 2.0)])
        self.assertEqual(actions[0].action, "scale_in")
        self.assertEqual(actions[0].source_node, 1)
        self.assertEqual(services[0].replicas, [0])

    def test_no_op_leaves_replica_set_unchanged(self):
        _, services, actions = self.choose("no_op", [0, 1], [record(0, 2.0)])
        self.assertEqual(actions[0].action, "no_op")
        self.assertEqual(services[0].replicas, [0, 1])

    def test_feedback_ignores_replica_count_change(self):
        adapter, services, actions = self.choose(
            "scale_out", [0], [record(0, 2.0)]
        )
        applied = actions[0]
        self.assertEqual(applied.baseline_service_cost, 2.0)
        adapter.observe_stage(record(services[0].replicas[0], 2.0))
        adapter.close_request(
            services,
            self.resources(),
            current_time=2.0,
            route_cost=lambda source, target, data, time: 0.1 + target * 0.01,
            compute_load=lambda node, time: node * 0.01,
        )
        self.assertAlmostEqual(applied.feedback_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
