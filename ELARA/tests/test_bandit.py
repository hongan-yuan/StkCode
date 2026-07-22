from __future__ import annotations

import unittest

import numpy as np

from ELARA.bandit import BanditReplicaAdapter, StageExecutionRecord
from ELARA.config import ELARAConfig
from ELARA.domain import Microservice, SatelliteResource


class BanditTests(unittest.TestCase):
    def test_relocation_preserves_replica_count_and_memory(self):
        config = ELARAConfig(
            num_planes=2,
            sats_per_plane=4,
            num_services=1,
            replicas_per_service=1,
            deployment_window_requests=1,
            adaptation_top_k_services=1,
        )
        adapter = BanditReplicaAdapter(config)
        # Prefer candidates with low network/load features after normalization.
        adapter.vector_b = np.asarray([0.0, -10.0, -10.0])
        services = {
            0: Microservice(0, 1.0e9, [0], memory_requirement_gb=2.0)
        }
        resources = {
            node: SatelliteResource(node, 100.0, 50.0, 1.0, 8.0)
            for node in range(8)
        }
        adapter.observe_stage(
            StageExecutionRecord(
                request_id=0,
                service_id=0,
                source_node=3,
                serving_node=0,
                data_gb=0.01,
                route_delay_s=1.0,
                route_energy_j=1.0,
                compute_queue_s=1.0,
                compute_delay_s=1.0,
                compute_energy_j=1.0,
                normalized_cost=1.0,
            )
        )
        actions = adapter.close_request(
            services,
            resources,
            current_time=0.0,
            route_cost=lambda source, target, data, time: 1.0 if target == 0 else 0.0,
            compute_load=lambda node, time: 1.0 if node == 0 else 0.0,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, "relocate")
        self.assertEqual(len(services[0].replicas), 1)
        self.assertNotIn(0, services[0].replicas)
        self.assertEqual(adapter.total_relocations, 1)

        restored = BanditReplicaAdapter(config)
        restored.load_state_dict(adapter.state_dict())
        self.assertTrue(np.allclose(restored.matrix_a, adapter.matrix_a))
        self.assertEqual(restored.total_relocations, 1)


if __name__ == "__main__":
    unittest.main()
