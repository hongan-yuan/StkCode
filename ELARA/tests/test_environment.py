from __future__ import annotations

import unittest

from ELARA.config import ELARAConfig
from ELARA.environment import ELARAEnvironment
from ELARA.state import CANDIDATE_FEATURES
from ELARA.topology import TemporalTopology


class EnvironmentTests(unittest.TestCase):
    def make_environment(self):
        config = ELARAConfig(
            seed=7,
            num_planes=1,
            sats_per_plane=12,
            num_services=8,
            replicas_per_service=3,
            chain_length=3,
            max_trace_slots=4,
            future_topology_horizon=2,
        )
        return ELARAEnvironment(config, TemporalTopology.synthetic_ring(12, 4))

    def test_observation_is_sparse_and_candidates_have_three_features(self):
        state = self.make_environment().reset()
        state.validate()
        self.assertEqual(state.current_edge_index.shape[0], 2)
        self.assertEqual(state.candidate_features.shape[1], len(CANDIDATE_FEATURES))
        self.assertEqual(len(state.future_topologies), 2)

    def test_full_random_episode(self):
        environment = self.make_environment()
        state = environment.reset()
        steps = 0
        info = {}
        while state is not None:
            state, reward, terminated, truncated, info = environment.step(0)
            self.assertLessEqual(reward, 0.0)
            steps += 1
        self.assertTrue(info["success"])
        self.assertEqual(steps, environment.config.chain_length)
        self.assertGreater(info["total_latency_s"], 0.0)
        self.assertGreaterEqual(info["relay_count"], 0)


if __name__ == "__main__":
    unittest.main()

