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
            service_cycles_min=1.0e8,
            service_cycles_max=1.0e8,
            input_data_gb_min=0.001,
            input_data_gb_max=0.001,
            request_data_mean_gb=0.001,
            request_data_variance_gb=0.0,
        )
        return ELARAEnvironment(config, TemporalTopology.synthetic_ring(12, 4))

    def test_observation_is_sparse_and_candidates_have_three_features(self):
        state = self.make_environment().reset()
        state.validate()
        self.assertEqual(state.current_edge_index.shape[0], 2)
        self.assertEqual(state.candidate_features.shape[1], len(CANDIDATE_FEATURES))
        self.assertEqual(len(state.future_topologies), 2)

    def test_stage_trace_contains_replay_and_candidate_fields(self):
        environment = self.make_environment()
        state = environment.reset()
        environment.step(0)
        trace = environment.replica_adapter.window_records[-1]
        self.assertEqual(trace.template_id, 1)
        self.assertGreaterEqual(trace.topology_slot, 0)
        self.assertEqual(len(trace.candidate_nodes), len(state.candidate_indices))
        self.assertEqual(len(trace.candidate_hop_distances), len(trace.candidate_nodes))
        self.assertGreaterEqual(trace.stage_finish_time_s, trace.stage_start_time_s)

    def test_ppo_pretraining_does_not_collect_replica_adaptation_traces(self):
        environment = self.make_environment()
        environment.config.adaptation_enabled = False
        environment.reset()
        environment.step(0)
        self.assertEqual(environment.replica_adapter.window_records, [])

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
