from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

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

    def test_step_reports_per_hop_delay_and_energy_decomposition(self):
        environment = self.make_environment()
        environment.reset()
        _, _, _, _, info = environment.step(0)
        hop = info["hop_metrics"]
        self.assertEqual(hop["hop_index"], 1)
        self.assertEqual(hop["stage_index"], 0)
        self.assertEqual(hop["hop_completed"], 1)
        self.assertAlmostEqual(
            hop["computation_delay_s"],
            hop["computation_queue_delay_s"] + hop["execution_delay_s"],
        )
        self.assertAlmostEqual(
            hop["hop_total_delay_s"],
            hop["communication_delay_s"] + hop["computation_delay_s"],
        )
        self.assertAlmostEqual(
            hop["hop_total_energy_j"],
            hop["communication_energy_j"] + hop["computation_energy_j"],
        )
        self.assertGreaterEqual(hop["route_phase_count"], 0)
        self.assertGreaterEqual(hop["route_used_path_count"], 0)

    def test_ppo_pretraining_does_not_collect_replica_adaptation_traces(self):
        environment = self.make_environment()
        environment.config.adaptation_enabled = False
        environment.reset()
        environment.step(0)
        self.assertEqual(environment.replica_adapter.window_records, [])

    def test_replica_adaptation_waits_until_the_slot_is_finished(self):
        environment = self.make_environment()
        environment.config.deployment_window_requests = 1
        state = environment.reset()
        arrival_slot = environment.topology.absolute_slot(
            environment.runtime.request.arrival_time_s
        )
        while state is not None:
            state, *_ = environment.step(0)
        self.assertEqual(environment.replica_adapter.total_windows, 0)
        environment.finish_time_slot(arrival_slot)
        self.assertEqual(environment.replica_adapter.total_windows, 1)

    def test_same_slot_requests_do_not_share_resource_reservations(self):
        sequential = self.make_environment()
        isolated = self.make_environment()
        sequential.config.adaptation_enabled = False
        isolated.config.adaptation_enabled = False

        first = replace(
            sequential.sample_request(), request_id=100, arrival_time_s=1.0
        )
        second = replace(
            sequential.sample_request(), request_id=101, arrival_time_s=2.0
        )
        self.assertEqual(
            sequential.topology.absolute_slot(first.arrival_time_s),
            sequential.topology.absolute_slot(second.arrival_time_s),
        )

        state = sequential.reset(first)
        while state is not None:
            state, *_ = sequential.step(0)

        second_state = sequential.reset(second)
        isolated_state = isolated.reset(second)
        np.testing.assert_allclose(
            second_state.candidate_features,
            isolated_state.candidate_features,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            second_state.node_features,
            isolated_state.node_features,
            rtol=0.0,
            atol=0.0,
        )

        sequential_info = {}
        isolated_info = {}
        while second_state is not None:
            second_state, _, _, _, sequential_info = sequential.step(0)
        while isolated_state is not None:
            isolated_state, _, _, _, isolated_info = isolated.step(0)
        self.assertAlmostEqual(
            sequential_info["total_latency_s"],
            isolated_info["total_latency_s"],
        )
        self.assertAlmostEqual(
            sequential_info["total_energy_j"],
            isolated_info["total_energy_j"],
        )
        self.assertEqual(
            sequential_info["serving_history"],
            isolated_info["serving_history"],
        )

    def test_interleaved_sessions_match_sequential_independent_requests(self):
        sequential = self.make_environment()
        interleaved = self.make_environment()
        sequential.config.adaptation_enabled = False
        interleaved.config.adaptation_enabled = False
        requests = [
            replace(
                sequential.sample_request(),
                request_id=200 + index,
                arrival_time_s=1.0 + index,
            )
            for index in range(2)
        ]

        sequential_results = []
        for request in requests:
            state = sequential.reset(request)
            final_info = {}
            total_reward = 0.0
            while state is not None:
                state, reward, _, _, final_info = sequential.step(0)
                total_reward += reward
            sequential_results.append((total_reward, final_info))

        sessions = interleaved.start_request_sessions(requests)
        active = list(range(len(sessions)))
        interleaved_rewards = [0.0 for _ in sessions]
        interleaved_infos = [{} for _ in sessions]
        while active:
            next_active = []
            for index in active:
                interleaved.restore_request_session(sessions[index])
                state, reward, _, _, info = interleaved.step(0)
                interleaved_rewards[index] += reward
                interleaved_infos[index] = info
                sessions[index] = interleaved.capture_request_session()
                if state is not None:
                    next_active.append(index)
            active = next_active

        for index, (expected_reward, expected_info) in enumerate(
            sequential_results
        ):
            self.assertAlmostEqual(interleaved_rewards[index], expected_reward)
            self.assertAlmostEqual(
                interleaved_infos[index]["total_latency_s"],
                expected_info["total_latency_s"],
            )
            self.assertAlmostEqual(
                interleaved_infos[index]["total_energy_j"],
                expected_info["total_energy_j"],
            )
            self.assertEqual(
                interleaved_infos[index]["serving_history"],
                expected_info["serving_history"],
            )

    def test_interleaved_sessions_restore_sequential_trace_order(self):
        sequential = self.make_environment()
        interleaved = self.make_environment()
        requests = [
            replace(
                sequential.sample_request(),
                request_id=300 + index,
                arrival_time_s=1.0 + index,
            )
            for index in range(2)
        ]
        for request in requests:
            state = sequential.reset(request)
            while state is not None:
                state, *_ = sequential.step(0)

        sessions = interleaved.start_request_sessions(requests)
        active = list(range(len(sessions)))
        while active:
            next_active = []
            for index in active:
                interleaved.restore_request_session(sessions[index])
                state, *_ = interleaved.step(0)
                sessions[index] = interleaved.capture_request_session()
                if state is not None:
                    next_active.append(index)
            active = next_active
        interleaved.finalize_request_sessions()

        def signature(environment):
            return [
                (
                    item.request_id,
                    item.stage_index,
                    item.service_id,
                    item.serving_node,
                    item.normalized_cost,
                )
                for item in environment.replica_adapter.window_records
            ]

        self.assertEqual(signature(interleaved), signature(sequential))

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
