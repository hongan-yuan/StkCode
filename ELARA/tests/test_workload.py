from __future__ import annotations

import unittest

from ELARA.background import MarkovBackgroundProcess
from ELARA.config import ELARAConfig
from ELARA.environment import ELARAEnvironment
from ELARA.topology import TemporalTopology


class WorkloadTests(unittest.TestCase):
    def make_environment(self, **overrides):
        values = dict(
            seed=19,
            num_planes=2,
            sats_per_plane=6,
            num_services=4,
            replica_count_range=(2, 3),
            request_template_chain_lengths=(2, 3, 4),
            service_memory_gb_min=1.0,
            service_memory_gb_max=1.0,
            satellite_memory_capacity_gb=8.0,
            max_trace_slots=4,
        )
        values.update(overrides)
        config = ELARAConfig(
            **values,
        )
        return ELARAEnvironment(config, TemporalTopology.synthetic_ring(12, 4))

    def test_dynamic_initial_replica_counts_and_memory_limits(self):
        environment = self.make_environment()
        counts = [len(service.replicas) for service in environment.services.values()]
        self.assertTrue(all(2 <= count <= 3 for count in counts))
        usage = {node: 0.0 for node in environment.resources}
        for service in environment.services.values():
            for node in service.replicas:
                usage[node] += service.memory_requirement_gb
        self.assertTrue(
            all(usage[node] <= resource.memory_capacity_gb for node, resource in environment.resources.items())
        )

    def test_default_request_and_background_loads_are_feasible(self):
        config = ELARAConfig()
        self.assertEqual(
            (config.service_cycles_min, config.service_cycles_max),
            (1.0e8, 1.0e9),
        )
        self.assertEqual(
            (config.input_data_gb_min, config.input_data_gb_max),
            (0.02, 0.20),
        )
        self.assertEqual(config.background_load_scale, 0.5)
        self.assertEqual(config.ppo_update_interval_slots, 5)
        self.assertEqual(
            (config.ppo_pretrain_cycles, config.ppo_joint_training_cycles),
            (1, 1),
        )

    def test_three_templates_and_poisson_arrival_clock(self):
        environment = self.make_environment()
        self.assertEqual(
            sorted(len(template.services) for template in environment.request_templates),
            [2, 3, 4],
        )
        requests = [environment.sample_request() for _ in range(20)]
        self.assertTrue(all(left.arrival_time_s < right.arrival_time_s for left, right in zip(requests, requests[1:])))
        self.assertTrue(all(request.template_id in {1, 2, 3} for request in requests))

    def test_cycle_request_count_is_generated_by_arrival_process(self):
        environment = self.make_environment()
        cycle_end = environment.topology.slot_count * environment.topology.slot_duration_s
        admitted = []
        while True:
            request = environment.sample_request()
            if request.arrival_time_s >= cycle_end:
                first_outside = request
                break
            admitted.append(request)
        self.assertTrue(admitted)
        self.assertTrue(all(request.arrival_time_s < cycle_end for request in admitted))
        self.assertGreaterEqual(first_outside.arrival_time_s, cycle_end)

    def test_inter_request_reservations_are_cleared(self):
        environment = self.make_environment()
        environment.compute_available_at[0] = 123.0
        environment.link_available_at[(0, 1)] = 123.0
        request = environment.sample_request()
        environment.reset(request)
        self.assertEqual(
            environment.compute_available_at[0], request.arrival_time_s
        )
        self.assertEqual(environment.link_available_at, {})

    def test_poisson_requests_are_batched_in_slot_order(self):
        environment = self.make_environment()
        batches = list(environment.iter_request_batches(slot_count=4))
        self.assertEqual([slot for slot, _ in batches], [0, 1, 2, 3])
        arrivals = []
        for slot, requests in batches:
            for request in requests:
                self.assertEqual(
                    int(request.arrival_time_s // environment.topology.slot_duration_s),
                    slot,
                )
                arrivals.append(request.arrival_time_s)
        self.assertEqual(arrivals, sorted(arrivals))

    def test_request_stream_is_exogenous_to_replica_adaptation(self):
        first = self.make_environment()
        second = self.make_environment()
        for service_id, service in second.services.items():
            service.replicas = [(service_id + 5) % second.config.total_satellites]
        first_requests = [first.sample_request() for _ in range(20)]
        second_requests = [second.sample_request() for _ in range(20)]
        self.assertEqual(first_requests, second_requests)

    def test_request_and_background_seeds_are_independent(self):
        first = self.make_environment(request_seed=123, background_seed=456)
        second = self.make_environment(request_seed=123, background_seed=789)
        self.assertEqual(
            [first.sample_request() for _ in range(20)],
            [second.sample_request() for _ in range(20)],
        )
        first_background = first.background.compute(0, 0)
        second_background = second.background.compute(0, 0)
        self.assertNotEqual(first_background, second_background)

    def test_chain_filter_and_total_arrival_rate(self):
        environment = self.make_environment(
            request_chain_length_filter=3,
            request_arrival_lambda_total_per_slot=1.25,
        )
        self.assertEqual(
            [len(template.services) for template in environment.request_templates],
            [3],
        )
        self.assertEqual(
            environment.config.request_arrival_lambda_total_per_slot, 1.25
        )

    def test_markov_background_is_reproducible_and_slot_correlated(self):
        environment = self.make_environment()
        first = MarkovBackgroundProcess(
            environment.topology, environment.resources, environment.config, seed=123
        )
        second = MarkovBackgroundProcess(
            environment.topology, environment.resources, environment.config, seed=123
        )
        compute_a = [first.compute(0, slot) for slot in range(8)]
        compute_b = [second.compute(0, slot) for slot in range(8)]
        self.assertEqual(compute_a, compute_b)
        edge = environment.topology.graph_at_slot(0).edge(0, 1)
        self.assertEqual(first.link(edge, 3), second.link(edge, 3))


if __name__ == "__main__":
    unittest.main()
