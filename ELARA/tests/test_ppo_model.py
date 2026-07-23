from __future__ import annotations

import math
import unittest

import numpy as np

from ELARA.config import ELARAConfig
from ELARA.model import EdgeAwareGraphAttention, F, torch
from ELARA.ppo import PPOAgent, PPOTransition
from ELARA.state import (
    CANDIDATE_FEATURES,
    EDGE_FEATURES,
    NODE_FEATURES,
    REQUEST_FEATURES,
    ServingSelectionState,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PPOModelTests(unittest.TestCase):
    def test_vectorized_attention_matches_segment_reference(self):
        torch.manual_seed(7)
        layer = EdgeAwareGraphAttention(hidden_dim=8, edge_dim=3)
        node_state = torch.randn(6, 8, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 1], [1, 1, 3, 3, 3, 0, 5]],
            dtype=torch.long,
        )
        edge_features = torch.randn(edge_index.shape[1], 3)

        actual = layer(node_state, edge_index, edge_features)

        source, target = edge_index
        query = layer.query(node_state)
        key = layer.key(node_state)
        value = layer.value(node_state)
        scores = (query[target] * key[source]).sum(dim=-1) / math.sqrt(8)
        scores = scores + layer.edge_bias(edge_features).squeeze(-1)
        messages = value[source] + layer.edge_value(edge_features)
        aggregated = torch.zeros_like(node_state)
        for node in range(node_state.shape[0]):
            mask = target == node
            if mask.any():
                weights = torch.softmax(scores[mask], dim=0)
                aggregated[node] = (weights.unsqueeze(-1) * messages[mask]).sum(dim=0)
        expected = layer.norm(node_state + F.relu(layer.output(aggregated)))

        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(node_state.grad).all())

    def test_ppo_steps_optimizer_once_per_minibatch(self):
        config = ELARAConfig(
            num_planes=1,
            sats_per_plane=4,
            num_services=2,
            hidden_dim=16,
            graph_layers=1,
            attention_heads=4,
            service_embedding_dim=4,
            future_topology_horizon=0,
            ppo_epochs=3,
            rollout_steps=5,
            ppo_minibatch_size=2,
        )
        agent = PPOAgent(config, "cpu")
        state = self._state()
        action, old_log_prob, value = agent.act(state)
        for index in range(5):
            agent.remember(
                PPOTransition(
                    state=state,
                    action=action,
                    old_log_prob=old_log_prob,
                    value=value,
                    reward=1.0,
                    done=index == 4,
                )
            )

        step_count = 0
        original_step = agent.optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        agent.optimizer.step = counted_step
        losses = agent.update(None)

        self.assertEqual(step_count, config.ppo_epochs * math.ceil(5 / 2))
        self.assertFalse(agent.buffer)
        self.assertEqual(set(losses), {"policy_loss", "value_loss", "entropy"})

    @staticmethod
    def _state() -> ServingSelectionState:
        node_features = np.zeros((4, len(NODE_FEATURES)), dtype=np.float32)
        node_features[0, NODE_FEATURES.index("is_current")] = 1.0
        node_features[3, NODE_FEATURES.index("is_destination")] = 1.0
        edge_index = np.asarray(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64
        )
        return ServingSelectionState(
            request_continuous=np.zeros(len(REQUEST_FEATURES), dtype=np.float32),
            service_ids=np.asarray([0, 1], dtype=np.int64),
            data_volumes=np.ones(2, dtype=np.float32),
            serving_history=np.zeros(2, dtype=np.int64),
            service_mask=np.asarray([True, True]),
            node_ids=np.arange(4, dtype=np.int64),
            node_features=node_features,
            node_mask=np.ones(4, dtype=bool),
            current_edge_index=edge_index,
            current_edge_features=np.ones(
                (edge_index.shape[1], len(EDGE_FEATURES)), dtype=np.float32
            ),
            future_topologies=(),
            candidate_indices=np.asarray([1, 2], dtype=np.int64),
            candidate_features=np.ones(
                (2, len(CANDIDATE_FEATURES)), dtype=np.float32
            ),
            action_mask=np.asarray([True, True]),
        )


if __name__ == "__main__":
    unittest.main()
