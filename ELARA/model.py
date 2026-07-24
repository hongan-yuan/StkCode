from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional training dependency
    torch = None
    nn = None
    F = None

from .state import EDGE_FEATURES, NODE_FEATURES, REQUEST_FEATURES, ServingSelectionState


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for ELARA PPO. Install ELARA/requirements.txt "
            "in the Python environment used for training."
        )


def state_to_tensors(state: ServingSelectionState, device):
    require_torch()
    return {
        "request": torch.as_tensor(state.request_continuous, dtype=torch.float32, device=device),
        "service_ids": torch.as_tensor(state.service_ids, dtype=torch.long, device=device),
        "service_mask": torch.as_tensor(state.service_mask, dtype=torch.bool, device=device),
        "node_features": torch.as_tensor(state.node_features, dtype=torch.float32, device=device),
        "node_mask": torch.as_tensor(state.node_mask, dtype=torch.bool, device=device),
        "edge_index": torch.as_tensor(state.current_edge_index, dtype=torch.long, device=device),
        "edge_features": torch.as_tensor(state.current_edge_features, dtype=torch.float32, device=device),
        "future_edges": [
            torch.as_tensor(item.edge_index, dtype=torch.long, device=device)
            for item in state.future_topologies
        ],
        "candidate_indices": torch.as_tensor(state.candidate_indices, dtype=torch.long, device=device),
        "candidate_features": torch.as_tensor(state.candidate_features, dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(state.action_mask, dtype=torch.bool, device=device),
        "current_index": torch.as_tensor(
            int(np.argmax(state.node_features[:, NODE_FEATURES.index("is_current")])),
            dtype=torch.long,
            device=device,
        ),
        "destination_index": torch.as_tensor(
            int(np.argmax(state.node_features[:, NODE_FEATURES.index("is_destination")])),
            dtype=torch.long,
            device=device,
        ),
    }


def batch_states_to_tensors(states: list[ServingSelectionState], device):
    """Pack variable request graphs into one disjoint graph mini batch."""
    require_torch()
    if not states:
        raise ValueError("at least one state is required")

    batch_size = len(states)
    node_counts = [len(state.node_ids) for state in states]
    node_offsets = np.cumsum([0, *node_counts[:-1]], dtype=np.int64)
    candidate_counts = [len(state.candidate_indices) for state in states]
    max_candidates = max(candidate_counts)
    max_services = max(len(state.service_ids) for state in states)
    future_count = max(len(state.future_topologies) for state in states)

    service_ids = np.full((batch_size, max_services), -1, dtype=np.int64)
    service_mask = np.zeros((batch_size, max_services), dtype=bool)
    current_indices = np.empty(batch_size, dtype=np.int64)
    destination_indices = np.empty(batch_size, dtype=np.int64)
    edge_indices = []
    future_edges: list[list[np.ndarray]] = [[] for _ in range(future_count)]
    candidate_indices = []
    candidate_batch = []
    candidate_local = []

    for batch_index, (state, node_offset) in enumerate(zip(states, node_offsets)):
        service_count = len(state.service_ids)
        service_ids[batch_index, :service_count] = state.service_ids
        service_mask[batch_index, :service_count] = state.service_mask
        current_indices[batch_index] = node_offset + int(
            np.argmax(state.node_features[:, NODE_FEATURES.index("is_current")])
        )
        destination_indices[batch_index] = node_offset + int(
            np.argmax(state.node_features[:, NODE_FEATURES.index("is_destination")])
        )
        edge_indices.append(state.current_edge_index + node_offset)
        for offset in range(future_count):
            if offset < len(state.future_topologies):
                item = state.future_topologies[offset].edge_index + node_offset
            else:
                item = np.empty((2, 0), dtype=np.int64)
            future_edges[offset].append(item)
        candidate_indices.append(state.candidate_indices + node_offset)
        candidate_batch.append(
            np.full(len(state.candidate_indices), batch_index, dtype=np.int64)
        )
        candidate_local.append(
            np.arange(len(state.candidate_indices), dtype=np.int64)
        )

    def concatenate(items, axis, shape, dtype):
        return (
            np.concatenate(items, axis=axis)
            if any(item.size for item in items)
            else np.empty(shape, dtype=dtype)
        )

    packed_edge_index = concatenate(
        edge_indices, axis=1, shape=(2, 0), dtype=np.int64
    )
    packed_edge_features = concatenate(
        [state.current_edge_features for state in states],
        axis=0,
        shape=(0, len(EDGE_FEATURES)),
        dtype=np.float32,
    )
    return {
        "request": torch.as_tensor(
            np.stack([state.request_continuous for state in states]),
            dtype=torch.float32,
            device=device,
        ),
        "service_ids": torch.as_tensor(
            service_ids, dtype=torch.long, device=device
        ),
        "service_mask": torch.as_tensor(
            service_mask, dtype=torch.bool, device=device
        ),
        "node_features": torch.as_tensor(
            np.concatenate([state.node_features for state in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "node_mask": torch.as_tensor(
            np.concatenate([state.node_mask for state in states]),
            dtype=torch.bool,
            device=device,
        ),
        "node_batch": torch.as_tensor(
            np.repeat(np.arange(batch_size, dtype=np.int64), node_counts),
            dtype=torch.long,
            device=device,
        ),
        "edge_index": torch.as_tensor(
            packed_edge_index, dtype=torch.long, device=device
        ),
        "edge_features": torch.as_tensor(
            packed_edge_features, dtype=torch.float32, device=device
        ),
        "future_edges": [
            torch.as_tensor(
                concatenate(items, axis=1, shape=(2, 0), dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            for items in future_edges
        ],
        "candidate_indices": torch.as_tensor(
            np.concatenate(candidate_indices),
            dtype=torch.long,
            device=device,
        ),
        "candidate_features": torch.as_tensor(
            np.concatenate([state.candidate_features for state in states], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "candidate_batch": torch.as_tensor(
            np.concatenate(candidate_batch), dtype=torch.long, device=device
        ),
        "candidate_local": torch.as_tensor(
            np.concatenate(candidate_local), dtype=torch.long, device=device
        ),
        "action_mask": torch.as_tensor(
            np.concatenate([state.action_mask for state in states]),
            dtype=torch.bool,
            device=device,
        ),
        "current_indices": torch.as_tensor(
            current_indices, dtype=torch.long, device=device
        ),
        "destination_indices": torch.as_tensor(
            destination_indices, dtype=torch.long, device=device
        ),
        "batch_size": batch_size,
        "max_candidates": max_candidates,
    }


if nn is not None:

    class EdgeAwareGraphAttention(nn.Module):
        """Sparse graph attention without a torch-geometric dependency."""

        def __init__(self, hidden_dim: int, edge_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.edge_bias = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.edge_value = nn.Linear(edge_dim, hidden_dim, bias=False)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, node_state, edge_index, edge_features):
            node_count = node_state.shape[0]
            if edge_index.numel() == 0:
                return self.norm(node_state + F.relu(self.output(node_state)))
            source, target = edge_index[0], edge_index[1]
            query = self.query(node_state)
            key = self.key(node_state)
            value = self.value(node_state)
            scores = (query[target] * key[source]).sum(dim=-1) / self.hidden_dim**0.5
            scores = scores + self.edge_bias(edge_features).squeeze(-1)
            edge_messages = value[source] + self.edge_value(edge_features)

            # Stable segment softmax over all incoming edges of each target.
            # scatter_reduce and index_add keep the entire operation on-device,
            # avoiding one Python loop and one host synchronization per node.
            target_max = torch.full(
                (node_count,),
                -torch.inf,
                dtype=scores.dtype,
                device=scores.device,
            )
            target_max.scatter_reduce_(
                0, target, scores, reduce="amax", include_self=True
            )
            exp_scores = torch.exp(scores - target_max[target])
            target_sum = torch.zeros(
                node_count, dtype=scores.dtype, device=scores.device
            )
            target_sum.scatter_add_(0, target, exp_scores)
            weights = exp_scores / target_sum[target].clamp_min(
                torch.finfo(scores.dtype).tiny
            )
            aggregated = torch.zeros_like(node_state)
            aggregated.index_add_(0, target, weights.unsqueeze(-1) * edge_messages)
            return self.norm(node_state + F.relu(self.output(aggregated)))


    class ELARANetwork(nn.Module):
        def __init__(
            self,
            num_services: int,
            hidden_dim: int = 128,
            graph_layers: int = 2,
            attention_heads: int = 4,
            service_embedding_dim: int = 32,
            max_future_horizon: int = 8,
        ):
            super().__init__()
            if hidden_dim % attention_heads:
                raise ValueError("hidden_dim must be divisible by attention_heads")
            self.hidden_dim = hidden_dim
            self.node_encoder = nn.Sequential(
                nn.Linear(len(NODE_FEATURES), hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(len(EDGE_FEATURES), hidden_dim // 2), nn.ReLU(),
                nn.Linear(hidden_dim // 2, len(EDGE_FEATURES)),
            )
            self.graph_layers = nn.ModuleList(
                EdgeAwareGraphAttention(hidden_dim, len(EDGE_FEATURES))
                for _ in range(graph_layers)
            )
            self.time_embedding = nn.Embedding(max_future_horizon + 1, hidden_dim)
            self.temporal_attention = nn.MultiheadAttention(
                hidden_dim, attention_heads, batch_first=True
            )

            self.service_embedding = nn.Embedding(num_services + 1, service_embedding_dim, padding_idx=0)
            self.request_encoder = nn.Sequential(
                nn.Linear(len(REQUEST_FEATURES) + service_embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )

            self.pool_query = nn.Linear(hidden_dim * 3, hidden_dim)
            self.pool_key = nn.Linear(hidden_dim, hidden_dim)
            self.pool_value = nn.Linear(hidden_dim, hidden_dim)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim * 5 + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def _spatial(self, initial, edge_index, edge_features):
            state = initial
            encoded_edges = self.edge_encoder(edge_features)
            for layer in self.graph_layers:
                state = layer(state, edge_index, encoded_edges)
            return state

        def _request(self, request, service_ids, service_mask):
            # Shift IDs by one so padding ID 0 remains reserved.
            embedded = self.service_embedding(service_ids + 1)
            weights = service_mask.float().unsqueeze(-1)
            if embedded.ndim == 2:
                chain = (embedded * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
            else:
                chain = (embedded * weights).sum(dim=1) / weights.sum(
                    dim=1
                ).clamp_min(1.0)
            return self.request_encoder(torch.cat((request, chain), dim=-1))

        @staticmethod
        def _segment_softmax(scores, segment, segment_count):
            maximum = torch.full(
                (segment_count,),
                -torch.inf,
                dtype=scores.dtype,
                device=scores.device,
            )
            maximum.scatter_reduce_(
                0, segment, scores, reduce="amax", include_self=True
            )
            exponent = torch.exp(scores - maximum[segment])
            denominator = torch.zeros(
                segment_count, dtype=scores.dtype, device=scores.device
            )
            denominator.scatter_add_(0, segment, exponent)
            return exponent / denominator[segment].clamp_min(
                torch.finfo(scores.dtype).tiny
            )

        def forward(self, observation):
            node_base = self.node_encoder(observation["node_features"])
            current = self._spatial(
                node_base,
                observation["edge_index"],
                observation["edge_features"],
            )
            snapshots = [current]
            zero_edge_dim = len(EDGE_FEATURES)
            for offset, future_edge_index in enumerate(observation["future_edges"], start=1):
                time_id = min(offset, self.time_embedding.num_embeddings - 1)
                future_initial = node_base + self.time_embedding.weight[time_id]
                future_edge_features = torch.zeros(
                    (future_edge_index.shape[1], zero_edge_dim),
                    dtype=node_base.dtype,
                    device=node_base.device,
                )
                snapshots.append(self._spatial(future_initial, future_edge_index, future_edge_features))
            temporal = torch.stack(snapshots, dim=1)  # [N, H+1, d]
            node_state, _ = self.temporal_attention(
                temporal[:, :1, :], temporal, temporal, need_weights=False
            )
            node_state = node_state[:, 0, :]

            request_state = self._request(
                observation["request"], observation["service_ids"], observation["service_mask"]
            )
            current_idx = observation["current_index"]
            destination_idx = observation["destination_index"]
            current_state = node_state[current_idx]
            destination_state = node_state[destination_idx]

            query = self.pool_query(torch.cat((request_state, current_state, destination_state)) )
            keys = self.pool_key(node_state)
            pool_logits = (keys * query).sum(dim=-1) / self.hidden_dim**0.5
            pool_logits = pool_logits.masked_fill(~observation["node_mask"], torch.finfo(pool_logits.dtype).min)
            pool_weights = torch.softmax(pool_logits, dim=0)
            graph_state = (pool_weights.unsqueeze(-1) * self.pool_value(node_state)).sum(dim=0)

            candidate_state = node_state[observation["candidate_indices"]]
            candidate_count = candidate_state.shape[0]
            shared = torch.cat(
                (current_state, destination_state, request_state, graph_state), dim=-1
            ).unsqueeze(0).expand(candidate_count, -1)
            actor_input = torch.cat(
                (candidate_state, shared, observation["candidate_features"]), dim=-1
            )
            logits = self.actor(actor_input).squeeze(-1)
            logits = logits.masked_fill(
                ~observation["action_mask"], torch.finfo(logits.dtype).min
            )
            value = self.critic(
                torch.cat((current_state, destination_state, request_state, graph_state))
            ).squeeze(-1)
            return logits, value

        def forward_batch(self, observation):
            """One forward pass over a disjoint batch of variable request graphs."""
            batch_size = observation["batch_size"]
            node_batch = observation["node_batch"]
            node_base = self.node_encoder(observation["node_features"])
            current = self._spatial(
                node_base,
                observation["edge_index"],
                observation["edge_features"],
            )
            snapshots = [current]
            zero_edge_dim = len(EDGE_FEATURES)
            for offset, future_edge_index in enumerate(
                observation["future_edges"], start=1
            ):
                time_id = min(offset, self.time_embedding.num_embeddings - 1)
                future_initial = node_base + self.time_embedding.weight[time_id]
                future_edge_features = torch.zeros(
                    (future_edge_index.shape[1], zero_edge_dim),
                    dtype=node_base.dtype,
                    device=node_base.device,
                )
                snapshots.append(
                    self._spatial(
                        future_initial,
                        future_edge_index,
                        future_edge_features,
                    )
                )
            temporal = torch.stack(snapshots, dim=1)
            node_state, _ = self.temporal_attention(
                temporal[:, :1, :], temporal, temporal, need_weights=False
            )
            node_state = node_state[:, 0, :]

            request_state = self._request(
                observation["request"],
                observation["service_ids"],
                observation["service_mask"],
            )
            current_state = node_state[observation["current_indices"]]
            destination_state = node_state[observation["destination_indices"]]
            query = self.pool_query(
                torch.cat((request_state, current_state, destination_state), dim=-1)
            )
            keys = self.pool_key(node_state)
            pool_logits = (
                keys * query[node_batch]
            ).sum(dim=-1) / self.hidden_dim**0.5
            pool_logits = pool_logits.masked_fill(
                ~observation["node_mask"], torch.finfo(pool_logits.dtype).min
            )
            pool_weights = self._segment_softmax(
                pool_logits, node_batch, batch_size
            )
            weighted_values = (
                pool_weights.unsqueeze(-1) * self.pool_value(node_state)
            )
            graph_state = torch.zeros(
                (batch_size, self.hidden_dim),
                dtype=node_state.dtype,
                device=node_state.device,
            )
            graph_state.index_add_(0, node_batch, weighted_values)

            candidate_batch = observation["candidate_batch"]
            candidate_state = node_state[observation["candidate_indices"]]
            shared_by_graph = torch.cat(
                (current_state, destination_state, request_state, graph_state),
                dim=-1,
            )
            actor_input = torch.cat(
                (
                    candidate_state,
                    shared_by_graph[candidate_batch],
                    observation["candidate_features"],
                ),
                dim=-1,
            )
            flat_logits = self.actor(actor_input).squeeze(-1)
            flat_logits = flat_logits.masked_fill(
                ~observation["action_mask"], torch.finfo(flat_logits.dtype).min
            )
            logits = torch.full(
                (batch_size, observation["max_candidates"]),
                torch.finfo(flat_logits.dtype).min,
                dtype=flat_logits.dtype,
                device=flat_logits.device,
            )
            logits[
                candidate_batch, observation["candidate_local"]
            ] = flat_logits
            values = self.critic(
                torch.cat(
                    (
                        current_state,
                        destination_state,
                        request_state,
                        graph_state,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            return logits, values

else:

    class EdgeAwareGraphAttention:  # pragma: no cover - optional dependency
        def __init__(self, *args, **kwargs):
            require_torch()

    class ELARANetwork:  # pragma: no cover - exercised only without torch
        def __init__(self, *args, **kwargs):
            require_torch()
