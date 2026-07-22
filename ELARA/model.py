from __future__ import annotations

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
            aggregated = torch.zeros_like(node_state)
            # Request graphs are small and degree <= 4; this explicit segmented
            # softmax keeps the implementation dependency-free and transparent.
            for node in range(node_count):
                mask = target == node
                if bool(mask.any()):
                    weights = torch.softmax(scores[mask], dim=0)
                    aggregated[node] = (weights.unsqueeze(-1) * edge_messages[mask]).sum(dim=0)
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
            chain = (embedded * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
            return self.request_encoder(torch.cat((request, chain), dim=-1))

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
            features = observation["node_features"]
            current_idx = int(torch.argmax(features[:, NODE_FEATURES.index("is_current")]).item())
            destination_idx = int(torch.argmax(features[:, NODE_FEATURES.index("is_destination")]).item())
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

else:

    class ELARANetwork:  # pragma: no cover - exercised only without torch
        def __init__(self, *args, **kwargs):
            require_torch()

