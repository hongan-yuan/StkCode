"""State specification for PPO-based serving-satellite selection.

The observation contains three logically separate parts:

1. request state, including the service chain, progress, and accumulated cost;
2. a request-specific connected graph with current node/link measurements;
3. sparse connectivity snapshots for the next ``H`` predictable time slots.

The graph node set is the deduplicated union of the source, destination, all
satellites hosting a replica required by the request, and only the relay
satellites needed by the connector tree.  Relay satellites are graph context
only: they never enter the serving-satellite action set.

Relay maintenance is deliberately lazy.  Build the connector tree when a
request arrives or replica placement changes.  At an ordinary service stage,
only refresh measurements and sparse topology snapshots.  At a slot boundary,
repair the affected connector paths only if the current tree is disconnected.
This avoids rebuilding the request graph at every PPO step.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except Exception:  # pragma: no cover - optional training dependency
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to construct or validate PPO observations")


# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------

# Continuous request features are normalized before entering the network.
REQUEST_CONTINUOUS_FEATURES = (
    "arrival_time_normalized",
    "current_time_normalized",
    "remaining_slot_fraction",
    "execution_stage_fraction",
    "current_data_volume_normalized",
    "accumulated_latency_normalized",
    "accumulated_energy_normalized",
)

# Variable-length request fields are stored separately and use service_mask.
REQUEST_SEQUENCE_FIELDS = (
    "service_ids",
    "data_volume_sequence",
    "serving_history_node_ids",
    "service_mask",
)

# A row in node_features represents one unique physical satellite.  Node IDs
# are kept separately and are not treated as continuous neural features.
NODE_FEATURES = (
    "plane_position_sin",
    "plane_position_cos",
    "intra_plane_position_sin",
    "intra_plane_position_cos",
    "computing_capacity_normalized",
    "computing_efficiency",
    "computing_queue_normalized",
    "computing_power_normalized",
    "is_source",
    "is_destination",
    "is_current_data_holder",
    "is_relay",
    "hosts_current_service",
    "selected_before",
)

# Only current-slot, observable link quantities appear here.  Future traffic,
# future transmission queues, and future efficiency factors are excluded.
CURRENT_EDGE_FEATURES = (
    "effective_rate_normalized",
    "transmission_efficiency",
    "transmission_queue_normalized",
    "transmission_power_normalized",
    "distance_normalized",
)

# The shared candidate scorer receives exactly these three known quantities in
# addition to the candidate node embedding and global request/graph context.
CANDIDATE_FEATURES = (
    "hop_distance_normalized",
    "bottleneck_rate_normalized",
    "computing_queue_normalized",
)


@dataclass(frozen=True)
class SparseTopology:
    """One predictable topology snapshot in COO-style sparse form.

    ``edge_index`` has shape ``[2, E]`` and contains local row indices into the
    observation's node tensor.  An undirected ISL may be stored once and made
    bidirectional by the graph encoder, or stored in both directions; the same
    convention must be used for every snapshot.

    Only connectivity derived from orbital motion is included for a future
    slot.  No future queue, background traffic, or computing load is exposed.
    ``slot_offset=0`` denotes the current topology; positive offsets denote
    predictable future topology.
    """

    edge_index: torch.Tensor
    slot_offset: int

    def validate(self, node_count: int) -> None:
        _require_torch()
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.slot_offset < 0:
            raise ValueError("slot_offset must be non-negative")
        if self.edge_index.numel() == 0:
            return
        low = int(self.edge_index.min().item())
        high = int(self.edge_index.max().item())
        if low < 0 or high >= node_count:
            raise ValueError("edge_index contains an invalid local node index")


@dataclass
class ServingSelectionObservation:
    """Variable-size observation consumed by the PPO actor-critic.

    Shapes:
        request_continuous: ``[F_request]``
        service_ids: ``[L_max]``
        data_volume_sequence: ``[L_max + 1]``
        serving_history_node_ids: ``[L_max]``
        service_mask: ``[L_max]``
        node_ids: ``[N]``
        node_features: ``[N, F_node]``
        node_mask: ``[N]``
        current_edge_index: ``[2, E]``
        current_edge_features: ``[E, F_edge]``
        future_topologies: one sparse snapshot for each future slot
        candidate_indices: ``[C]`` local indices into node_features
        candidate_features: ``[C, 3]`` in CANDIDATE_FEATURES order
        action_mask: ``[C]``

    Padding is needed only when observations are batched.  service_mask,
    node_mask, and action_mask have different meanings and must remain
    separate.
    """

    request_continuous: torch.Tensor
    service_ids: torch.Tensor
    data_volume_sequence: torch.Tensor
    serving_history_node_ids: torch.Tensor
    service_mask: torch.Tensor

    node_ids: torch.Tensor
    node_features: torch.Tensor
    node_mask: torch.Tensor

    current_edge_index: torch.Tensor
    current_edge_features: torch.Tensor
    future_topologies: tuple[SparseTopology, ...]

    candidate_indices: torch.Tensor
    candidate_features: torch.Tensor
    action_mask: torch.Tensor

    def validate(self) -> None:
        """Fail early when graph, padding, and action mappings disagree."""

        _require_torch()
        if self.request_continuous.shape != (len(REQUEST_CONTINUOUS_FEATURES),):
            raise ValueError("request_continuous has an invalid feature dimension")

        chain_length = self.service_ids.numel()
        if self.service_ids.dtype != torch.long:
            raise TypeError("service_ids must have dtype torch.long")
        if self.service_mask.shape != self.service_ids.shape:
            raise ValueError("service_mask must have the same shape as service_ids")
        if self.serving_history_node_ids.shape != self.service_ids.shape:
            raise ValueError("serving history must have the same shape as service_ids")
        if self.data_volume_sequence.numel() != chain_length + 1:
            raise ValueError("data_volume_sequence must have L_max + 1 entries")

        node_count = self.node_ids.numel()
        if self.node_ids.dtype != torch.long:
            raise TypeError("node_ids must have dtype torch.long")
        if self.node_features.shape != (node_count, len(NODE_FEATURES)):
            raise ValueError("node_features has an invalid shape")
        if self.node_mask.shape != (node_count,):
            raise ValueError("node_mask must have shape [N]")
        real_node_ids = self.node_ids[self.node_mask.bool()]
        if real_node_ids.unique().numel() != real_node_ids.numel():
            raise ValueError("every physical satellite must appear at most once")

        if self.current_edge_index.dtype != torch.long:
            raise TypeError("current_edge_index must have dtype torch.long")
        if self.current_edge_index.ndim != 2 or self.current_edge_index.shape[0] != 2:
            raise ValueError("current_edge_index must have shape [2, E]")
        edge_count = self.current_edge_index.shape[1]
        if self.current_edge_features.shape != (
            edge_count,
            len(CURRENT_EDGE_FEATURES),
        ):
            raise ValueError("current_edge_features has an invalid shape")
        SparseTopology(self.current_edge_index, slot_offset=0).validate(node_count)

        expected_offset = 1
        for topology in self.future_topologies:
            topology.validate(node_count)
            if topology.slot_offset != expected_offset:
                raise ValueError("future topology offsets must be consecutive from 1")
            expected_offset += 1

        candidate_count = self.candidate_indices.numel()
        if self.candidate_indices.dtype != torch.long:
            raise TypeError("candidate_indices must have dtype torch.long")
        if self.candidate_features.shape != (
            candidate_count,
            len(CANDIDATE_FEATURES),
        ):
            raise ValueError("candidate_features must have shape [C, 3]")
        if self.action_mask.shape != (candidate_count,):
            raise ValueError("action_mask must have shape [C]")
        if candidate_count and (
            int(self.candidate_indices.min().item()) < 0
            or int(self.candidate_indices.max().item()) >= node_count
        ):
            raise ValueError("candidate_indices contains an invalid local node index")

        valid_candidates = self.candidate_indices[self.action_mask.bool()]
        if valid_candidates.numel() == 0:
            raise ValueError("a non-terminal observation needs at least one valid action")
        if not bool(self.node_mask[valid_candidates].all()):
            raise ValueError("an action candidate cannot refer to a padded node")

        relay_index = NODE_FEATURES.index("is_relay")
        host_index = NODE_FEATURES.index("hosts_current_service")
        if bool((self.node_features[valid_candidates, relay_index] > 0.5).any()):
            raise ValueError("relay satellites cannot be serving candidates")
        if not bool((self.node_features[valid_candidates, host_index] > 0.5).all()):
            raise ValueError("every valid candidate must host the current service")


# Backward-readable schema aliases.  These describe feature order only; actual
# observations are variable-size tensors in ServingSelectionObservation.
user_request = list(REQUEST_CONTINUOUS_FEATURES + REQUEST_SEQUENCE_FIELDS)
satellite_link_state = list(NODE_FEATURES)


"""
Joint encoder used by the PPO actor-critic
------------------------------------------

1. Encode normalized request scalars and the masked service sequence into h_r.
2. Encode current node and edge measurements with edge-aware graph attention;
   current_edge_index is the attention/message-passing structure, not a vector
   flattened into an independent MLP.
3. Apply the same spatial graph encoder to every future SparseTopology using
   only predictable connectivity plus a slot-offset embedding.  Fuse the
   per-node representations across time with temporal attention or a GRU.
4. Use request-conditioned attention pooling to obtain a graph summary h_g.
5. For candidate v, the shared scorer consumes
       [z_v, z_current, z_destination, h_r, h_g,
        hop_distance, bottleneck_rate, computing_queue]
   and emits one logit.  Apply action_mask before the candidate softmax.
6. The critic consumes [z_current, z_destination, h_r, h_g] and emits V(s).

Future snapshots may contain deterministic orbital connectivity only.  Future
background traffic, link queues, transmission efficiency, and computing queues
must never be added to the observation.

After action a_i, execute routing and computation, obtain their actual finish
time, update accumulated latency/energy and the current data holder, refresh all
current measurements, and then update the sparse topology horizon.  The reward
is
    -(alpha * stage_latency / T_0 + beta * stage_energy / E_0).
The last-stage reward also includes final-output delivery to the destination.
"""
