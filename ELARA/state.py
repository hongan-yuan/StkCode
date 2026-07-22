from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REQUEST_FEATURES = (
    "arrival_time",
    "current_time",
    "remaining_slot_fraction",
    "execution_stage_fraction",
    "current_data_volume",
    "accumulated_latency",
    "accumulated_energy",
)

NODE_FEATURES = (
    "plane_sin", "plane_cos", "position_sin", "position_cos",
    "compute_capacity", "compute_efficiency", "compute_queue", "compute_power",
    "is_source", "is_destination", "is_current", "is_relay",
    "hosts_current_service", "selected_before",
)

EDGE_FEATURES = (
    "effective_rate", "transmission_efficiency", "transmission_queue",
    "transmission_power", "distance",
)

CANDIDATE_FEATURES = ("hop_distance", "bottleneck_rate", "computing_queue")


@dataclass
class SparseTopologyState:
    edge_index: np.ndarray
    slot_offset: int


@dataclass
class ServingSelectionState:
    request_continuous: np.ndarray
    service_ids: np.ndarray
    data_volumes: np.ndarray
    serving_history: np.ndarray
    service_mask: np.ndarray
    node_ids: np.ndarray
    node_features: np.ndarray
    node_mask: np.ndarray
    current_edge_index: np.ndarray
    current_edge_features: np.ndarray
    future_topologies: tuple[SparseTopologyState, ...]
    candidate_indices: np.ndarray
    candidate_features: np.ndarray
    action_mask: np.ndarray

    def validate(self) -> None:
        n = len(self.node_ids)
        if self.request_continuous.shape != (len(REQUEST_FEATURES),):
            raise ValueError("invalid request feature shape")
        if self.node_features.shape != (n, len(NODE_FEATURES)):
            raise ValueError("invalid node feature shape")
        if len(set(map(int, self.node_ids[self.node_mask]))) != int(self.node_mask.sum()):
            raise ValueError("physical nodes must be unique")
        if self.current_edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.current_edge_features.shape != (self.current_edge_index.shape[1], len(EDGE_FEATURES)):
            raise ValueError("current edge features do not align with edge_index")
        if self.candidate_features.shape != (len(self.candidate_indices), len(CANDIDATE_FEATURES)):
            raise ValueError("invalid candidate feature shape")
        if not self.action_mask.any():
            raise ValueError("state has no valid serving action")

