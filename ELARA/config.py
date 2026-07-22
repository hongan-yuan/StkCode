from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ELARAConfig:
    seed: int = 42
    num_planes: int = 10
    sats_per_plane: int = 18
    num_services: int = 24
    replicas_per_service: int = 4
    chain_length: int = 5

    trace_csv: Path = PROJECT_ROOT / "WalkerDeltaConstellationSimu" / "Walker_Delta_ISL_Simu.csv"
    max_trace_slots: int | None = 120
    slot_duration_s: float = 10.0
    future_topology_horizon: int = 3

    compute_capacity_gflops_min: float = 80.0
    compute_capacity_gflops_max: float = 240.0
    compute_power_w_min: float = 45.0
    compute_power_w_max: float = 85.0
    service_cycles_min: float = 0.5e9
    service_cycles_max: float = 3.0e9
    service_memory_gb_min: float = 1.0
    service_memory_gb_max: float = 2.5
    satellite_memory_capacity_gb: float = 16.0
    replica_activation_delay_s: float = 0.05
    input_data_gb_min: float = 0.005
    input_data_gb_max: float = 0.030
    data_shrink_min: float = 0.65
    data_shrink_max: float = 0.95

    delay_weight: float = 0.5
    energy_weight: float = 0.5
    latency_scale_s: float = 10.0
    energy_scale_j: float = 100.0
    failure_penalty: float = 10.0
    max_episode_steps: int = 32

    connector_edge_weight: str = "hop"
    connector_repair_on_slot_change: bool = True
    max_route_hops: int = 64
    speed_of_light_km_s: float = 299_792.458
    route_horizon_slots: int = 3
    route_max_paths_per_slot: int = 3
    route_switch_delay_s: float = 0.001
    route_failure_risk_weight: float = 0.0

    adaptation_enabled: bool = True
    deployment_window_requests: int = 20
    adaptation_top_k_services: int = 10
    bandit_exploration: float = 1.25
    pressure_ewma: float = 0.5

    hidden_dim: int = 128
    graph_layers: int = 2
    attention_heads: int = 4
    service_embedding_dim: int = 32
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_epsilon: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_epochs: int = 4
    rollout_steps: int = 128
    max_grad_norm: float = 0.5

    output_dir: Path = PROJECT_ROOT / "ELARA" / "outputs"

    @property
    def total_satellites(self) -> int:
        return self.num_planes * self.sats_per_plane

    def to_dict(self) -> dict:
        result = asdict(self)
        result["trace_csv"] = str(self.trace_csv)
        result["output_dir"] = str(self.output_dir)
        return result
