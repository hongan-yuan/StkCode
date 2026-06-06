from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int = 42

    num_planes: int = 10
    sats_per_plane: int = 18
    num_microservices: int = 30
    replica_count_range: tuple[int, int] = (5, 10)
    max_services_per_satellite: int = 3

    microservice_workload_range_cycles: tuple[float, float] = (1.0e9, 1.0e10)
    microservice_image_size_range_gb: tuple[float, float] = (0.2, 1.2)
    microservice_storage_range_gb: tuple[float, float] = (1.0, 4.0)

    cpu_freq_choices_ghz: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    cpu_power_by_freq_w: dict[float, float] = field(
        default_factory=lambda: {1.0: 50.0, 2.0: 60.0, 3.0: 70.0, 4.0: 80.0}
    )
    cpu_discount_range: tuple[float, float] = (0.7, 1.0)
    queue_zero_probability: float = 0.20
    queue_max_fraction_of_slot: float = 0.50

    satellite_storage_capacity_gb: float = 12.0
    default_tx_power_w: float = 1.0
    speed_of_light_m_per_s: float = 299_792_458.0

    isl_log_csv: Path = (
        ROOT_DIR / "WalkerDeltaConstellationSimu" / "Walker_Delta_ISL_Simu.csv"
    )
    max_slots_to_load: int | None = None
    slot_duration_override_s: float | None = 10.0

    request_chain_plan: tuple[tuple[int, int], ...] = ((8, 5), (4, 10), (2, 15))
    process_single_request: bool = True
    selected_request_id: int | None = None
    single_request_start_time: float = 0.0
    request_arrival_lambda_per_slot: float = 0.35
    request_arrival_lambda_per_pattern_per_slot: float = 0.35
    request_data_mean_gb: float = 2.0
    request_data_variance_gb: float = 0.5
    request_data_min_gb: float = 0.5
    source_dest_near_hops: int = 2
    request_endpoint_max_low_speed_score: float = 0.75
    request_endpoint_sample_top_k: int = 4
    request_endpoint_route_check_limit: int = 12

    migration_safety_margin: float = 0.05

    # Parameter Sensitivity
    route_horizon_slots: int = 3
    future_link_horizon_slots: int = 3
    min_cost_flow_max_augmentations_per_slot: int = 10
    switch_penalty_s: float = 0.02
    candidate_bottleneck_shortage_penalty_weight: float = 2.0
    candidate_egress_shortage_penalty_weight: float = 2.0
    low_speed_neighbor_rate_threshold_mbps: float = 1000.0
    low_speed_neighbor_penalty_weight: float = 0.5
    migration_failure_relief_bonus: float = 2.0

    background_link_lambda_per_slot: float = 0.35   # number of background requests
    background_link_data_mean_gb: float = 0.15      # data volume of each background requests   Exponential
    background_link_data_min_gb: float = 0.01       #
    background_link_rho_max: float = 0.95           # 背景链路利用率 rho 的上限
    background_link_eta_min: float = 0.35           # 链路有效速率折扣系数 eta 的下限，背景负载再重，也至少保留 35% 的额定链路速率
    background_link_kappa: float = 1.0
    background_link_queue_base_s: float = 0.02
    background_epsilon: float = 1.0e-6

    delay_weight: float = 0.4
    energy_weight: float = 0.6
    migration_weight: float = 0.15
    failure_penalty: float = 1_000_000.0
    slot_switch_penalty_weight: float = 1.0
    route_failure_risk_weight: float = 1.0

    ppo_hidden_dim: int = 128
    ppo_clip_epsilon: float = 0.2
    ppo_gamma: float = 0.99
    ppo_learning_rate: float = 3.0e-4

    output_dir: Path = ROOT_DIR / "Simulation" / "outputs"

    @property
    def total_sats(self) -> int:
        return self.num_planes * self.sats_per_plane
