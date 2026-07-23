from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ELARAConfig:
    seed: int = 42
    num_planes: int = 10
    sats_per_plane: int = 18
    num_services: int = 30
    # ``replicas_per_service`` is a fixed-count compatibility override used by
    # small tests. Normal experiments draw an independent initial count for
    # every service and the adaptation layer changes it at runtime.
    replicas_per_service: int | None = None
    replica_count_range: tuple[int, int] = (5, 10)
    chain_length: int | None = None
    request_template_chain_lengths: tuple[int, ...] = (5, 10, 15)
    request_template_file: Path | None = None
    request_data_scale: float = 1.0

    trace_csv: Path = PROJECT_ROOT / "WalkerDeltaConstellationSimu" / "Walker_Delta_ISL_Simu.csv"
    max_trace_slots: int | None = 120
    slot_duration_s: float = 10.0
    future_topology_horizon: int = 3

    compute_capacity_choices_gflops: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    compute_capacity_scale: float = 1.0
    compute_capacity_gflops_min: float = 1.0
    compute_capacity_gflops_max: float = 4.0
    compute_power_by_capacity_w: dict[float, float] = field(
        default_factory=lambda: {1.0: 50.0, 2.0: 60.0, 3.0: 70.0, 4.0: 80.0}
    )
    compute_power_w_min: float = 50.0
    compute_power_w_max: float = 80.0
    service_cycles_min: float = 1.0e9
    service_cycles_max: float = 1.0e10
    service_memory_gb_min: float = 2.4
    service_memory_gb_max: float = 4.0
    satellite_memory_capacity_gb: float = 12.0
    replica_activation_delay_s_min: float = 0.2
    replica_activation_delay_s_max: float = 2.0
    input_data_gb_min: float = 0.5
    input_data_gb_max: float = 4.0
    request_data_mean_gb: float = 2.0
    request_data_variance_gb: float = 0.5
    request_arrival_lambda_per_template_per_slot: float = 0.35
    request_endpoint_near_hops: int = 2
    preserve_inter_request_reservations: bool = True

    compute_load_states: tuple[str, ...] = ("Idle", "Light", "Medium", "Heavy")
    compute_load_initial_distribution: dict[str, float] = field(
        default_factory=lambda: {"Idle": 0.35, "Light": 0.35, "Medium": 0.20, "Heavy": 0.10}
    )
    compute_load_transition_matrix: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "Idle": {"Idle": 0.70, "Light": 0.25, "Medium": 0.04, "Heavy": 0.01},
            "Light": {"Idle": 0.20, "Light": 0.55, "Medium": 0.20, "Heavy": 0.05},
            "Medium": {"Idle": 0.05, "Light": 0.25, "Medium": 0.50, "Heavy": 0.20},
            "Heavy": {"Idle": 0.02, "Light": 0.08, "Medium": 0.30, "Heavy": 0.60},
        }
    )
    compute_load_lambda_per_slot: dict[str, float] = field(
        default_factory=lambda: {"Idle": 0.05, "Light": 0.20, "Medium": 0.65, "Heavy": 1.40}
    )
    compute_load_utilization_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "Idle": (0.00, 0.20), "Light": (0.20, 0.45),
            "Medium": (0.45, 0.70), "Heavy": (0.70, 0.95),
        }
    )
    compute_load_discount_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "Idle": (0.80, 1.00), "Light": (0.60, 0.80),
            "Medium": (0.40, 0.60), "Heavy": (0.20, 0.40),
        }
    )
    background_compute_cycles_mean: float = 2.0e9
    background_compute_cycles_min: float = 1.0e8
    background_compute_rho_max: float = 0.95
    background_compute_queue_base_s: float = 0.05

    link_load_states: tuple[str, ...] = ("Idle", "Light", "Medium", "Heavy")
    link_load_initial_distribution: dict[str, float] = field(
        default_factory=lambda: {"Idle": 0.40, "Light": 0.35, "Medium": 0.20, "Heavy": 0.05}
    )
    link_load_transition_matrix: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "Idle": {"Idle": 0.72, "Light": 0.23, "Medium": 0.04, "Heavy": 0.01},
            "Light": {"Idle": 0.18, "Light": 0.57, "Medium": 0.21, "Heavy": 0.04},
            "Medium": {"Idle": 0.05, "Light": 0.22, "Medium": 0.53, "Heavy": 0.20},
            "Heavy": {"Idle": 0.02, "Light": 0.08, "Medium": 0.28, "Heavy": 0.62},
        }
    )
    background_link_lambda_per_slot_by_state: dict[str, float] = field(
        default_factory=lambda: {"Idle": 0.05, "Light": 0.20, "Medium": 0.55, "Heavy": 1.20}
    )
    background_link_data_mean_gb: float = 0.15
    background_link_data_min_gb: float = 0.01
    background_link_rho_max: float = 0.95
    background_link_eta_min: float = 0.35
    background_link_kappa: float = 1.0
    background_link_queue_base_s: float = 0.02
    background_epsilon: float = 1.0e-6
    link_capacity_scale: float = 1.0
    background_load_scale: float = 1.0

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
    adaptation_window_slots: int = 10
    deployment_window_requests: int | None = None
    adaptation_top_k_services: int = 10
    bandit_exploration: float = 1.25
    pressure_ewma: float = 0.5
    adaptation_trace_sample_limit: int = 32

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
    ppo_minibatch_size: int = 16
    max_grad_norm: float = 0.5

    output_dir: Path = PROJECT_ROOT / "ELARA" / "outputs"

    def __post_init__(self) -> None:
        low, high = self.replica_count_range
        if low < 1 or high < low:
            raise ValueError("replica_count_range must satisfy 1 <= min <= max")
        if not self.request_template_chain_lengths and self.chain_length is None:
            raise ValueError("at least one request template chain length is required")
        if self.request_arrival_lambda_per_template_per_slot <= 0.0:
            raise ValueError("request arrival lambda must be positive")
        if self.delay_weight < 0.0 or self.energy_weight < 0.0:
            raise ValueError("delay and energy weights must be nonnegative")
        if abs(self.delay_weight + self.energy_weight - 1.0) > 1.0e-9:
            raise ValueError("delay and energy weights must sum to one")
        for name, value in (
            ("request_data_scale", self.request_data_scale),
            ("compute_capacity_scale", self.compute_capacity_scale),
            ("link_capacity_scale", self.link_capacity_scale),
            ("background_load_scale", self.background_load_scale),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.ppo_minibatch_size < 1:
            raise ValueError("ppo_minibatch_size must be at least 1")

    @property
    def total_satellites(self) -> int:
        return self.num_planes * self.sats_per_plane

    def to_dict(self) -> dict:
        result = asdict(self)
        result["trace_csv"] = str(self.trace_csv)
        result["request_template_file"] = (
            str(self.request_template_file) if self.request_template_file else None
        )
        result["output_dir"] = str(self.output_dir)
        return result
