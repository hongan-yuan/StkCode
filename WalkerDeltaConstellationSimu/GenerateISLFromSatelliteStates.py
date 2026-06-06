import csv
import math
import os

import numpy as np


# ==========================================
# 1. Constellation and file parameters
# ==========================================
NUM_PLANES = 10
SATS_PER_PLANE = 18
TOTAL_SATS = NUM_PLANES * SATS_PER_PLANE
ALTITUDE_KM = 800.0
EARTH_RADIUS_KM = 6371.0

SATELLITE_STATE_CSV = "Walker_Delta_Satellite_States.csv"
ISL_OUTPUT_CSV = "Walker_Delta_ISL_Simu.csv"


# ==========================================
# 2. ISL communication model parameters
# ==========================================
LASER_CARRIER_FREQUENCY_HZ = 193.0e12
SPEED_OF_LIGHT_M_PER_SEC = 299_792_458.0
LASER_WAVELENGTH_M = SPEED_OF_LIGHT_M_PER_SEC / LASER_CARRIER_FREQUENCY_HZ
CHANNEL_BANDWIDTH_HZ = 0.02 * LASER_CARRIER_FREQUENCY_HZ
SYSTEM_OPTICAL_EFFICIENCY = 0.8
RECEIVING_TELESCOPE_DIAMETER_M = 6.0e-3
POINTING_ERROR_RAD = 0.01
THREE_DB_BEAMWIDTH_RAD = 0.1
FULL_TRANSMIT_DIVERGENCE_ANGLE_RAD = 1.0e-4
BOLTZMANN_CONSTANT_J_PER_K = 1.38e-23
SOLAR_BRIGHTNESS_TEMP_K = 6000.0
SYSTEM_NOISE_TEMP_K = 1000.0
CMB_TEMP_K = 2.725
SNR_THRESHOLD_DB = -110.0
SNR_THRESHOLD_LINEAR = 10.0 ** (SNR_THRESHOLD_DB / 10.0)

MAX_TRACKING_ANGLE_DEG = 60.0
POLAR_LATITUDE_BOUND_DEG = 70.0
APPLY_POLAR_LIMIT_TO_INTRA_PLANE = False
ENABLE_POLAR_EXCLUSION = True
DOPPLER_COMPENSATION_RATIO = 1.0e-3
MAX_RESIDUAL_DOPPLER_SHIFT_HZ = 10.0e6

MAX_LOS_DISTANCE_KM = 2.0 * math.sqrt(
    ALTITUDE_KM * (ALTITUDE_KM + 2.0 * EARTH_RADIUS_KM)
)

ISL_HEADER = [
    "Time_EpSec",
    "Time_UTC",
    "Endpoint_A",
    "Endpoint_B",
    "Link_ID",
    "Link_Type",
    "Status",
    "Failure_Reason",
    "Distance_km",
    "Max_LoS_Distance_km",
    "Tracking_Angle_Deg",
    "Link_Tx_Power_W",
    "Pointing_Loss",
    "Noise_Power_W",
    "SNR_Threshold_Linear",
    "Received_Power_W",
    "SNR_Linear",
    "Capacity_bps",
    "Effective_DataRate_Mbps",
    "Endpoint_A_Lat_Deg",
    "Endpoint_B_Lat_Deg",
    "Endpoint_A_Lon_Deg",
    "Endpoint_B_Lon_Deg",
    "Motion_Angle_Deg",
    "Relative_Radial_Velocity_km_s",
    "Doppler_Shift_Hz",
    "Residual_Doppler_Shift_Hz",
    "Doppler_Compensation_Ratio",
]


def make_link_id(endpoint_a, endpoint_b):
    left, right = sorted([endpoint_a, endpoint_b])
    return f"{left}<->{right}"


def estimate_distance_from_states(state_a, state_b):
    return float(np.linalg.norm(state_b["pos_icrf_km"] - state_a["pos_icrf_km"]))


def parse_state_row(row):
    return {
        "time_epsec": float(row["Time_EpSec"]),
        "time_utc": row["Time_UTC"],
        "satellite": row["Satellite"],
        "plane": int(row["Plane"]),
        "pos": int(row["Sat_Index"]),
        "tx_power_w": float(row["Tx_Power_W"]),
        "lat_deg": float(row["Lat_Deg"]),
        "lon_deg": float(row["Lon_Deg"]),
        "alt_km": float(row["Alt_Km"]),
        "pos_icrf_km": np.array(
            [
                float(row["X_ICRF_Km"]),
                float(row["Y_ICRF_Km"]),
                float(row["Z_ICRF_Km"]),
            ],
            dtype=float,
        ),
        "vel_icrf_km_s": np.array(
            [
                float(row["Vx_ICRF_Km_s"]),
                float(row["Vy_ICRF_Km_s"]),
                float(row["Vz_ICRF_Km_s"]),
            ],
            dtype=float,
        ),
    }


def load_satellite_states(path):
    states_by_time = {}
    with open(path, mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = parse_state_row(row)
            t_sec = state["time_epsec"]
            states_by_time.setdefault(t_sec, {})[state["satellite"]] = state
    return dict(sorted(states_by_time.items(), key=lambda item: item[0]))


def get_snapshot_undirected_isl_edges(state_cache):
    edges = []
    for p in range(NUM_PLANES):
        for s in range(SATS_PER_PLANE):
            source = f"SAT_{p}_{s}"
            intra_target = f"SAT_{p}_{(s + 1) % SATS_PER_PLANE}"
            edges.append(
                {
                    "a": source,
                    "b": intra_target,
                    "type": "IntraPlane",
                    "id": make_link_id(source, intra_target),
                }
            )

    for p in range(NUM_PLANES):
        adjacent_plane = (p + 1) % NUM_PLANES
        for s in range(SATS_PER_PLANE):
            source = f"SAT_{p}_{s}"
            nearest_target = None
            nearest_distance_km = math.inf
            for target_s in range(SATS_PER_PLANE):
                candidate = f"SAT_{adjacent_plane}_{target_s}"
                candidate_distance_km = estimate_distance_from_states(
                    state_cache[source], state_cache[candidate]
                )
                if candidate_distance_km < nearest_distance_km:
                    nearest_target = candidate
                    nearest_distance_km = candidate_distance_km

            edges.append(
                {
                    "a": source,
                    "b": nearest_target,
                    "type": "CrossPlane",
                    "id": make_link_id(source, nearest_target),
                }
            )

    unique_edges = {}
    for edge in edges:
        unique_edges[edge["id"]] = edge
    return list(unique_edges.values())


def compute_geometry_and_doppler(state_a, state_b):
    pos_a = state_a["pos_icrf_km"]
    pos_b = state_b["pos_icrf_km"]
    vel_a = state_a["vel_icrf_km_s"]
    vel_b = state_b["vel_icrf_km_s"]

    relative_pos = pos_b - pos_a
    distance_km = float(np.linalg.norm(relative_pos))
    if distance_km <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    los_unit = relative_pos / distance_km
    relative_vel = vel_b - vel_a
    radial_velocity_km_s = float(np.dot(relative_vel, los_unit))
    doppler_hz = abs(
        LASER_CARRIER_FREQUENCY_HZ
        * radial_velocity_km_s
        * 1000.0
        / SPEED_OF_LIGHT_M_PER_SEC
    )

    norm_a = float(np.linalg.norm(pos_a))
    norm_b = float(np.linalg.norm(pos_b))
    cos_angle = float(np.dot(pos_a, pos_b) / (norm_a * norm_b))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    tracking_angle_deg = math.degrees(math.acos(cos_angle))

    speed_a = float(np.linalg.norm(vel_a))
    speed_b = float(np.linalg.norm(vel_b))
    if speed_a <= 0.0 or speed_b <= 0.0:
        motion_angle_deg = 0.0
    else:
        cos_motion_angle = float(np.dot(vel_a, vel_b) / (speed_a * speed_b))
        cos_motion_angle = min(1.0, max(-1.0, cos_motion_angle))
        motion_angle_deg = math.degrees(math.acos(cos_motion_angle))

    return (
        distance_km,
        tracking_angle_deg,
        motion_angle_deg,
        radial_velocity_km_s,
        doppler_hz,
    )


def compute_pointing_loss():
    g0 = 4.0 * math.log(2.0) / (THREE_DB_BEAMWIDTH_RAD**2)
    return math.exp(-g0 * (POINTING_ERROR_RAD**2))


def compute_noise_power_w():
    return (
        BOLTZMANN_CONSTANT_J_PER_K
        * CHANNEL_BANDWIDTH_HZ
        * (SOLAR_BRIGHTNESS_TEMP_K + SYSTEM_NOISE_TEMP_K + CMB_TEMP_K)
    )


def compute_fso_capacity(distance_km, transmit_power_w):
    distance_m = max(distance_km * 1000.0, 1.0)
    transmitter_gain = 16.0 / (FULL_TRANSMIT_DIVERGENCE_ANGLE_RAD**2)
    receiver_gain = (
        math.pi * RECEIVING_TELESCOPE_DIAMETER_M / LASER_WAVELENGTH_M
    ) ** 2
    pointing_loss = compute_pointing_loss()
    free_space_path_loss = (LASER_WAVELENGTH_M / (4.0 * math.pi * distance_m)) ** 2
    received_power_w = (
        transmit_power_w
        * SYSTEM_OPTICAL_EFFICIENCY
        * transmitter_gain
        * receiver_gain
        * pointing_loss
        * free_space_path_loss
    )
    noise_power_w = compute_noise_power_w()
    snr_linear = received_power_w / noise_power_w
    capacity_bps = CHANNEL_BANDWIDTH_HZ * math.log2(1.0 + snr_linear)
    return received_power_w, snr_linear, capacity_bps, pointing_loss, noise_power_w


def apply_isl_connectivity_constraints(
    link_type,
    state_a,
    state_b,
    distance_km,
    tracking_angle_deg,
    snr_linear,
    radial_velocity_km_s,
    residual_doppler_hz,
):
    reasons = []
    if distance_km > MAX_LOS_DISTANCE_KM:
        if link_type == "CrossPlane":
            reasons.append("CommunicationRange")
        else:
            reasons.append("LoS_Occlusion")

    if link_type == "CrossPlane" and tracking_angle_deg > MAX_TRACKING_ANGLE_DEG:
        reasons.append("TrackingAngleLimit")

    polar_limit_applies = (
        ENABLE_POLAR_EXCLUSION
        and (link_type == "CrossPlane" or APPLY_POLAR_LIMIT_TO_INTRA_PLANE)
    )
    if polar_limit_applies and (
        max(abs(state_a["lat_deg"]), abs(state_b["lat_deg"]))
        > POLAR_LATITUDE_BOUND_DEG
    ):
        reasons.append("PolarExclusion")

    if snr_linear < SNR_THRESHOLD_LINEAR:
        reasons.append("SNRThreshold")

    if residual_doppler_hz > MAX_RESIDUAL_DOPPLER_SHIFT_HZ:
        reasons.append("DopplerLimit")

    return reasons


def validate_state_snapshot(t_sec, state_cache):
    missing = []
    for p in range(NUM_PLANES):
        for s in range(SATS_PER_PLANE):
            sat_name = f"SAT_{p}_{s}"
            if sat_name not in state_cache:
                missing.append(sat_name)
    if missing:
        raise ValueError(
            f"Time {t_sec} is missing {len(missing)} satellite states, "
            f"examples: {missing[:5]}"
        )


def main():
    state_path = os.path.abspath(SATELLITE_STATE_CSV)
    output_path = os.path.abspath(ISL_OUTPUT_CSV)
    print(f"Reading satellite states: {state_path}")
    states_by_time = load_satellite_states(SATELLITE_STATE_CSV)
    if not states_by_time:
        raise ValueError(f"No satellite states found in {state_path}")

    with open(ISL_OUTPUT_CSV, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(ISL_HEADER)
        file.flush()
        print(f"ISL output log created: {output_path}")

        for t_sec, state_cache in states_by_time.items():
            validate_state_snapshot(t_sec, state_cache)
            current_utc = next(iter(state_cache.values()))["time_utc"]
            print(f"Computing ISLs at time: {current_utc}")
            undirected_edges = get_snapshot_undirected_isl_edges(state_cache)

            for edge in undirected_edges:
                endpoint_a = edge["a"]
                endpoint_b = edge["b"]
                state_a = state_cache[endpoint_a]
                state_b = state_cache[endpoint_b]
                (
                    distance_km,
                    tracking_angle_deg,
                    motion_angle_deg,
                    radial_velocity_km_s,
                    doppler_hz,
                ) = compute_geometry_and_doppler(state_a, state_b)
                residual_doppler_hz = doppler_hz * DOPPLER_COMPENSATION_RATIO
                transmit_power_w = min(
                    state_a["tx_power_w"],
                    state_b["tx_power_w"],
                )
                (
                    received_power_w,
                    snr_linear,
                    capacity_bps,
                    pointing_loss,
                    noise_power_w,
                ) = compute_fso_capacity(distance_km, transmit_power_w)

                failure_reasons = apply_isl_connectivity_constraints(
                    edge["type"],
                    state_a,
                    state_b,
                    distance_km,
                    tracking_angle_deg,
                    snr_linear,
                    radial_velocity_km_s,
                    residual_doppler_hz,
                )

                if failure_reasons:
                    status = "Dead"
                    effective_data_rate_mbps = 0.0
                else:
                    status = "Alive"
                    effective_data_rate_mbps = capacity_bps / 1.0e6

                row = [
                    t_sec,
                    current_utc,
                    endpoint_a,
                    endpoint_b,
                    edge["id"],
                    edge["type"],
                    status,
                    "|".join(failure_reasons) if failure_reasons else "None",
                    round(distance_km, 2),
                    round(MAX_LOS_DISTANCE_KM, 2),
                    round(tracking_angle_deg, 6),
                    round(transmit_power_w, 6),
                    f"{pointing_loss:.12e}",
                    f"{noise_power_w:.12e}",
                    f"{SNR_THRESHOLD_LINEAR:.12e}",
                    f"{received_power_w:.12e}",
                    f"{snr_linear:.12e}",
                    round(capacity_bps, 3),
                    round(effective_data_rate_mbps, 6),
                    round(state_a["lat_deg"], 6),
                    round(state_b["lat_deg"], 6),
                    round(state_a["lon_deg"], 6),
                    round(state_b["lon_deg"], 6),
                    round(motion_angle_deg, 6),
                    round(radial_velocity_km_s, 6),
                    round(doppler_hz, 3),
                    round(residual_doppler_hz, 3),
                    DOPPLER_COMPENSATION_RATIO,
                ]
                writer.writerow(row)
            file.flush()
    print(f"ISL output log closed: {output_path}")


if __name__ == "__main__":
    main()
