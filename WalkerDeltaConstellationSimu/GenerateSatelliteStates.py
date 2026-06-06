import csv
import math
import os
import random
from datetime import datetime, timedelta

import numpy as np
from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *

# ==========================================
# 1. Scenario and constellation parameters
# ==========================================
NUM_PLANES = 10
SATS_PER_PLANE = 18
TOTAL_SATS = NUM_PLANES * SATS_PER_PLANE
ALTITUDE_KM = 800.0
INCLINATION_DEG = 80.0
PHASE_F = 1
EARTH_RADIUS_KM = 6371.0
EARTH_GRAVITATIONAL_PARAMETER_KM3_S2 = 398600.4418

STEP_SEC = 10.0
RANDOM_SEED = 42
TX_POWER_RANGE_W = (0.0316, 5.0)

LASER_CARRIER_FREQUENCY_HZ = 193.0e12
TX_FREQUENCY_GHZ = LASER_CARRIER_FREQUENCY_HZ / 1.0e9
STK_MAX_DATA_RATE_BPS = 1.0e12
STK_TX_DATA_RATE_MBPS = STK_MAX_DATA_RATE_BPS / 1.0e6

SATELLITE_STATE_CSV = "Walker_Delta_Satellite_States.csv"


def format_stk_utc(dt):
    month_names = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return (
        f"{dt.day} {month_names[dt.month]} {dt.year} "
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{dt.microsecond // 1000:03d}"
    )


ORBITAL_RADIUS_KM = EARTH_RADIUS_KM + ALTITUDE_KM
CONSTELLATION_PERIOD_SEC = 2.0 * math.pi * math.sqrt(
    ORBITAL_RADIUS_KM ** 3 / EARTH_GRAVITATIONAL_PARAMETER_KM3_S2
)
START_DATETIME_UTC = datetime(2026, 1, 1, 0, 0, 0)
STOP_DATETIME_UTC = START_DATETIME_UTC + timedelta(seconds=CONSTELLATION_PERIOD_SEC)
START_TIME = format_stk_utc(START_DATETIME_UTC)
STOP_TIME = format_stk_utc(STOP_DATETIME_UTC)

STATE_HEADER = [
    "Time_EpSec",
    "Time_UTC",
    "Satellite",
    "Plane",
    "Sat_Index",
    "Tx_Power_W",
    "Lat_Deg",
    "Lon_Deg",
    "Alt_Km",
    "X_ICRF_Km",
    "Y_ICRF_Km",
    "Z_ICRF_Km",
    "Vx_ICRF_Km_s",
    "Vy_ICRF_Km_s",
    "Vz_ICRF_Km_s",
]


def get_dataset_value(data_sets, *names):
    for name in names:
        try:
            return data_sets.GetDataSetByName(name).GetValues()[0]
        except Exception:
            pass
    raise KeyError(f"Dataset not found: {', '.join(names)}")


def get_cartesian_vector(sat_obj, provider_name, time_ep_sec):
    result = (
        sat_obj.DataProviders.Item(provider_name)
        .Group.Item("ICRF")
        .ExecSingle(float(time_ep_sec))
    )
    return np.array(
        [
            float(get_dataset_value(result.DataSets, "x", "X")),
            float(get_dataset_value(result.DataSets, "y", "Y")),
            float(get_dataset_value(result.DataSets, "z", "Z")),
        ],
        dtype=float,
    )


def get_satellite_state(sat_obj, time_ep_sec):
    lla_result = (
        sat_obj.DataProviders.Item("LLA State")
        .Group.Item("Fixed")
        .ExecSingle(float(time_ep_sec))
    )
    lat_deg = float(get_dataset_value(lla_result.DataSets, "Lat", "Latitude"))
    lon_deg = float(get_dataset_value(lla_result.DataSets, "Lon", "Longitude"))
    alt_km = float(get_dataset_value(lla_result.DataSets, "Alt", "Altitude"))
    return {
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "alt_km": alt_km,
        "pos_icrf_km": get_cartesian_vector(sat_obj, "Cartesian Position", time_ep_sec),
        "vel_icrf_km_s": get_cartesian_vector(
            sat_obj, "Cartesian Velocity", time_ep_sec
        ),
    }


def get_simulation_stop_ep_sec(root):
    start_date = root.ConversionUtility.NewDate("UTCG", START_TIME)
    stop_date = root.ConversionUtility.NewDate("UTCG", STOP_TIME)
    scenario_duration = stop_date.Span(start_date)
    scenario_duration.ConvertToUnit("sec")
    return abs(float(scenario_duration.Value))


def main():
    output_path = os.path.abspath(SATELLITE_STATE_CSV)
    file = open(SATELLITE_STATE_CSV, mode="w", newline="", encoding="utf-8-sig")
    writer = csv.writer(file)
    writer.writerow(STATE_HEADER)
    file.flush()
    print(f"Satellite state log created: {output_path}")

    try:
        print("Starting STK Engine... This can take a few seconds.")
        rng = random.Random(RANDOM_SEED)
        stk = STKDesktop.StartApplication(visible=True, userControl=True)
        root = stk.Root
        root.NewScenario("Walker_Delta_Satellite_State_Simu")
        scenario = root.CurrentScenario
        scenario.SetTimePeriod(START_TIME, STOP_TIME)
        root.UnitPreferences.Item("DateFormat").SetCurrentUnit("EpSec")
        root.Rewind()

        print(f"Creating {TOTAL_SATS} satellites with TX/RX payloads...")
        sat_dict = {}
        for p in range(NUM_PLANES):
            raan = p * (360.0 / NUM_PLANES)
            for s in range(SATS_PER_PLANE):
                mean_anomaly = (
                        s * (360.0 / SATS_PER_PLANE)
                        + p * PHASE_F * (360.0 / TOTAL_SATS)
                )
                sat_name = f"SAT_{p}_{s}"
                sat = scenario.Children.New(AgESTKObjectType.eSatellite, sat_name)
                sat.SetPropagatorType(AgEVePropagatorType.ePropagatorJ4Perturbation)
                prop = sat.Propagator
                prop.InitialState.Representation.AssignClassical(
                    AgECoordinateSystem.eCoordinateSystemJ2000,
                    ALTITUDE_KM + EARTH_RADIUS_KM,
                    0.0,
                    INCLINATION_DEG,
                    0.0,
                    raan,
                    mean_anomaly,
                )
                prop.Propagate()

                tx = sat.Children.New(AgESTKObjectType.eTransmitter, "TX")
                tx.SetModel("Complex Transmitter Model")
                tx_model = tx.Model
                tx_model.Frequency = TX_FREQUENCY_GHZ
                tx_model.DataRate = STK_TX_DATA_RATE_MBPS
                tx_power_w = rng.uniform(*TX_POWER_RANGE_W)
                tx_model.Power = 10.0 * math.log10(tx_power_w)

                rx = sat.Children.New(AgESTKObjectType.eReceiver, "RX")
                rx.SetModel("Complex Receiver Model")
                rx_model = rx.Model
                rx_model.AutoTrackFrequency = True
                rx_link_margin = rx_model.LinkMargin
                rx_link_margin.Enable = True
                rx_link_margin.Type = AgELinkMarginType.eLinkMarginTypeEbOverNo
                rx_link_margin.Threshold = 6.0

                sat_dict[sat_name] = {
                    "plane": p,
                    "pos": s,
                    "sat_obj": sat,
                    "tx_power_w": tx_power_w,
                }

        start_ep = 0.0
        stop_ep = get_simulation_stop_ep_sec(root)
        print(f"Constellation orbital period: {CONSTELLATION_PERIOD_SEC:.3f} seconds")
        print(f"Scenario start: {START_TIME}")
        print(f"Scenario stop: {STOP_TIME}")
        print(f"Sampling satellite states every {STEP_SEC} seconds...")
        sample_times = list(np.arange(start_ep, stop_ep + 1.0e-6, STEP_SEC))
        if not sample_times or sample_times[-1] < stop_ep:
            sample_times.append(stop_ep)

        for raw_t_sec in sample_times:
            t_sec = float(raw_t_sec)
            current_utc = root.ConversionUtility.NewDate("EpSec", str(t_sec)).Format(
                "UTCG"
            )
            print(f"Sampling time: {current_utc}")
            for sat_name, info in sat_dict.items():
                state = get_satellite_state(info["sat_obj"], t_sec)
                pos = state["pos_icrf_km"]
                vel = state["vel_icrf_km_s"]
                writer.writerow(
                    [
                        t_sec,
                        current_utc,
                        sat_name,
                        info["plane"],
                        info["pos"],
                        round(info["tx_power_w"], 6),
                        round(state["lat_deg"], 6),
                        round(state["lon_deg"], 6),
                        round(state["alt_km"], 6),
                        round(float(pos[0]), 9),
                        round(float(pos[1]), 9),
                        round(float(pos[2]), 9),
                        round(float(vel[0]), 12),
                        round(float(vel[1]), 12),
                        round(float(vel[2]), 12),
                    ]
                )
            file.flush()
    finally:
        file.close()
        print(f"Satellite state log closed: {output_path}")


if __name__ == "__main__":
    main()
