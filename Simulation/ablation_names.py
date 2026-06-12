from __future__ import annotations

ABLATION_NAME_MAP = {
    "full": "ELARA",
    "elara": "ELARA",
    "ELARA": "ELARA",
    "no_bandit": "ELARA-NB",
    "no-bandit": "ELARA-NB",
    "ELARA-NB": "ELARA-NB",
    "elara-nb": "ELARA-NB",
    "shortest_hop_routing": "ELARA-SH",
    "shortest-hop-routing": "ELARA-SH",
    "ELARA-SH": "ELARA-SH",
    "elara-sh": "ELARA-SH",
    "nearest_replica": "ELARA-NR",
    "nearest-replica": "ELARA-NR",
    "ELARA-NR": "ELARA-NR",
    "elara-nr": "ELARA-NR",
    "service_pressure": "SP-Routing",
    "service-pressure": "SP-Routing",
    "SP-Routing": "SP-Routing",
    "sp-routing": "SP-Routing",
    "sc_nfv": "SC-NFV",
    "sc-nfv": "SC-NFV",
    "SC-NFV": "SC-NFV",
    "fairness_nfv_greedy": "Fair-NFV",
    "fairness-nfv-greedy": "Fair-NFV",
    "Fair-NFV": "Fair-NFV",
    "fair-nfv": "Fair-NFV",
}

OFFICIAL_ABLATIONS = (
    "ELARA",
    "ELARA-NB",
    "ELARA-NR",
    "ELARA-SH",
    "Fair-NFV",
    "SP-Routing",
    "SC-NFV",
)

ABLATION_LABELS = {name: name for name in OFFICIAL_ABLATIONS}

ABLATION_GROUP_ABLATIONS = (
    "ELARA",
    "ELARA-NB",
    "ELARA-NR",
    "ELARA-SH",
)

COMPARISON_GROUP_ABLATIONS = (
    "ELARA",
    "Fair-NFV",
    "SP-Routing",
    "SC-NFV",
)

PLOT_GROUPS = {
    "ablation_group": ABLATION_GROUP_ABLATIONS,
    "comparison_group": COMPARISON_GROUP_ABLATIONS,
}


def canonical_ablation_name(name: str) -> str:
    return ABLATION_NAME_MAP.get(str(name), str(name))


def canonical_ablation_names(value: str) -> list[str]:
    names = []
    seen = set()
    for item in value.replace(",", " ").split():
        name = canonical_ablation_name(item)
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def canonicalize_ablation_row(row: dict) -> dict:
    if "ablation" in row:
        row["ablation"] = canonical_ablation_name(row["ablation"])
    return row
