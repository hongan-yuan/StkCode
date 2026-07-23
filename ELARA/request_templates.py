from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from .domain import ServiceRequestTemplate


DEFAULT_CHAIN_PLAN = ((5, 8), (10, 4), (15, 2))
DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "data" / "request_templates_seed2026.json"


def parse_chain_plan(value: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in value.split(","):
        length_text, count_text = item.strip().split(":", 1)
        length, count = int(length_text), int(count_text)
        if length < 1 or count < 1:
            raise ValueError("chain lengths and template counts must be positive")
        result.append((length, count))
    if not result:
        raise ValueError("at least one chain-plan entry is required")
    return tuple(result)


def _sample_data_gb(
    rng: random.Random,
    mean_gb: float,
    variance_gb: float,
    minimum_gb: float,
    maximum_gb: float,
) -> float:
    return min(
        maximum_gb,
        max(minimum_gb, rng.gauss(mean_gb, math.sqrt(variance_gb))),
    )


def generate_templates(
    *,
    seed: int,
    num_services: int,
    chain_plan: tuple[tuple[int, int], ...] = DEFAULT_CHAIN_PLAN,
    data_mean_gb: float = 2.0,
    data_variance_gb: float = 0.5,
    data_min_gb: float = 0.5,
    data_max_gb: float = 4.0,
) -> tuple[ServiceRequestTemplate, ...]:
    if num_services < 1:
        raise ValueError("num_services must be positive")
    rng = random.Random(seed)
    service_ids = list(range(num_services))
    templates = []
    template_id = 1
    for chain_length, count in chain_plan:
        for _ in range(count):
            services = tuple(rng.choices(service_ids, k=chain_length))
            volumes = tuple(
                _sample_data_gb(
                    rng, data_mean_gb, data_variance_gb, data_min_gb, data_max_gb
                )
                for _ in range(chain_length + 1)
            )
            templates.append(ServiceRequestTemplate(template_id, services, volumes))
            template_id += 1
    return tuple(templates)


def _validate_templates(
    templates: tuple[ServiceRequestTemplate, ...], num_services: int | None = None
) -> tuple[ServiceRequestTemplate, ...]:
    if not templates:
        raise ValueError("request template file is empty")
    identifiers = set()
    for template in templates:
        if template.template_id in identifiers:
            raise ValueError(f"duplicate template_id: {template.template_id}")
        identifiers.add(template.template_id)
        if len(template.data_volumes_gb) != len(template.services) + 1:
            raise ValueError(
                f"template {template.template_id} must contain chain_length + 1 data volumes"
            )
        if num_services is not None and any(
            service < 0 or service >= num_services for service in template.services
        ):
            raise ValueError(
                f"template {template.template_id} contains a service outside "
                f"[0, {num_services - 1}]"
            )
    return templates


def save_templates(
    path: Path,
    templates: tuple[ServiceRequestTemplate, ...],
    *,
    metadata: dict | None = None,
) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = {
            "schema_version": 1,
            "metadata": metadata or {},
            "templates": [
                {
                    "template_id": template.template_id,
                    "chain_length": len(template.services),
                    "services": list(template.services),
                    "data_volumes_gb": list(template.data_volumes_gb),
                }
                for template in templates
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "template_id", "chain_length", "services", "data_volumes_gb"
                ),
            )
            writer.writeheader()
            for template in templates:
                writer.writerow(
                    {
                        "template_id": template.template_id,
                        "chain_length": len(template.services),
                        "services": json.dumps(template.services),
                        "data_volumes_gb": json.dumps(template.data_volumes_gb),
                    }
                )
    else:
        raise ValueError("request template path must end with .json or .csv")
    return path


def load_templates(
    path: Path,
    *,
    num_services: int | None = None,
    data_scale: float = 1.0,
) -> tuple[ServiceRequestTemplate, ...]:
    path = path.expanduser().resolve()
    if data_scale <= 0.0:
        raise ValueError("data_scale must be positive")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["templates"] if isinstance(payload, dict) else payload
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("request template path must end with .json or .csv")
    templates = tuple(
        ServiceRequestTemplate(
            int(row["template_id"]),
            tuple(int(value) for value in _list_value(row["services"])),
            tuple(float(value) * data_scale for value in _list_value(row["data_volumes_gb"])),
        )
        for row in rows
    )
    return _validate_templates(templates, num_services)


def _list_value(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate a fixed ELARA request-template catalog."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-services", type=int, default=30)
    parser.add_argument("--chain-plan", default="5:8,10:4,15:2")
    parser.add_argument("--data-mean-gb", type=float, default=2.0)
    parser.add_argument("--data-variance-gb", type=float, default=0.5)
    parser.add_argument("--data-min-gb", type=float, default=0.5)
    parser.add_argument("--data-max-gb", type=float, default=4.0)
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="also write a CSV with the same stem",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chain_plan = parse_chain_plan(args.chain_plan)
    templates = generate_templates(
        seed=args.seed,
        num_services=args.num_services,
        chain_plan=chain_plan,
        data_mean_gb=args.data_mean_gb,
        data_variance_gb=args.data_variance_gb,
        data_min_gb=args.data_min_gb,
        data_max_gb=args.data_max_gb,
    )
    metadata = {
        "seed": args.seed,
        "num_services": args.num_services,
        "chain_plan": chain_plan,
        "data_mean_gb": args.data_mean_gb,
        "data_variance_gb": args.data_variance_gb,
        "data_min_gb": args.data_min_gb,
        "data_max_gb": args.data_max_gb,
    }
    output = save_templates(args.output, templates, metadata=metadata)
    outputs = [output]
    if args.also_csv:
        csv_path = output.with_suffix(".csv")
        outputs.append(save_templates(csv_path, templates, metadata=metadata))
    print(f"generated {len(templates)} fixed request templates")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
