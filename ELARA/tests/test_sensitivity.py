from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ELARA.config import ELARAConfig
from ELARA.plot_sensitivity import main as plot_main
from ELARA.request_templates import (
    DEFAULT_CHAIN_PLAN,
    generate_templates,
    load_templates,
    save_templates,
)
from ELARA.sensitivity_runner import build_jobs, experiment_specs, parse_args


class SensitivityTests(unittest.TestCase):
    def test_fixed_catalog_matches_three_chain_lengths(self):
        templates = generate_templates(
            seed=2026,
            num_services=30,
            chain_plan=DEFAULT_CHAIN_PLAN,
        )
        counts = Counter(len(template.services) for template in templates)
        self.assertEqual(counts, {5: 8, 10: 4, 15: 2})
        self.assertEqual(templates, generate_templates(
            seed=2026,
            num_services=30,
            chain_plan=DEFAULT_CHAIN_PLAN,
        ))

    def test_json_and_csv_round_trip_and_scale(self):
        templates = generate_templates(
            seed=3,
            num_services=4,
            chain_plan=((2, 2),),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extension in ("json", "csv"):
                path = root / f"templates.{extension}"
                save_templates(path, templates)
                loaded = load_templates(path, num_services=4, data_scale=0.5)
                self.assertEqual(loaded[0].services, templates[0].services)
                self.assertAlmostEqual(
                    loaded[0].data_volumes_gb[0],
                    templates[0].data_volumes_gb[0] * 0.5,
                )

    def test_default_design_has_two_separate_three_by_four_experiments(self):
        args = parse_args(
            [
                "--output-root", "ELARA/outputs/sensitivity/unit-test",
                "--request-template-file", "ELARA/data/request_templates_seed2026.json",
            ]
        )
        specs = experiment_specs(args)
        self.assertEqual(args.train_tasks, 2)
        self.assertEqual(args.test_tasks, 4)
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            Counter(spec["category"] for spec in specs),
            {"latency_energy_weights": 3, "routing_max_paths": 3},
        )
        jobs = build_jobs(args, "train", "cpu", [])
        self.assertEqual(len(jobs), 24)
        self.assertTrue(all("--request-template-file" in job["command"] for job in jobs))
        self.assertTrue(
            all("--ppo-update-interval-slots" in job["command"] for job in jobs)
        )
        self.assertTrue(all(job["expected_total"] == 1212 for job in jobs))

    def test_config_rejects_invalid_objective_weights(self):
        with self.assertRaises(ValueError):
            ELARAConfig(delay_weight=0.7, energy_weight=0.4)

    def test_plotter_keeps_weight_and_routing_outputs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "sensitivity_summary.csv"
            rows = []
            for category, conditions in (
                ("latency_energy_weights", (("d50_e50", 0.5, 0.5, 3), ("d35_e65", 0.35, 0.65, 3), ("d65_e35", 0.65, 0.35, 3))),
                ("routing_max_paths", (("paths_3", 0.5, 0.5, 3), ("paths_5", 0.5, 0.5, 5), ("paths_7", 0.5, 0.5, 7))),
            ):
                for condition, delay, energy, paths in conditions:
                    for seed in (42, 43, 44, 45):
                        rows.append(
                            {
                                "category": category,
                                "condition": condition,
                                "seed": seed,
                                "delay_weight": delay,
                                "energy_weight": energy,
                                "route_max_paths": paths,
                                "request_count": 10,
                                "success_rate": 1.0,
                                "mean_return": -1.0,
                                "mean_latency_s": 10.0 + paths,
                                "mean_energy_j": 100.0 + paths,
                                "mean_route_slot_crossings": 0.1,
                                "mean_route_phase_count": 1.0,
                                "mean_route_augmentation_count": float(paths),
                            }
                        )
            with summary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "plots"
            self.assertEqual(
                plot_main([str(summary), "--output-dir", str(output), "--formats", "png"]),
                0,
            )
            self.assertTrue(
                (output / "latency_energy_weights" / "performance_sensitivity.png").is_file()
            )
            self.assertTrue(
                (output / "routing_max_paths" / "routing_overhead.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
