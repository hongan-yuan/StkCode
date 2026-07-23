from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ELARA.baseline_test_runner import BASELINES, _job_specs, parse_args


class BaselineTestRunnerTests(unittest.TestCase):
    def test_defaults_cover_all_requested_baselines(self):
        args, forwarded = parse_args(["--dry-run"])
        self.assertEqual(tuple(args.baselines), BASELINES)
        self.assertEqual(args.seeds, [42, 43, 44, 45])
        self.assertEqual(args.tasks, 4)
        self.assertEqual(forwarded, [])

    def test_builds_one_job_per_baseline_and_seed_with_round_robin_gpus(self):
        with tempfile.TemporaryDirectory(dir="ELARA") as directory:
            args, forwarded = parse_args(
                [
                    "--baselines",
                    "ELARA,SECO,SC-NFV",
                    "--seeds",
                    "7,8",
                    "--output-root",
                    directory,
                    "--no-load-checkpoint",
                ]
            )
            jobs = _job_specs(args, forwarded, "cuda", ["2", "5"])
        self.assertEqual(len(jobs), 6)
        self.assertEqual([job["gpu"] for job in jobs], ["2", "5", "2", "5", "2", "5"])
        self.assertTrue(all("--skip-aggregate" in job["command"] for job in jobs))
        self.assertTrue(all("ELARA" in Path(job["output_dir"]).parts for job in jobs))

    def test_rejects_output_outside_elara(self):
        with self.assertRaises(SystemExit):
            parse_args(["--output-root", "/tmp/elara-baseline-output"])


if __name__ == "__main__":
    unittest.main()
