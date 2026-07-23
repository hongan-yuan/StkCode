from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ELARA.plot_ppo_curves import aggregate_series, discover_metrics, load_run, moving_average


class PlotPPOCurvesTests(unittest.TestCase):
    def test_loads_sparse_ppo_updates_and_derives_total_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "task-seed-7"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps({"seed": 7, "ppo_value_coef": 0.5, "ppo_entropy_coef": 0.01}),
                encoding="utf-8",
            )
            self._write_metrics(run_dir / "training_metrics.csv")
            run = load_run(run_dir / "training_metrics.csv")
        self.assertEqual(run.label, "seed 7")
        self.assertEqual(len(run.rewards), 3)
        self.assertEqual(len(run.total_loss), 2)
        self.assertAlmostEqual(run.total_loss[0], 0.2 + 0.5 * 2.0 - 0.01 * 0.7)

    def test_discovers_multiple_runs_and_aggregates_different_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1, 2):
                run_dir = root / f"task-seed-{seed}"
                run_dir.mkdir()
                self._write_metrics(run_dir / "training_metrics.csv")
            paths = discover_metrics(root)
            runs = [load_run(path) for path in paths]
            runs[1].rewards = runs[1].rewards[:2]
            x, mean, lower, upper = aggregate_series(runs, "rewards", 1)
        self.assertEqual(len(paths), 2)
        np.testing.assert_allclose(x, [0.0, 1.0])
        np.testing.assert_allclose(mean, [-1.0, -2.0])
        np.testing.assert_allclose(lower, upper)

    def test_causal_moving_average_preserves_length(self):
        values = moving_average(np.asarray([1.0, 3.0, 5.0, 7.0]), 3)
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0, 5.0])

    @staticmethod
    def _write_metrics(path: Path) -> None:
        rows = [
            {"episode": 0, "return": -1, "policy_loss": "", "value_loss": "", "entropy": ""},
            {"episode": 1, "return": -2, "policy_loss": 0.2, "value_loss": 2.0, "entropy": 0.7},
            {"episode": 2, "return": -3, "policy_loss": 0.1, "value_loss": 1.0, "entropy": 0.6},
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("episode", "return", "policy_loss", "value_loss", "entropy"),
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
