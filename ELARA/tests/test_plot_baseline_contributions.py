from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ELARA.plot_baseline_contributions import (
    ABLATION_METHODS,
    COMPARISON_METHODS,
    REQUIRED_FILES,
    improvement,
    mean_ci95,
    nan_moving_average,
    resolve_run_root,
)


class PlotBaselineContributionsTests(unittest.TestCase):
    def test_experiment_groups_are_separate_and_share_only_elara(self):
        self.assertEqual(ABLATION_METHODS[0], "ELARA")
        self.assertEqual(COMPARISON_METHODS[0], "ELARA")
        self.assertEqual(set(ABLATION_METHODS) & set(COMPARISON_METHODS), {"ELARA"})

    def test_resolve_run_root_selects_latest_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "20260101-000000"
            latest = root / "20260102-000000"
            incomplete = root / "20260103-000000"
            for run in (older, latest, incomplete):
                run.mkdir()
            for run in (older, latest):
                for filename in REQUIRED_FILES:
                    (run / filename).touch()
            (incomplete / REQUIRED_FILES[0]).touch()
            self.assertEqual(resolve_run_root(root), latest.resolve())
            self.assertEqual(resolve_run_root(older), older.resolve())

    def test_improvement_sign_and_ci(self):
        self.assertAlmostEqual(improvement(80.0, 100.0), 20.0)
        self.assertAlmostEqual(improvement(110.0, 100.0), -10.0)
        mean, ci = mean_ci95([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mean, 2.5)
        self.assertGreater(ci, 0.0)

    def test_nan_aware_moving_average_is_causal(self):
        values = np.asarray([1.0, np.nan, 5.0, 7.0])
        np.testing.assert_allclose(
            nan_moving_average(values, 3),
            [1.0, 1.0, 3.0, 6.0],
        )


if __name__ == "__main__":
    unittest.main()
