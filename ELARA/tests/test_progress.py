from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ELARA.parallel_runner import _forwarded_int, format_duration, progress_line
from ELARA.progress import ProgressReporter


class ProgressTests(unittest.TestCase):
    def test_reporter_writes_progress_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            reporter = ProgressReporter(path, total=10)
            reporter.update(4)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["completed"], 4)
            self.assertEqual(payload["total"], 10)
            self.assertAlmostEqual(payload["fraction"], 0.4)
            self.assertIsNotNone(payload["eta_s"])

    def test_aggregate_progress_bar_and_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = []
            for index, eta in enumerate((6.0, 9.0)):
                path = Path(directory) / f"{index}.json"
                path.write_text(
                    json.dumps(
                        {"completed": 5, "total": 10, "elapsed_s": 5.0, "eta_s": eta}
                    ),
                    encoding="utf-8",
                )
                jobs.append({"progress_file": str(path)})
            line = progress_line(jobs, width=10)
            self.assertIn("50.00%", line)
            self.assertIn("10/20", line)
            self.assertIn("ETA 00:00:09", line)

    def test_duration_format(self):
        self.assertEqual(format_duration(3661), "01:01:01")
        self.assertEqual(format_duration(None), "--:--:--")

    def test_forwarded_episode_count(self):
        self.assertEqual(_forwarded_int(["--episodes", "25"], "--episodes", 100), 25)
        self.assertEqual(_forwarded_int(["--episodes=30"], "--episodes", 100), 30)


if __name__ == "__main__":
    unittest.main()
