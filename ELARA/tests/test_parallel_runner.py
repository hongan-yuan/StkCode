from __future__ import annotations

import unittest
from unittest.mock import patch

from ELARA.parallel_runner import (
    detect_gpu_ids,
    gpu_for_task,
    parse_args,
    select_accelerator,
)


class ParallelRunnerTests(unittest.TestCase):
    def test_tasks_are_distributed_round_robin(self):
        gpu_ids = ["0", "1"]
        self.assertEqual(
            [gpu_for_task(index, gpu_ids) for index in range(5)],
            ["0", "1", "0", "1", "0"],
        )

    def test_no_gpu_falls_back_to_cpu(self):
        self.assertIsNone(gpu_for_task(0, []))

    def test_visible_devices_are_respected(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,5"}, clear=False):
            self.assertEqual(detect_gpu_ids(), ["2", "5"])

    def test_default_parallelism_is_four(self):
        args, forwarded = parse_args(["train", "--episodes", "2"])
        self.assertEqual(args.tasks, 4)
        self.assertEqual(forwarded, ["--episodes", "2"])

    def test_auto_uses_mps_when_cuda_is_unavailable(self):
        self.assertEqual(select_accelerator("auto", [], True), "mps")
        self.assertEqual(select_accelerator("auto", [], False), "cpu")

    def test_explicit_unavailable_accelerator_is_rejected(self):
        with self.assertRaises(RuntimeError):
            select_accelerator("cuda", [], False)
        with self.assertRaises(RuntimeError):
            select_accelerator("mps", [], False)


if __name__ == "__main__":
    unittest.main()
