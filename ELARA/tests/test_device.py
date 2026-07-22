from __future__ import annotations

import unittest
from types import SimpleNamespace

from ELARA.device import resolve_torch_device_name


class _Availability:
    def __init__(self, available: bool, built: bool = True):
        self.available = available
        self.built = built

    def is_available(self):
        return self.available

    def is_built(self):
        return self.built


def fake_torch(cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=_Availability(cuda),
        backends=SimpleNamespace(mps=_Availability(mps)),
    )


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(resolve_torch_device_name("auto", fake_torch(True, True)), "cuda")
        self.assertEqual(resolve_torch_device_name("auto", fake_torch(False, True)), "mps")
        self.assertEqual(resolve_torch_device_name("auto", fake_torch(False, False)), "cpu")

    def test_explicit_mps_is_supported_and_validated(self):
        self.assertEqual(resolve_torch_device_name("mps", fake_torch(False, True)), "mps")
        with self.assertRaises(RuntimeError):
            resolve_torch_device_name("mps", fake_torch(False, False))


if __name__ == "__main__":
    unittest.main()
