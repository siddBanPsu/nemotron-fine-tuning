from __future__ import annotations

import sys
import unittest
from unittest import mock

from nemotron_ft_lab.hardware import (
    HardwareInventory,
    inspect_cuda,
    validate_full_sft_hardware,
    validate_inference_hardware,
    validate_peft_hardware,
)


class HardwareTests(unittest.TestCase):
    def test_missing_torch_points_to_the_container_jupyter(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "Secure Link on host port 8889"):
                inspect_cuda()

    def test_profile_specific_memory_gates(self):
        small = HardwareInventory(1, ("small GPU",), (12.0,), 12.0)
        with self.assertRaisesRegex(RuntimeError, "20 GiB"):
            validate_inference_hardware(small, "nano9b_workshop")
        with self.assertRaisesRegex(RuntimeError, "75 GiB"):
            validate_inference_hardware(small, "lightning35_advanced")
        with self.assertRaisesRegex(RuntimeError, "45 GiB"):
            validate_peft_hardware(HardwareInventory(1, ("L4",), (24.0,), 24.0), "nano9b_workshop")

    def test_full_sft_is_gated_to_verified_16_h100_reference(self):
        eight_h100 = HardwareInventory(8, ("H100",) * 8, (80.0,) * 8, 640.0)
        with self.assertRaisesRegex(RuntimeError, "16x H100"):
            validate_full_sft_hardware(eight_h100)


if __name__ == "__main__":
    unittest.main()
