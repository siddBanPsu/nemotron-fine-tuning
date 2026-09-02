from __future__ import annotations

import unittest

from nemotron_ft_lab.megatron_compat import (
    EXPERT_BIAS_PADDING_MASK_MARKER,
    configure_expert_bias_padding_mask_compatibility,
)


class NativeMaskRouter:
    def _apply_expert_bias(self, routing_map, padding_mask=None):
        if padding_mask is not None:
            routing_map = routing_map & (~padding_mask).unsqueeze(-1)
        return routing_map


class LegacyMaskRouter:
    def _apply_expert_bias(self, routing_map, padding_mask=None):
        if padding_mask is not None:
            routing_map = routing_map & (~padding_mask)
        return routing_map


class UnknownMaskRouter:
    def _apply_expert_bias(self, routing_map, padding_mask=None):
        return routing_map


class MegatronCompatibilityTests(unittest.TestCase):
    def test_marks_native_mcore_fix_to_skip_bridge_wrapper(self):
        mode = configure_expert_bias_padding_mask_compatibility(NativeMaskRouter)

        self.assertEqual(mode, "native-mcore")
        self.assertTrue(getattr(NativeMaskRouter._apply_expert_bias, EXPERT_BIAS_PADDING_MASK_MARKER))
        self.assertEqual(
            configure_expert_bias_padding_mask_compatibility(NativeMaskRouter),
            "already-compatible",
        )

    def test_leaves_legacy_mcore_for_bridge_wrapper(self):
        self.assertEqual(
            configure_expert_bias_padding_mask_compatibility(LegacyMaskRouter),
            "bridge-wrapper",
        )
        self.assertFalse(getattr(LegacyMaskRouter._apply_expert_bias, EXPERT_BIAS_PADDING_MASK_MARKER, False))

    def test_rejects_an_unknown_router_implementation(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot determine"):
            configure_expert_bias_padding_mask_compatibility(UnknownMaskRouter)


if __name__ == "__main__":
    unittest.main()
