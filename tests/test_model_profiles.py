from __future__ import annotations

import unittest
from unittest.mock import patch

from nemotron_ft_lab.constants import MEGATRON_BRIDGE_REVISION
from nemotron_ft_lab.model_profiles import (
    DEFAULT_MODEL_PROFILE_NAME,
    LIGHTNING35_ADVANCED,
    NANO9B_WORKSHOP,
    get_model_profile,
)


class ModelProfileTests(unittest.TestCase):
    def test_nano_is_the_default_one_gpu_workshop(self):
        with patch.dict("os.environ", {}, clear=True):
            profile = get_model_profile()
        self.assertEqual(DEFAULT_MODEL_PROFILE_NAME, "nano9b_workshop")
        self.assertEqual(profile, NANO9B_WORKSHOP)
        self.assertEqual(profile.peft_world_sizes, (1,))
        self.assertEqual(profile.system_prompt, "/no_think")
        self.assertEqual(profile.mamba_ssm_cache_dtype, "float32")
        self.assertEqual(profile.revision, "6533e8de2c68e4536bf7c411d7a3ce5734111476")
        self.assertFalse(profile.include_reasoning)

    def test_bridge_pin_is_a_full_commit(self):
        self.assertEqual(len(MEGATRON_BRIDGE_REVISION), 40)

    def test_lightning_profile_stays_separate_and_advanced(self):
        profile = get_model_profile("lightning")
        self.assertEqual(profile, LIGHTNING35_ADVANCED)
        self.assertIn(2, profile.peft_world_sizes)
        self.assertTrue(profile.include_reasoning)
        self.assertTrue(profile.full_sft_supported)
        self.assertNotEqual(profile.artifact_slug, NANO9B_WORKSHOP.artifact_slug)

    def test_unknown_profile_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown model profile"):
            get_model_profile("mystery")


if __name__ == "__main__":
    unittest.main()
