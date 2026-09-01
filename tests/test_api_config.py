from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_ft_lab.api_config import load_nvidia_api_config
from nemotron_ft_lab.constants import NVIDIA_API_BASE_URL


class ApiConfigTests(unittest.TestCase):
    def test_public_defaults_need_no_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_nvidia_api_config(directory, environ={})
        self.assertEqual(config.profile_name, "public")
        self.assertEqual(config.base_url, NVIDIA_API_BASE_URL)
        self.assertEqual(config.requests_per_minute, 30)
        self.assertEqual(config.artifact_prefix, "")

    def test_local_file_is_auto_selected_and_namespaces_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/api.local.toml").write_text(
                """
profile_name = "internal-fast"
base_url = "https://internal.example/v1"
requests_per_minute = 240

[models.lightning]
model_id = "internal/lightning"

[models.ultra]
model_id = "internal/ultra"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_nvidia_api_config(root, environ={})
        self.assertEqual(config.profile_name, "internal-fast")
        self.assertEqual(config.base_url, "https://internal.example/v1")
        self.assertEqual(config.lightning.model_id, "internal/lightning")
        self.assertEqual(config.lightning.served_variant, "internal/lightning")
        self.assertEqual(config.requests_per_minute, 240)
        self.assertEqual(config.artifact_prefix, "internal_fast_")

    def test_public_selector_ignores_default_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/api.local.toml").write_text(
                'profile_name = "internal"\nbase_url = "https://internal.example/v1"\n',
                encoding="utf-8",
            )
            config = load_nvidia_api_config(
                root,
                profile="public",
                environ={
                    "NEMOTRON_API_PROFILE": "local",
                    "NEMOTRON_API_BASE_URL": "https://also-internal.example/v1",
                },
            )
        self.assertEqual(config.profile_name, "public")
        self.assertEqual(config.base_url, NVIDIA_API_BASE_URL)

    def test_environment_overrides_are_custom_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_nvidia_api_config(
                directory,
                environ={
                    "NEMOTRON_API_BASE_URL": "https://gateway.example/v1/",
                    "NEMOTRON_API_LIGHTNING_MODEL_ID": "gateway/lightning",
                    "NEMOTRON_API_REQUESTS_PER_MINUTE": "90",
                },
            )
        self.assertEqual(config.profile_name, "custom")
        self.assertEqual(config.base_url, "https://gateway.example/v1")
        self.assertEqual(config.lightning.model_id, "gateway/lightning")
        self.assertEqual(config.lightning.served_variant, "gateway/lightning")
        self.assertEqual(config.requests_per_minute, 90)
        self.assertEqual(config.artifact_prefix, "custom_")


if __name__ == "__main__":
    unittest.main()
