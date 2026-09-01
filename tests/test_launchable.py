from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LaunchableTests(unittest.TestCase):
    def test_brev_host_port_avoids_managed_jupyter_default(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        self.assertIn('NEMOTRON_JUPYTER_PORT:-8889', setup)
        self.assertIn('127.0.0.1:${JUPYTER_PORT}:8888', setup)
        self.assertIn("port_is_free", setup)
        self.assertIn("jupyter_port: 8889", manifest)
        self.assertIn("port: 8889", manifest)
        self.assertIn('default: "8889"', manifest)


if __name__ == "__main__":
    unittest.main()
