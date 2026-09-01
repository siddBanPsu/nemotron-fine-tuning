from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_preflight_direct_execution_bootstraps_src_layout(self):
        result = subprocess.run(
            [sys.executable, "-I", "scripts/preflight.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)


if __name__ == "__main__":
    unittest.main()
