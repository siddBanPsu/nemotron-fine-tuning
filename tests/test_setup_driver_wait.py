"""Behavioural tests for the setup script's GPU-driver readiness gate."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "launchable/setup.sh"


def run_setup(*, nvidia_smi: str | None, wait_seconds: str = "0", docker: str | None = None):
    """Run setup.sh far enough to exercise the driver gate, with stub binaries."""
    with tempfile.TemporaryDirectory() as directory:
        stub_dir = Path(directory)
        docker_stub = stub_dir / "docker"
        docker_stub.write_text(docker or "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        docker_stub.chmod(0o755)
        if nvidia_smi is not None:
            stub = stub_dir / "nvidia-smi"
            stub.write_text(nvidia_smi, encoding="utf-8")
            stub.chmod(0o755)
        environment = dict(os.environ)
        environment["PATH"] = f"{stub_dir}:/usr/bin:/bin"
        environment["NEMOTRON_DRIVER_WAIT_SECONDS"] = wait_seconds
        environment["NEMOTRON_DATA_ROOT"] = str(stub_dir / "data")
        return subprocess.run(
            ["/bin/bash", str(SETUP)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )


HEALTHY_NVIDIA_SMI = (
    "#!/usr/bin/env bash\n"
    'if [[ "$*" == *noheader* ]]; then echo "595.91.07"; exit 0; fi\n'
    'if [[ "$*" == -L ]]; then echo "GPU 0: Stub"; exit 0; fi\n'
    'echo "0, Stub GPU, 81920 MiB, 8.0, 595.91.07"\n'
)


class SetupDriverWaitTests(unittest.TestCase):
    def test_absent_driver_explains_itself_instead_of_crashing(self):
        result = run_setup(nvidia_smi=None)
        self.assertEqual(result.returncode, 1)
        self.assertIn("exposes no NVIDIA driver", result.stderr)
        self.assertIn("check_driver.sh", result.stderr)
        self.assertNotIn("command not found", result.stderr)

    def test_unresponsive_driver_is_reported_separately_from_a_missing_one(self):
        result = run_setup(nvidia_smi="#!/usr/bin/env bash\nexit 9\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("could not talk to the driver", result.stderr)
        self.assertIn("needs a reboot", result.stderr)

    def test_a_driver_that_appears_late_is_waited_for(self):
        # Fails once, then succeeds: the gate must retry rather than give up.
        with tempfile.TemporaryDirectory() as state:
            marker = Path(state) / "attempted"
            script = (
                "#!/usr/bin/env bash\n"
                f'if [ ! -f "{marker}" ]; then touch "{marker}"; exit 9; fi\n'
                'if [[ "$*" == *noheader* ]]; then echo "595.91.07"; exit 0; fi\n'
                'if [[ "$*" == -L ]]; then echo "GPU 0: Stub"; exit 0; fi\n'
                'echo "0, Stub GPU, 81920 MiB, 8.0, 595.91.07"\n'
            )
            result = run_setup(nvidia_smi=script, wait_seconds="30")
        self.assertIn("NVIDIA driver answered after", result.stdout)
        # The run continues past the driver gate; later stages need real Docker.
        self.assertNotIn("exposes no NVIDIA driver", result.stderr)

    def test_an_unavailable_docker_daemon_is_waited_for_then_explained(self):
        # 'docker info' fails the way a rate-limited daemon does.
        result = run_setup(
            nvidia_smi=HEALTHY_NVIDIA_SMI,
            docker='#!/usr/bin/env bash\nif [[ "$1" == "info" ]]; then exit 1; fi\nexit 0\n',
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("stayed unavailable", result.stderr)
        self.assertIn("start-limit-hit", result.stderr)
        self.assertIn("reset-failed docker.service", result.stderr)

    def test_invalid_wait_budget_is_rejected(self):
        result = run_setup(nvidia_smi=None, wait_seconds="soon")
        self.assertEqual(result.returncode, 1)
        self.assertIn("NEMOTRON_DRIVER_WAIT_SECONDS", result.stderr)


if __name__ == "__main__":
    unittest.main()
