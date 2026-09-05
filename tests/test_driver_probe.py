"""Behavioural tests for the standalone pre-launch driver probe."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "launchable/check_driver.sh"


def run_probe(*, driver_versions: list[str] | None, with_docker: bool = True):
    """Run the probe against a stubbed nvidia-smi so no GPU is required."""
    with tempfile.TemporaryDirectory() as directory:
        stub_dir = Path(directory)
        if driver_versions is not None:
            (stub_dir / "plain.txt").write_text(
                "".join(f"{version}\n" for version in driver_versions), encoding="utf-8"
            )
            (stub_dir / "rows.txt").write_text(
                "index, name, memory.total [MiB], compute_cap, driver_version\n"
                + "".join(
                    f"{index}, Stub GPU, 81920 MiB, 8.0, {version}\n"
                    for index, version in enumerate(driver_versions)
                ),
                encoding="utf-8",
            )
            stub = stub_dir / "nvidia-smi"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$*" == *noheader* ]]; then\n'
                f'  cat "{stub_dir}/plain.txt"\n'
                "else\n"
                f'  cat "{stub_dir}/rows.txt"\n'
                "fi\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)

        environment = dict(os.environ)
        if with_docker:
            docker_stub = stub_dir / "docker"
            docker_stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            docker_stub.chmod(0o755)
            environment["PATH"] = f"{stub_dir}:{environment['PATH']}"
        else:
            # Keep the standard utilities the probe needs, but no docker binary.
            environment["PATH"] = f"{stub_dir}:/usr/bin:/bin"
        return subprocess.run(
            ["/bin/bash", str(PROBE)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )


class DriverProbeTests(unittest.TestCase):
    def test_supported_driver_passes(self):
        result = run_probe(driver_versions=["595.91.07"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("minor-version/forward compatibility", result.stdout)
        self.assertIn("Driver gate passed", result.stdout)

    def test_native_driver_is_reported_as_native(self):
        result = run_probe(driver_versions=["610.43.02"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("natively", result.stdout)

    def test_observed_brev_a100_image_is_rejected_with_remedies(self):
        result = run_probe(driver_versions=["565.57.01"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("below 580.65.06", result.stderr)
        self.assertIn("upgrade_driver.sh", result.stderr)
        self.assertIn("still requires a base driver of 580 or newer", result.stderr)

    def test_mixed_driver_versions_fail_on_the_oldest(self):
        result = run_probe(driver_versions=["610.43.02", "565.57.01"])
        self.assertEqual(result.returncode, 1)

    def test_missing_nvidia_smi_is_a_distinct_exit_code(self):
        result = run_probe(driver_versions=None, with_docker=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no NVIDIA driver is installed", result.stderr)

    def test_missing_docker_warns_without_failing_the_driver_verdict(self):
        result = run_probe(driver_versions=["595.91.07"], with_docker=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WARN", result.stderr)


if __name__ == "__main__":
    unittest.main()
