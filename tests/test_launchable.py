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

    def test_pasted_brev_script_bootstraps_the_repository(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        self.assertIn("https://github.com/siddBanPsu/nemotron-fine-tuning.git", setup)
        self.assertIn("NEMOTRON_REPOSITORY_REF:-main", setup)
        self.assertIn("git clone --filter=blob:none --no-checkout", setup)
        self.assertIn("checkout --detach FETCH_HEAD", setup)
        self.assertIn("launchable/container-entrypoint.sh", setup)

    def test_driver_gate_precedes_the_large_container_pull(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        driver_gate = setup.index("driver_version_at_least")
        driver_check = setup.index('driver_version_at_least "${DRIVER_VERSION}"')
        container_pull = setup.index('retry docker pull "${IMAGE}"')
        self.assertLess(driver_gate, driver_check)
        self.assertLess(driver_check, container_pull)
        self.assertIn("NEMOTRON_MINIMUM_DRIVER_VERSION:-580.65.06", setup)
        self.assertIn('NATIVE_DRIVER_VERSION="610.43.02"', setup)

    def test_cuda_compatibility_is_proven_before_jupyter_starts(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        container_pull = setup.index('retry docker pull "${IMAGE}"')
        cuda_smoke = setup.index("torch.cuda.is_available()")
        jupyter_container = setup.index("docker run --detach")
        self.assertLess(container_pull, cuda_smoke)
        self.assertLess(cuda_smoke, jupyter_container)
        self.assertIn("torch.ones(1, device=device)", setup)

    def test_persistent_storage_can_use_a_large_vm_volume(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        self.assertIn('DATA_ROOT="${NEMOTRON_DATA_ROOT:-}"', setup)
        self.assertIn('${NEMOTRON_STORAGE_DIR:-${DATA_ROOT}/storage}', setup)
        self.assertIn('${NEMOTRON_HF_CACHE_DIR:-${DATA_ROOT}/huggingface}', setup)
        self.assertIn("name: NEMOTRON_DATA_ROOT", manifest)


if __name__ == "__main__":
    unittest.main()
