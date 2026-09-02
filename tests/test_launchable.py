from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LaunchableTests(unittest.TestCase):
    def test_brev_host_port_avoids_managed_jupyter_default(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        self.assertIn("NEMOTRON_JUPYTER_PORT:-8889", setup)
        self.assertIn("127.0.0.1:${JUPYTER_PORT}:8888", setup)
        self.assertIn("port_is_free", setup)
        self.assertIn("jupyter_port: 8889", manifest)
        self.assertIn("port: 8889", manifest)
        self.assertIn('default: "8889"', manifest)
        self.assertIn("Jupyter is listening on VM loopback port", setup)
        self.assertIn("ssh -N -L", setup)
        self.assertIn(
            "http://127.0.0.1:${JUPYTER_PORT}/lab/tree/notebooks/01_cloud_api_baseline.ipynb", setup
        )

    def test_pasted_brev_script_bootstraps_the_repository(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        self.assertIn("https://github.com/siddBanPsu/nemotron-fine-tuning.git", setup)
        self.assertIn("NEMOTRON_REPOSITORY_REF:-main", setup)
        self.assertIn("git clone --filter=blob:none --no-checkout", setup)
        self.assertIn("checkout --detach FETCH_HEAD", setup)
        self.assertIn("launchable/container-entrypoint.sh", setup)

    def test_data_root_avoids_bind_mounting_a_restricted_home_checkout(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        self.assertIn('REPOSITORY_MODE="${NEMOTRON_REPOSITORY_MODE:-auto}"', setup)
        self.assertIn('[[ "${LOCAL_REPOSITORY_DIR}" == "${DATA_ROOT}"/* ]]', setup)
        self.assertIn("staging a clean clone for Docker bind mounts", setup)
        self.assertIn("NEMOTRON_REPOSITORY_MODE=local", setup)
        self.assertIn("name: NEMOTRON_REPOSITORY_MODE", manifest)

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
        self.assertIn("Repository bind mount is unreadable", setup)
        self.assertIn("--ipc=host", setup)

    def test_persistent_storage_can_use_a_large_vm_volume(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        self.assertIn('DATA_ROOT="${NEMOTRON_DATA_ROOT:-}"', setup)
        self.assertIn("${NEMOTRON_STORAGE_DIR:-${DATA_ROOT}/storage}", setup)
        self.assertIn("${NEMOTRON_HF_CACHE_DIR:-${DATA_ROOT}/huggingface}", setup)
        self.assertIn("${NEMOTRON_ARTIFACTS_DIR:-${DATA_ROOT}/artifacts}", setup)
        self.assertIn("${NEMOTRON_CACHE_DIR:-${DATA_ROOT}/cache}", setup)
        self.assertIn("${NEMOTRON_TEMP_DIR:-${DATA_ROOT}/tmp}", setup)
        self.assertIn("${ARTIFACTS_DIR}:/workspace/launchable/artifacts", setup)
        self.assertIn("${CACHE_DIR}:/workspace/cache", setup)
        self.assertIn("${TEMP_DIR}:/workspace/tmp", setup)
        self.assertIn("TMPDIR=/workspace/tmp", setup)
        self.assertIn("TRITON_CACHE_DIR=/workspace/cache/triton", setup)
        self.assertIn("name: NEMOTRON_DATA_ROOT", manifest)

    def test_torch_is_verified_from_the_nemo_image_not_host_installed(self):
        entrypoint = (ROOT / "launchable/container-entrypoint.sh").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements-lab.txt").read_text(encoding="utf-8")
        self.assertIn("Preinstalled PyTorch", entrypoint)
        self.assertIn("torch.cuda.is_available()", entrypoint)
        self.assertNotIn("\ntorch", requirements)

    def test_missing_docker_fails_with_standalone_vm_guidance(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        docker_command_check = setup.index("command -v docker")
        docker_first_use = setup.index("docker info")
        self.assertLess(docker_command_check, docker_first_use)
        self.assertIn("Docker Engine plus NVIDIA Container Toolkit", setup)
        self.assertIn("apptainer singularity enroot podman nerdctl", setup)


if __name__ == "__main__":
    unittest.main()
