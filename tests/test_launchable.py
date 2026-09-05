from __future__ import annotations

import unittest
from pathlib import Path

from nemotron_ft_lab.constants import MEGATRON_BRIDGE_REVISION

ROOT = Path(__file__).resolve().parents[1]


class LaunchableTests(unittest.TestCase):
    def test_nano_is_default_and_profile_controls_prefetch_conversion(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        entrypoint = (ROOT / "launchable/container-entrypoint.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        self.assertIn('NEMOTRON_MODEL_PROFILE:-nano9b_workshop', setup)
        self.assertIn('NEMOTRON_MODEL_PROFILE:-nano9b_workshop', entrypoint)
        self.assertIn('default: nano9b_workshop', manifest)
        self.assertIn('allowed_values: [nano9b_workshop, lightning35_advanced]', manifest)
        self.assertIn('python scripts/show_model_profile.py --profile "${MODEL_PROFILE}" --tsv', entrypoint)
        self.assertIn('--model-profile "${MODEL_PROFILE}"', entrypoint)
        self.assertIn('nemotron_nano_9b_v2_peft_config', entrypoint)
        self.assertIn(MEGATRON_BRIDGE_REVISION, entrypoint)
        self.assertIn('-e NVIDIA_API_KEY', setup)
        self.assertIn('-e HF_TOKEN', setup)

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
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        driver_gate = setup.index("driver_version_at_least")
        driver_check = setup.index('driver_version_at_least "${DRIVER_VERSION}"')
        container_pull = setup.index('retry docker pull "${IMAGE}"')
        self.assertLess(driver_gate, driver_check)
        self.assertLess(driver_check, container_pull)
        self.assertIn("NEMOTRON_MINIMUM_DRIVER_VERSION:-580.65.06", setup)
        self.assertIn('NATIVE_DRIVER_VERSION="610.43.02"', setup)
        self.assertIn("VM-image compatibility failure", setup)
        self.assertIn("driver 595.91.07", setup)
        self.assertIn("driver 565.57.01", manifest)
        self.assertIn("RTX PRO 6000 Blackwell Server Edition 96 GB", manifest)

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

    def test_forward_compatibility_is_retried_before_the_instance_is_rejected(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        smoke_definition = setup.index("run_cuda_smoke_test() {")
        first_attempt = setup.index("if run_cuda_smoke_test; then")
        compat_retry = setup.index('FORWARD_COMPAT_ARGS=(-e "LD_LIBRARY_PATH=')
        jupyter_container = setup.index("docker run --detach")
        self.assertLess(smoke_definition, first_attempt)
        self.assertLess(first_attempt, compat_retry)
        self.assertLess(compat_retry, jupyter_container)
        self.assertIn("/usr/local/cuda/compat/lib.real", setup)
        # A passing compat retry must be inherited by the long-lived container.
        self.assertIn('${FORWARD_COMPAT_ARGS[@]+"${FORWARD_COMPAT_ARGS[@]}"}', setup)
        # Compat cannot rescue a driver below the floor; the gate stays honest.
        self.assertIn("still requires a base driver of 580 or newer", setup)

    def test_driver_recovery_tooling_is_documented_and_gated(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        probe = (ROOT / "launchable/check_driver.sh").read_text(encoding="utf-8")
        upgrade = (ROOT / "launchable/upgrade_driver.sh").read_text(encoding="utf-8")
        manifest = (ROOT / "launchable/brev-launchable.yaml").read_text(encoding="utf-8")
        readme = (ROOT / "launchable/README.md").read_text(encoding="utf-8")
        self.assertIn("launchable/check_driver.sh", setup)
        self.assertIn("launchable/upgrade_driver.sh", setup)
        self.assertIn("preflight_script_file: launchable/check_driver.sh", manifest)
        self.assertIn("bash launchable/check_driver.sh", readme)
        self.assertIn("NEMOTRON_CONFIRM_DRIVER_UPGRADE", upgrade)
        self.assertIn("NEMOTRON_MINIMUM_DRIVER_VERSION:-580.65.06", probe)
        # The probe must not need the repository install or a container pull.
        self.assertNotIn("docker pull", probe)
        self.assertNotIn("pip install", probe)

    def test_secure_link_publish_requires_a_token_beyond_loopback(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        entrypoint = (ROOT / "launchable/container-entrypoint.sh").read_text(encoding="utf-8")
        # Loopback is always published, so SSH tunnels keep working.
        self.assertIn('PUBLISH_ARGS=(-p "127.0.0.1:${JUPYTER_PORT}:8888")', setup)
        self.assertIn('PUBLISH_ARGS+=(-p "${BIND_ADDRESS}:${JUPYTER_PORT}:8888")', setup)
        # A reachable server must authenticate; there is no tokenless escape.
        token_guard = setup.index('if [ -n "${BIND_ADDRESS}" ] && [ -z "${JUPYTER_TOKEN}" ]')
        publish = setup.index("PUBLISH_ARGS=(-p")
        self.assertLess(token_guard, publish)
        self.assertIn("JUPYTER_TOKEN=\"$(generate_jupyter_token)\"", setup)
        self.assertNotIn("ALLOW_TOKENLESS", setup)
        self.assertIn('--ServerApp.token="${NEMOTRON_JUPYTER_TOKEN:-}"', entrypoint)
        self.assertNotIn("--ServerApp.token='' ", entrypoint)
        # Readiness must authenticate too, or it would spin until timeout.
        self.assertIn('Authorization: token ${JUPYTER_TOKEN}', setup)

    def test_missing_docker_fails_with_standalone_vm_guidance(self):
        setup = (ROOT / "launchable/setup.sh").read_text(encoding="utf-8")
        docker_command_check = setup.index("command -v docker")
        docker_first_use = setup.index("docker info")
        self.assertLess(docker_command_check, docker_first_use)
        self.assertIn("Docker Engine plus NVIDIA Container Toolkit", setup)
        self.assertIn("apptainer singularity enroot podman nerdctl", setup)


if __name__ == "__main__":
    unittest.main()
