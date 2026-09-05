#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="nemotron-35-ft-lab"
IMAGE="${NEMOTRON_CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.08}"
MINIMUM_DRIVER_VERSION="${NEMOTRON_MINIMUM_DRIVER_VERSION:-580.65.06}"
NATIVE_DRIVER_VERSION="610.43.02"
JUPYTER_PORT="${NEMOTRON_JUPYTER_PORT:-8889}"
PREFETCH_MODEL="${NEMOTRON_PREFETCH_MODEL:-0}"
MODEL_PROFILE="${NEMOTRON_MODEL_PROFILE:-nano9b_workshop}"
REPOSITORY_URL="${NEMOTRON_REPOSITORY_URL:-https://github.com/siddBanPsu/nemotron-fine-tuning.git}"
REPOSITORY_REF="${NEMOTRON_REPOSITORY_REF:-main}"
REPOSITORY_MODE="${NEMOTRON_REPOSITORY_MODE:-auto}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPOSITORY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_DIR=""
DATA_ROOT="${NEMOTRON_DATA_ROOT:-}"
if [ -n "${DATA_ROOT}" ]; then
  BOOTSTRAP_REPOSITORY_DIR="${NEMOTRON_REPOSITORY_DIR:-${DATA_ROOT}/repository}"
  STORAGE_DIR="${NEMOTRON_STORAGE_DIR:-${DATA_ROOT}/storage}"
  HF_CACHE_DIR="${NEMOTRON_HF_CACHE_DIR:-${DATA_ROOT}/huggingface}"
  ARTIFACTS_DIR="${NEMOTRON_ARTIFACTS_DIR:-${DATA_ROOT}/artifacts}"
  CACHE_DIR="${NEMOTRON_CACHE_DIR:-${DATA_ROOT}/cache}"
  TEMP_DIR="${NEMOTRON_TEMP_DIR:-${DATA_ROOT}/tmp}"
else
  BOOTSTRAP_REPOSITORY_DIR="${NEMOTRON_REPOSITORY_DIR:-${HOME}/nemotron-fine-tuning-launchable}"
  STORAGE_DIR="${NEMOTRON_STORAGE_DIR:-${HOME}/nemotron-35-ft-storage}"
  HF_CACHE_DIR="${NEMOTRON_HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
  ARTIFACTS_DIR="${NEMOTRON_ARTIFACTS_DIR:-}"
  CACHE_DIR="${NEMOTRON_CACHE_DIR:-${STORAGE_DIR}/cache}"
  TEMP_DIR="${NEMOTRON_TEMP_DIR:-${STORAGE_DIR}/tmp}"
fi

for PERSISTENT_PATH in \
  "${BOOTSTRAP_REPOSITORY_DIR}" "${STORAGE_DIR}" "${HF_CACHE_DIR}" \
  "${CACHE_DIR}" "${TEMP_DIR}"; do
  case "${PERSISTENT_PATH}" in
    /*) ;;
    *)
      echo "Persistent storage paths must be absolute; received '${PERSISTENT_PATH}'." >&2
      echo "Set NEMOTRON_DATA_ROOT to an absolute path on the large VM data volume." >&2
      exit 1
      ;;
  esac
done

case "${PREFETCH_MODEL}" in
  0|1) ;;
  *)
    echo "NEMOTRON_PREFETCH_MODEL must be 0 or 1; received '${PREFETCH_MODEL}'." >&2
    exit 1
    ;;
esac

case "${MODEL_PROFILE}" in
  nano9b_workshop) DISK_GUIDANCE_GB=200 ;;
  lightning35_advanced) DISK_GUIDANCE_GB=300 ;;
  *)
    echo "NEMOTRON_MODEL_PROFILE must be nano9b_workshop or lightning35_advanced; received '${MODEL_PROFILE}'." >&2
    exit 1
    ;;
esac
echo "Selected model profile: ${MODEL_PROFILE}"

case "${REPOSITORY_MODE}" in
  auto|local|bootstrap) ;;
  *)
    echo "NEMOTRON_REPOSITORY_MODE must be auto, local, or bootstrap; received '${REPOSITORY_MODE}'." >&2
    exit 1
    ;;
esac

if ! [[ "${JUPYTER_PORT}" =~ ^[0-9]+$ ]] || (( JUPYTER_PORT < 1024 || JUPYTER_PORT > 65535 )); then
  echo "NEMOTRON_JUPYTER_PORT must be an unprivileged TCP port; received '${JUPYTER_PORT}'." >&2
  exit 1
fi

retry() {
  local attempt=1
  local maximum=4
  local delay=5
  until "$@"; do
    if (( attempt >= maximum )); then
      echo "Command failed after ${maximum} attempts: $*" >&2
      return 1
    fi
    echo "Attempt ${attempt} failed; retrying in ${delay}s..." >&2
    sleep "${delay}"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

driver_version_at_least() {
  python3 - "$1" "$2" <<'PY'
import re
import sys


def version_parts(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in re.findall(r"\d+", value))
    if not parts:
        raise ValueError(f"No numeric version found in {value!r}")
    return parts


actual, required = sys.argv[1:]
raise SystemExit(0 if version_parts(actual) >= version_parts(required) else 1)
PY
}

resolve_repository() {
  local use_local=0
  if [ "${REPOSITORY_MODE}" = "local" ]; then
    use_local=1
  elif [ "${REPOSITORY_MODE}" = "auto" ] && \
       [ -f "${LOCAL_REPOSITORY_DIR}/launchable/container-entrypoint.sh" ]; then
    if [ -z "${DATA_ROOT}" ] || [[ "${LOCAL_REPOSITORY_DIR}" == "${DATA_ROOT}"/* ]]; then
      use_local=1
    else
      echo "The local checkout is outside NEMOTRON_DATA_ROOT; staging a clean clone for Docker bind mounts."
    fi
  fi

  if [ "${use_local}" = "1" ]; then
    if [ ! -f "${LOCAL_REPOSITORY_DIR}/launchable/container-entrypoint.sh" ]; then
      echo "NEMOTRON_REPOSITORY_MODE=local, but no repository surrounds this setup script." >&2
      exit 1
    fi
    REPOSITORY_DIR="${LOCAL_REPOSITORY_DIR}"
    echo "Using the repository that contains this setup script: ${REPOSITORY_DIR}"
    return
  fi

  echo "The Brev lifecycle script is outside the checkout; bootstrapping ${REPOSITORY_URL}."
  if [ -e "${BOOTSTRAP_REPOSITORY_DIR}" ] && [ ! -d "${BOOTSTRAP_REPOSITORY_DIR}/.git" ]; then
    echo "Cannot bootstrap into ${BOOTSTRAP_REPOSITORY_DIR}: it exists but is not a Git checkout." >&2
    exit 1
  fi
  if [ ! -d "${BOOTSTRAP_REPOSITORY_DIR}/.git" ]; then
    mkdir -p "$(dirname "${BOOTSTRAP_REPOSITORY_DIR}")"
    retry git clone --filter=blob:none --no-checkout \
      "${REPOSITORY_URL}" "${BOOTSTRAP_REPOSITORY_DIR}"
  fi
  retry git -C "${BOOTSTRAP_REPOSITORY_DIR}" fetch --depth 1 origin "${REPOSITORY_REF}"
  git -C "${BOOTSTRAP_REPOSITORY_DIR}" checkout --detach FETCH_HEAD
  REPOSITORY_DIR="${BOOTSTRAP_REPOSITORY_DIR}"

  if [ ! -f "${REPOSITORY_DIR}/launchable/container-entrypoint.sh" ]; then
    echo "Repository ref '${REPOSITORY_REF}' does not contain launchable/container-entrypoint.sh." >&2
    exit 1
  fi
  echo "Using repository commit $(git -C "${REPOSITORY_DIR}" rev-parse HEAD)."
}

port_is_free() {
  python3 - "${JUPYTER_PORT}" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    try:
        listener.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

echo "[1/7] Checking GPU, driver, and container runtime"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not on PATH." >&2
  echo "This standalone-VM path requires Docker Engine plus NVIDIA Container Toolkit." >&2
  echo "Brev VM Mode provides both; on a managed cluster, ask the administrator which GPU container runtime is supported." >&2
  for ALTERNATIVE_RUNTIME in apptainer singularity enroot podman nerdctl; do
    if command -v "${ALTERNATIVE_RUNTIME}" >/dev/null 2>&1; then
      echo "Detected alternative runtime: ${ALTERNATIVE_RUNTIME}. This setup script does not invoke it." >&2
    fi
  done
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
mapfile -t DRIVER_VERSIONS < <(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
)
if [ "${#DRIVER_VERSIONS[@]}" -eq 0 ]; then
  echo "No NVIDIA driver version was reported by nvidia-smi." >&2
  exit 1
fi
for DRIVER_VERSION in "${DRIVER_VERSIONS[@]}"; do
  if ! driver_version_at_least "${DRIVER_VERSION}" "${MINIMUM_DRIVER_VERSION}"; then
    echo "NVIDIA driver ${DRIVER_VERSION} is too old for ${IMAGE}." >&2
    echo "CUDA 13.x minor-version compatibility requires driver ${MINIMUM_DRIVER_VERSION} or newer." >&2
    echo "This is a VM-image compatibility failure, not an A100 memory or compute-capability failure." >&2
    echo "The lifecycle script cannot replace the kernel driver: activating a new driver requires a VM reboot, which would fail this Brev on-create run." >&2
    echo "Create a fresh Brev instance whose base image reports driver ${MINIMUM_DRIVER_VERSION}+; the setup stops before the large container pull." >&2
    echo "A Brev RTX PRO 6000 Blackwell Server Edition instance with driver 595.91.07 has passed this repository's container/CUDA startup gate." >&2
    echo "An A100 instance is suitable only when its selected provider/base image also supplies driver ${MINIMUM_DRIVER_VERSION}+." >&2
    echo "Do not substitute an older NeMo image: it is not the verified dependency stack for these training recipes." >&2
    exit 1
  fi
  if ! driver_version_at_least "${DRIVER_VERSION}" "${NATIVE_DRIVER_VERSION}"; then
    echo "Driver ${DRIVER_VERSION} will use documented CUDA 13.x minor-version compatibility."
    echo "The container CUDA smoke test after the pull must pass before Jupyter starts."
  fi
done
if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed, but the daemon is unavailable or this user lacks access." >&2
  echo "Run 'docker info' directly for the host-specific error before retrying setup." >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Replacing existing ${CONTAINER_NAME} container"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
if ! port_is_free; then
  echo "Host port 127.0.0.1:${JUPYTER_PORT} is already in use." >&2
  echo "Set NEMOTRON_JUPYTER_PORT to a free port and configure the Brev Secure Link to the same port." >&2
  exit 1
fi

echo "[2/7] Resolving the versioned repository"
resolve_repository

if [ -z "${ARTIFACTS_DIR}" ]; then
  ARTIFACTS_DIR="${REPOSITORY_DIR}/artifacts"
fi
case "${ARTIFACTS_DIR}" in
  /*) ;;
  *)
    echo "NEMOTRON_ARTIFACTS_DIR must be absolute; received '${ARTIFACTS_DIR}'." >&2
    exit 1
    ;;
esac

echo "[3/7] Preparing persistent model/checkpoint storage"
if ! mkdir -p \
  "${STORAGE_DIR}" "${HF_CACHE_DIR}" "${ARTIFACTS_DIR}" "${CACHE_DIR}" "${TEMP_DIR}"; then
  echo "Could not create persistent storage. The selected filesystem may be full." >&2
  echo "Set NEMOTRON_DATA_ROOT to an absolute path on a volume with roughly ${DISK_GUIDANCE_GB} GB available." >&2
  exit 1
fi
echo "Checkpoint storage: ${STORAGE_DIR}"
echo "Hugging Face cache: ${HF_CACHE_DIR}"
echo "Notebook artifacts: ${ARTIFACTS_DIR}"
echo "Runtime cache: ${CACHE_DIR}"
echo "Temporary files: ${TEMP_DIR}"
df -h \
  "${STORAGE_DIR}" "${HF_CACHE_DIR}" "${ARTIFACTS_DIR}" "${CACHE_DIR}" "${TEMP_DIR}" \
  | awk 'NR == 1 || !seen[$1]++'

echo "[4/7] Pulling the pinned NeMo container"
retry docker pull "${IMAGE}"

echo "[5/7] Validating the repository mount and CUDA inside the pinned container"
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${REPOSITORY_DIR}:/workspace/launchable:ro" \
  --interactive "${IMAGE}" python - <<'PY'
from pathlib import Path

import torch

entrypoint = Path("/workspace/launchable/launchable/container-entrypoint.sh")
if not entrypoint.is_file():
    raise RuntimeError(f"Repository bind mount is unreadable: {entrypoint}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot initialize CUDA inside the NeMo container.")
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    value = (torch.ones(1, device=device) + 1).item()
    if value != 2:
        raise RuntimeError(f"CUDA arithmetic smoke test failed on {device}: {value}")
    torch.cuda.synchronize(device)
print(
    f"CUDA smoke test passed on {torch.cuda.device_count()} GPU(s): "
    f"{torch.cuda.get_device_name(0)}"
)
PY

echo "[6/7] Starting the isolated Jupyter lab container"
docker run --detach \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "127.0.0.1:${JUPYTER_PORT}:8888" \
  -e "NEMOTRON_PREFETCH_MODEL=${PREFETCH_MODEL}" \
  -e "NEMOTRON_MODEL_PROFILE=${MODEL_PROFILE}" \
  -e "NEMOTRON_ARTIFACTS_DIR=/workspace/launchable/artifacts" \
  -e NVIDIA_API_KEY \
  -e HF_TOKEN \
  -e "HF_HOME=/root/.cache/huggingface" \
  -e "XDG_CACHE_HOME=/workspace/cache/xdg" \
  -e "PIP_CACHE_DIR=/workspace/cache/pip" \
  -e "TORCH_HOME=/workspace/cache/torch" \
  -e "TORCH_EXTENSIONS_DIR=/workspace/cache/torch-extensions" \
  -e "TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torch-inductor" \
  -e "TRITON_CACHE_DIR=/workspace/cache/triton" \
  -e "CUDA_CACHE_PATH=/workspace/cache/cuda" \
  -e "NUMBA_CACHE_DIR=/workspace/cache/numba" \
  -e "PYTHONPYCACHEPREFIX=/workspace/cache/pycache" \
  -e "TMPDIR=/workspace/tmp" \
  -e "IPYTHONDIR=/workspace/cache/ipython" \
  -e "JUPYTER_CONFIG_DIR=/workspace/cache/jupyter/config" \
  -e "JUPYTER_DATA_DIR=/workspace/cache/jupyter/data" \
  -e "JUPYTER_RUNTIME_DIR=/workspace/cache/jupyter-runtime" \
  -e "PYTHONPATH=/workspace/launchable/src" \
  -v "${REPOSITORY_DIR}:/workspace/launchable" \
  -v "${STORAGE_DIR}:/workspace/storage" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -v "${ARTIFACTS_DIR}:/workspace/launchable/artifacts" \
  -v "${CACHE_DIR}:/workspace/cache" \
  -v "${TEMP_DIR}:/workspace/tmp" \
  -w /workspace/launchable \
  "${IMAGE}" \
  bash /workspace/launchable/launchable/container-entrypoint.sh

echo "[7/7] Waiting for Jupyter readiness"
for _ in $(seq 1 1080); do
  if curl --fail --silent "http://127.0.0.1:${JUPYTER_PORT}/api" >/dev/null; then
    echo "Ready: Jupyter is listening on VM loopback port ${JUPYTER_PORT}."
    echo "Brev: open the authenticated Secure Link configured for port ${JUPYTER_PORT}."
    echo "Standalone VM: from your workstation run:"
    echo "  ssh -N -L ${JUPYTER_PORT}:127.0.0.1:${JUPYTER_PORT} USER@VM_HOST"
    echo "Then open:"
    echo "  http://127.0.0.1:${JUPYTER_PORT}/lab/tree/notebooks/01_cloud_api_baseline.ipynb"
    echo "Landing notebook: notebooks/01_cloud_api_baseline.ipynb"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker logs --tail 300 "${CONTAINER_NAME}" >&2
    exit 1
  fi
  sleep 5
done

docker logs --tail 300 "${CONTAINER_NAME}" >&2
echo "Jupyter did not become ready within 90 minutes." >&2
exit 1
