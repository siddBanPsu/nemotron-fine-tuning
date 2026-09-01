#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="nemotron-35-ft-lab"
IMAGE="${NEMOTRON_CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.08}"
MINIMUM_DRIVER_VERSION="${NEMOTRON_MINIMUM_DRIVER_VERSION:-610.43}"
JUPYTER_PORT="${NEMOTRON_JUPYTER_PORT:-8889}"
PREFETCH_MODEL="${NEMOTRON_PREFETCH_MODEL:-0}"
REPOSITORY_URL="${NEMOTRON_REPOSITORY_URL:-https://github.com/siddBanPsu/nemotron-fine-tuning.git}"
REPOSITORY_REF="${NEMOTRON_REPOSITORY_REF:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPOSITORY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BOOTSTRAP_REPOSITORY_DIR="${HOME}/nemotron-fine-tuning-launchable"
REPOSITORY_DIR=""
STORAGE_DIR="${HOME}/nemotron-35-ft-storage"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

case "${PREFETCH_MODEL}" in
  0|1) ;;
  *)
    echo "NEMOTRON_PREFETCH_MODEL must be 0 or 1; received '${PREFETCH_MODEL}'." >&2
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
  if [ -f "${LOCAL_REPOSITORY_DIR}/launchable/container-entrypoint.sh" ]; then
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

echo "[1/6] Checking GPU, driver, and container runtime"
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
    echo "This pinned Nemotron 3.5 training environment requires driver ${MINIMUM_DRIVER_VERSION} or newer." >&2
    echo "Create a fresh Brev instance whose base image reports driver ${MINIMUM_DRIVER_VERSION}+; the setup stops before the large container pull." >&2
    echo "Do not substitute an older NeMo image: it is not the verified dependency stack for these training recipes." >&2
    exit 1
  fi
done
docker info >/dev/null
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Replacing existing ${CONTAINER_NAME} container"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi
if ! port_is_free; then
  echo "Host port 127.0.0.1:${JUPYTER_PORT} is already in use." >&2
  echo "Set NEMOTRON_JUPYTER_PORT to a free port and configure the Brev Secure Link to the same port." >&2
  exit 1
fi

echo "[2/6] Resolving the versioned repository"
resolve_repository

echo "[3/6] Preparing persistent model/checkpoint storage"
mkdir -p "${STORAGE_DIR}" "${HF_CACHE_DIR}"

echo "[4/6] Pulling the pinned NeMo container"
retry docker pull "${IMAGE}"

echo "[5/6] Starting the isolated Jupyter lab container"
docker run --detach \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "127.0.0.1:${JUPYTER_PORT}:8888" \
  -e "NEMOTRON_PREFETCH_MODEL=${PREFETCH_MODEL}" \
  -e "HF_HOME=/root/.cache/huggingface" \
  -e "PYTHONPATH=/workspace/launchable/src" \
  -v "${REPOSITORY_DIR}:/workspace/launchable" \
  -v "${STORAGE_DIR}:/workspace/storage" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -w /workspace/launchable \
  "${IMAGE}" \
  bash /workspace/launchable/launchable/container-entrypoint.sh

echo "[6/6] Waiting for Jupyter readiness"
for _ in $(seq 1 1080); do
  if curl --fail --silent "http://127.0.0.1:${JUPYTER_PORT}/api" >/dev/null; then
    echo "Ready: open the Brev Secure Link on port ${JUPYTER_PORT}."
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
