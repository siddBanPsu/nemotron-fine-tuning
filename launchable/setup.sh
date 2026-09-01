#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="nemotron-35-ft-lab"
IMAGE="nvcr.io/nvidia/nemo:26.08"
JUPYTER_PORT="${NEMOTRON_JUPYTER_PORT:-8889}"
PREFETCH_MODEL="${NEMOTRON_PREFETCH_MODEL:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

echo "[1/5] Checking GPU and container runtime"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv
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

echo "[2/5] Preparing persistent model/checkpoint storage"
mkdir -p "${STORAGE_DIR}" "${HF_CACHE_DIR}"

echo "[3/5] Pulling the pinned NeMo container"
retry docker pull "${IMAGE}"

echo "[4/5] Starting the isolated Jupyter lab container"
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

echo "[5/5] Waiting for Jupyter readiness"
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
