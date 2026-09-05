#!/usr/bin/env bash
# Standalone pre-launch probe: decide in seconds whether this instance can run
# the pinned NeMo container, before paying for a multi-gigabyte image pull.
#
#   bash launchable/check_driver.sh
#   curl -fsSL <raw-url>/launchable/check_driver.sh | bash
#
# Exit codes: 0 supported, 1 driver too old, 2 no usable NVIDIA stack.
set -uo pipefail

IMAGE="${NEMOTRON_CONTAINER_IMAGE:-nvcr.io/nvidia/nemo:26.08}"
MINIMUM_DRIVER_VERSION="${NEMOTRON_MINIMUM_DRIVER_VERSION:-580.65.06}"
NATIVE_DRIVER_VERSION="610.43.02"

version_at_least() {
  awk -v actual="$1" -v required="$2" 'BEGIN {
    actual_count = split(actual, a, /[^0-9]+/)
    required_count = split(required, r, /[^0-9]+/)
    count = (actual_count > required_count ? actual_count : required_count)
    for (i = 1; i <= count; i++) {
      left = (i <= actual_count ? a[i] + 0 : 0)
      right = (i <= required_count ? r[i] + 0 : 0)
      if (left > right) exit 0
      if (left < right) exit 1
    }
    exit 0
  }'
}

echo "Pinned container: ${IMAGE}"
echo "Required host driver: ${MINIMUM_DRIVER_VERSION}+ (CUDA 13.x), ${NATIVE_DRIVER_VERSION}+ native"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi is not on PATH, so no NVIDIA driver is installed or exposed." >&2
  echo "Pick a Brev image that already ships a GPU driver; this lab does not install one on create." >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv; then
  echo "FAIL: nvidia-smi is present but cannot talk to the driver." >&2
  exit 2
fi
echo

# A while-read loop keeps this probe runnable on bash 3.2 hosts and laptops,
# where mapfile does not exist.
DRIVER_VERSIONS=()
while IFS= read -r REPORTED_VERSION; do
  [ -n "${REPORTED_VERSION}" ] && DRIVER_VERSIONS+=("${REPORTED_VERSION}")
done < <(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)
if [ "${#DRIVER_VERSIONS[@]}" -eq 0 ]; then
  echo "FAIL: no driver version was reported." >&2
  exit 2
fi

STATUS=0
for DRIVER_VERSION in "${DRIVER_VERSIONS[@]}"; do
  if version_at_least "${DRIVER_VERSION}" "${NATIVE_DRIVER_VERSION}"; then
    echo "OK: driver ${DRIVER_VERSION} runs ${IMAGE} natively."
  elif version_at_least "${DRIVER_VERSION}" "${MINIMUM_DRIVER_VERSION}"; then
    echo "OK: driver ${DRIVER_VERSION} runs ${IMAGE} through CUDA 13.x minor-version/forward compatibility."
  else
    echo "FAIL: driver ${DRIVER_VERSION} is below ${MINIMUM_DRIVER_VERSION} and cannot run ${IMAGE}." >&2
    STATUS=1
  fi
done

if [ "${STATUS}" -ne 0 ]; then
  cat >&2 <<'GUIDANCE'

This is an image/driver failure, not a GPU capability failure. NVIDIA's CUDA
forward-compatibility package cannot close this gap either: a CUDA 13.x compat
package still requires a base driver of 580 or newer, and it applies only to
data-center GPUs and select NGC-Server-Ready RTX SKUs.

Options, cheapest first:
  1. Destroy this instance and create one whose base image reports 580.65.06+.
     Driver version follows the provider/image, not the GPU model: this
     repository has seen a Brev A100 80 GB image at 565.57.01 and an RTX PRO
     6000 Blackwell image at 595.91.07. Try another region, provider, or image.
  2. Upgrade in place, then reboot:  sudo bash launchable/upgrade_driver.sh
     A reboot is unavoidable, so this cannot run inside the Brev on-create
     script. Re-run launchable/setup.sh after the instance comes back; it is
     idempotent and will resume.
  3. Ask your Brev/cloud administrator for an image family on a 580+ branch.

Do not substitute an older NeMo image: it is not the verified dependency stack
for these training recipes.
GUIDANCE
  exit 1
fi

echo
if ! command -v docker >/dev/null 2>&1; then
  echo "WARN: Docker is not on PATH. The driver is fine, but setup.sh needs Docker plus the NVIDIA Container Toolkit." >&2
elif ! docker info >/dev/null 2>&1; then
  echo "WARN: Docker is installed but the daemon is unreachable for this user. Run 'docker info' for the host error." >&2
else
  echo "OK: Docker daemon is reachable."
fi
echo "Driver gate passed. Run launchable/setup.sh next."
