#!/usr/bin/env bash
# Opt-in, operator-run driver upgrade for an instance that failed the driver
# gate. This CANNOT run from the Brev on-create lifecycle script: activating a
# new kernel module requires a reboot, which would fail that run.
#
#   sudo NEMOTRON_CONFIRM_DRIVER_UPGRADE=1 bash launchable/upgrade_driver.sh
#
# It replaces the host NVIDIA driver and then requires a reboot. Do not run it
# on an instance whose GPU state you still need, and never mid-workshop.
set -euo pipefail

DRIVER_BRANCH="${NEMOTRON_DRIVER_BRANCH:-580}"
DRIVER_PACKAGE="${NEMOTRON_DRIVER_PACKAGE:-cuda-drivers-${DRIVER_BRANCH}}"
REBOOT_AFTERWARDS="${NEMOTRON_REBOOT_AFTER_DRIVER_UPGRADE:-0}"

if [ "${NEMOTRON_CONFIRM_DRIVER_UPGRADE:-0}" != "1" ]; then
  cat >&2 <<'REFUSAL'
Refusing to change the host NVIDIA driver without explicit confirmation.

This replaces a kernel driver and needs a reboot; any running GPU work dies.
Re-run with NEMOTRON_CONFIRM_DRIVER_UPGRADE=1 once you are certain, or simply
create a fresh instance whose image already reports driver 580.65.06+, which is
usually faster and always less risky.
REFUSAL
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this script as root (sudo)." >&2
  exit 1
fi

if [ ! -r /etc/os-release ]; then
  echo "Cannot identify the distribution: /etc/os-release is unreadable." >&2
  exit 1
fi
# shellcheck disable=SC1091
. /etc/os-release
ARCHITECTURE="$(uname -m)"
case "${ID:-}:${VERSION_ID:-}:${ARCHITECTURE}" in
  ubuntu:22.04:x86_64) REPOSITORY_SLUG="ubuntu2204/x86_64" ;;
  ubuntu:24.04:x86_64) REPOSITORY_SLUG="ubuntu2404/x86_64" ;;
  ubuntu:22.04:aarch64) REPOSITORY_SLUG="ubuntu2204/sbsa" ;;
  ubuntu:24.04:aarch64) REPOSITORY_SLUG="ubuntu2404/sbsa" ;;
  *)
    echo "Unsupported platform for automated upgrade: ${ID:-unknown} ${VERSION_ID:-unknown} ${ARCHITECTURE}." >&2
    echo "Install a ${DRIVER_BRANCH}+ data-center driver using your distribution's documented method, then reboot." >&2
    exit 1
    ;;
esac

echo "Current driver:"
nvidia-smi --query-gpu=name,driver_version --format=csv || echo "  (nvidia-smi unavailable)"
echo "Target: ${DRIVER_PACKAGE} from the NVIDIA CUDA repository for ${REPOSITORY_SLUG}"

export DEBIAN_FRONTEND=noninteractive
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${REPOSITORY_SLUG}/cuda-keyring_1.1-1_all.deb"
KEYRING_DEB="$(mktemp /tmp/cuda-keyring.XXXXXX.deb)"
trap 'rm -f "${KEYRING_DEB}"' EXIT

echo "[1/4] Adding the NVIDIA CUDA repository"
curl -fsSL "${KEYRING_URL}" -o "${KEYRING_DEB}"
dpkg -i "${KEYRING_DEB}"
apt-get update

echo "[2/4] Installing ${DRIVER_PACKAGE}"
if ! apt-get install -y "${DRIVER_PACKAGE}"; then
  echo "Installation of ${DRIVER_PACKAGE} failed." >&2
  echo "Check 'apt-cache search cuda-drivers' for the branches this repository offers, then set NEMOTRON_DRIVER_PACKAGE." >&2
  exit 1
fi

echo "[3/4] Verifying the installed package version"
dpkg -l | grep -E 'cuda-drivers|nvidia-driver' || true

echo "[4/4] Reboot required"
echo "The new driver is installed but the running kernel still holds the old module."
echo "After the reboot, confirm and continue with:"
echo "  bash launchable/check_driver.sh && bash launchable/setup.sh"
if [ "${REBOOT_AFTERWARDS}" = "1" ]; then
  echo "Rebooting now because NEMOTRON_REBOOT_AFTER_DRIVER_UPGRADE=1."
  systemctl reboot
else
  echo "Reboot when ready:  sudo reboot"
fi
