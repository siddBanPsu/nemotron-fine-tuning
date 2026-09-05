# Brev Launchable setup

This Launchable starts an isolated NeMo 26.08 container and JupyterLab for the
five-notebook executable Text2SQL lab. The default is the smaller one-GPU Nano
9B v2 workshop; Lightning 3.5 remains an explicit two-GPU advanced path.

## Choose the profile before creating the instance

| Profile | Recommended Brev hardware | Notebook | Honest duration status |
| --- | --- | --- | --- |
| `nano9b_workshop` | 1×RTX PRO 6000 Blackwell Server Edition 96 GB; or 1×A100/H100 80 GB with driver 580.65.06+, 128 GB RAM, 200 GB disk | 03 | warm-cache under-one-hour target; target-SKU rehearsal still required |
| `lightning35_advanced` | 2×A100/H100 80 GB, 192 GB RAM, 300 GB disk | 04 | measured 6 h 49 min warm-cache on 2×A100 for train, merge, and evaluation |

A 48 GB L40S passes the conservative Nano software gate but remains
experimental because the pinned Megatron-Bridge recipe is specifically a
one-H100 BF16 recipe. Do not schedule a workshop on 48 GB without a full
rehearsal.

Choose the VM image by driver as well as GPU. A Brev RTX PRO 6000 Blackwell
Server Edition instance with driver 595.91.07 completed this repository's NeMo
26.08 container and CUDA smoke test using forward compatibility. This is setup
evidence only, not a measured Nano training run. An observed Brev A100 80 GB
image with driver 565.57.01 failed correctly: its VRAM and compute capability
are sufficient, but that VM image cannot initialize the pinned CUDA 13 stack.

Nano v2 is intentionally compatibility-frozen: the pinned Megatron-Bridge
commit contains its one-GPU recipe, while current upstream documentation marks
the model family's Bridge support deprecated. The profile is useful for this
bounded workshop, not a promise of future recipe maintenance.

## Screen the instance before spending on it

```bash
bash launchable/check_driver.sh
```

This standalone probe needs no Python, no repository install, and no image
pull. It prints every GPU, its driver, and a verdict, then exits 0 (supported),
1 (driver too old), or 2 (no usable NVIDIA stack). Run it right after SSH, or
pipe it from the raw file URL, before running setup.

## Launch parameters

For the recommended workshop:

```bash
export NEMOTRON_MODEL_PROFILE=nano9b_workshop
export NEMOTRON_PREFETCH_MODEL=1
export NEMOTRON_JUPYTER_PORT=8889
export NEMOTRON_REPOSITORY_REF=main
export NEMOTRON_REPOSITORY_MODE=auto
bash launchable/setup.sh
```

For Lightning, change only the model profile before setup:

```bash
export NEMOTRON_MODEL_PROFILE=lightning35_advanced
export NEMOTRON_PREFETCH_MODEL=1
bash launchable/setup.sh
```

Prefetch `1` downloads the selected pinned Hugging Face snapshot and converts it
to a reusable Megatron checkpoint before Jupyter opens. Use `0` for an immediate
API-first start; the same work then happens in Notebook 03 or 04.

`NEMOTRON_REPOSITORY_MODE=auto` uses the surrounding checkout only when Docker
can safely bind it; otherwise it clones the requested ref under the data root.
The clone is deliberately detached at the fetched commit. To update it, rerun
setup with a newer `NEMOTRON_REPOSITORY_REF`; do not run a branchless `git pull`
inside the container.

## What setup validates

The script checks the NVIDIA driver before pulling the image, runs real CUDA
arithmetic inside `nvcr.io/nvidia/nemo:26.08`, pins Megatron-Bridge at
`8bc33cd2ca1cd044e1520130f4c5e1f5e0181434`, verifies both Nano and Lightning
recipes import, resolves the selected model profile, and starts Jupyter on
container port 8888 mapped to VM loopback port 8889.

Use a base driver at least 580.65.06 for CUDA 13.x minor-version compatibility;
610.43.02+ is native for the pinned image. The Brev Secure Link must point to
host port 8889. Do not expose the tokenless container Jupyter port publicly.

## Persistent storage

Point `NEMOTRON_DATA_ROOT` at the instance's large persistent volume. On a
standalone VM, verify both that mount and Docker's own data root:

```bash
df -hT /path/to/large-volume /var/lib/docker
export NEMOTRON_DATA_ROOT=/path/to/large-volume/nemotron-fine-tuning
export NEMOTRON_MODEL_PROFILE=nano9b_workshop
export NEMOTRON_PREFETCH_MODEL=1
bash launchable/setup.sh
```

The root receives `storage`, `huggingface`, `artifacts`, `cache`, `tmp`, and,
when bootstrapped, `repository`. This covers model snapshots, converted bases,
LoRA adapters, merged exports, the BIRD package and databases, compilation
caches, reports, and temporary files. Docker image layers remain in Docker's
daemon data root.

`/tmp` is safe only on a disposable VM where it is a large disk rather than
RAM-backed `tmpfs`. Copy reports and checkpoints to persistent storage before
shutdown.

## Open and diagnose Jupyter

When setup reports readiness, open the Brev-authenticated Secure Link for port
8889. On a standalone VM:

```bash
ssh -N -L 8889:127.0.0.1:8889 USER@VM_HOST
```

Then open:

```text
http://127.0.0.1:8889/lab/tree/notebooks/01_cloud_api_baseline.ipynb
```

If the link is not ready, inspect the existing container rather than starting a
second Jupyter process:

```bash
docker ps --filter name=nemotron-35-ft-lab
docker logs --tail 200 nemotron-35-ft-lab
curl -v http://127.0.0.1:8889/api
```

## Notebook boundaries

- Notebook 01 is CPU/API-only and makes 50 hosted requests by default.
- Notebook 02 creates a profile-namespaced 100-row local BF16 baseline. Run it
  once with Nano before Notebook 03, or set `MODEL_PROFILE_NAME` to
  `lightning35_advanced` before Notebook 04.
- Notebook 03 is the one-GPU Nano workshop. `WORKSHOP_MODE = True` is the
  default: 1,024 direct rows and at most 16 packed steps. Set it to `False` for
  the extended 2,048-row, 32-step schedule. The modes use separate data,
  checkpoint, merged-model, and report paths.
- Notebook 04 is the two-GPU Lightning advanced path: 4,096 direct/reasoning
  rows and at most 64 steps. It is a multi-hour A100 exercise.
- Notebook 05 is design-only and never launches full SFT.

Nano checkpoints save every eight steps. A rerun resumes only when the existing
run contract exactly matches; otherwise the driver tells you to choose a new
output directory. Lightning retains final-step checkpointing. A non-zero
subprocess exit always means the stage is incomplete, regardless of GPU logs.

The host `.venv` deliberately contains no PyTorch. GPU notebooks must use the
container kernel reached through port 8889; installing `torch` on the host does
not repair a wrong-kernel problem.

## When the Brev on-create service fails immediately

`Job for instance-oneshot.service failed` means Brev's lifecycle unit exited
non-zero; the reason is in the journal, not in the bootstrap output:

```bash
sudo journalctl -xeu instance-oneshot.service --no-pager | tail -100
sudo systemctl status instance-oneshot.service --no-pager
```

A failure within seconds is usually one of: no GPU driver yet, Docker not
ready, or a rejected launch parameter.

### Observed: Brev's own bootstrap fails before this launchable runs

On one Brev instance the unit failed with `NVIDIA driver installation failed`,
but that message was misleading. The journal showed Brev's bootstrap restarting
`docker.service` twice within a second after configuring the NVIDIA runtime;
the second restart failed, and its error handler reported it as a driver
problem. The driver (580.173.02) and `/etc/docker/daemon.json` were both fine —
`dockerd --validate` returned `configuration OK` and Docker started normally
afterwards. Repeated rapid restarts trip systemd's start rate limiter.

Confirm which layer actually failed before blaming the GPU:

```bash
sudo journalctl -xeu docker.service --no-pager | tail -60
sudo dockerd --validate --config-file /etc/docker/daemon.json
nvidia-smi
```

When Docker and the driver are healthy, the instance needs no recreation. Brev's
oneshot failing only means this repository's setup never started, so run it
yourself:

```bash
sudo systemctl reset-failed instance-oneshot.service
bash launchable/check_driver.sh
export NEMOTRON_DATA_ROOT=/path/to/large/persistent/volume/nemotron-fine-tuning
bash launchable/setup.sh
```

If Docker itself is rate-limited rather than misconfigured:

```bash
sudo systemctl reset-failed docker.service && sudo systemctl start docker.service
```

setup.sh also waits up to `NEMOTRON_DRIVER_WAIT_SECONDS` for the daemon to
answer, so it survives a restart that is still in progress. The driver case is handled explicitly —
setup waits up to `NEMOTRON_DRIVER_WAIT_SECONDS` (default 300) for `nvidia-smi`
to answer, because a new VM often finishes loading its driver after the
lifecycle service starts, then fails with a specific message rather than a bare
crash.

Setup is idempotent, so after fixing the cause you can rerun it over SSH
without recreating the instance:

```bash
bash launchable/check_driver.sh && bash launchable/setup.sh
```

## Credentials and runtime failures

Prefer `NVIDIA_API_KEY` and optional `HF_TOKEN` in the process environment.
Never put secrets in the manifest, repository, TOML, setup script, notebook
output, or evaluation artifacts.

The host NVIDIA driver cannot be upgraded from inside the lifecycle container.
Although an administrator can replace a driver on some raw VMs, activating its
kernel module requires a reboot; doing that from Brev's on-create service marks
the build failed and can create a retry loop. If setup rejects an older driver,
prefer a provider/base image that already reports 580.65.06+ in `nvidia-smi`.
Do not lower `NEMOTRON_MINIMUM_DRIVER_VERSION`: the subsequent CUDA smoke test
will still fail on an incompatible driver.

Forward compatibility is not an escape hatch below that floor. A CUDA 13.x
`cuda-compat` package still requires a base driver of 580 or newer, and it is
supported only on data-center GPUs and select NGC-Server-Ready RTX SKUs. Within
the supported band, setup makes it robust: if the NGC image does not activate
compat itself, setup retries the CUDA smoke test with
`LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:...` and, when that succeeds,
starts Jupyter with the same loader path.

When recreating the instance is not an option, upgrade it in place over SSH:

```bash
sudo NEMOTRON_CONFIRM_DRIVER_UPGRADE=1 bash launchable/upgrade_driver.sh
sudo reboot
bash launchable/check_driver.sh && bash launchable/setup.sh
```

`upgrade_driver.sh` installs `cuda-drivers-580` from NVIDIA's repository on
Ubuntu 22.04/24.04 (x86_64 or aarch64), refuses to run without the explicit
confirmation variable, and requires a reboot. Override the branch or package
with `NEMOTRON_DRIVER_BRANCH` or `NEMOTRON_DRIVER_PACKAGE`. It replaces a
kernel driver, so never run it mid-workshop.
Docker registry access, Docker daemon access, NVIDIA container runtime support,
and in-container CUDA initialization are separate checks.
