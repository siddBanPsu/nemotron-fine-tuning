# Brev Launchable setup

This Launchable starts an isolated NeMo 26.08 container and JupyterLab for the
five-notebook executable Text2SQL lab. The default is the smaller one-GPU Nano
9B v2 workshop; Lightning 3.5 remains an explicit two-GPU advanced path.

## Choose the profile before creating the instance

| Profile | Recommended Brev hardware | Notebook | Honest duration status |
| --- | --- | --- | --- |
| `nano9b_workshop` | 1×A100 80 GB or 1×H100 80 GB, 128 GB RAM, 200 GB disk | 03 | warm-cache under-one-hour target; target-SKU rehearsal still required |
| `lightning35_advanced` | 2×A100/H100 80 GB, 192 GB RAM, 300 GB disk | 04 | measured 6 h 49 min warm-cache on 2×A100 for train, merge, and evaluation |

A 48 GB L40S passes the conservative Nano software gate but remains
experimental because the pinned Megatron-Bridge recipe is specifically a
one-H100 BF16 recipe. Do not schedule a workshop on 48 GB without a full
rehearsal.

Nano v2 is intentionally compatibility-frozen: the pinned Megatron-Bridge
commit contains its one-GPU recipe, while current upstream documentation marks
the model family's Bridge support deprecated. The profile is useful for this
bounded workshop, not a promise of future recipe maintenance.

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
- Notebook 03 is the one-GPU Nano workshop: 2,048 direct rows, at most 32 packed
  steps, CPU merge, local evaluation, and a paired accuracy interval.
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

## Credentials and runtime failures

Prefer `NVIDIA_API_KEY` and optional `HF_TOKEN` in the process environment.
Never put secrets in the manifest, repository, TOML, setup script, notebook
output, or evaluation artifacts.

The host NVIDIA driver cannot be upgraded from inside the lifecycle container.
If setup rejects an older driver, choose a newer Brev base image or instance.
Docker registry access, Docker daemon access, NVIDIA container runtime support,
and in-container CUDA initialization are separate checks.
