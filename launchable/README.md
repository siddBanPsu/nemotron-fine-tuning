# Brev Launchable setup

This directory starts an isolated NeMo 26.08 container and JupyterLab for the
four-notebook Nemotron 3.5 Lightning Text2SQL lab. The repository remains the
source of notebook code; model weights, BIRD databases, caches, reports, and
checkpoints live on the configured data volume.

## Recommended Brev configuration

- Mode: VM
- GPU: 1×H100 80 GB for the normal workshop; 2×A100/H100 80 GB is a faster
  expert-parallel option for LoRA
- Host RAM: 128 GB minimum
- Disk: 300 GB minimum
- Base driver: 580.65.06 or newer; 610.43.02+ is native for the pinned CUDA stack
- Secure Link: host port 8889
- Public TCP/UDP ports: none
- On-create script: the contents of `launchable/setup.sh`

The setup script checks the driver before pulling the large image, runs real
CUDA arithmetic inside `nvcr.io/nvidia/nemo:26.08`, pins Megatron-Bridge, and
starts Jupyter on container port 8888 mapped to VM loopback port 8889. The Brev
Secure Link must point to 8889.

## Launch parameters

Use these environment variables in Brev when needed:

```bash
NEMOTRON_PREFETCH_MODEL=0
NEMOTRON_JUPYTER_PORT=8889
NEMOTRON_REPOSITORY_REF=main
NEMOTRON_REPOSITORY_MODE=auto
```

Set `NEMOTRON_PREFETCH_MODEL=1` for a scheduled workshop. It downloads the
pinned ~62 GB BF16 snapshot and creates the reusable Megatron checkpoint before
Jupyter opens. Leave it at `0` for an immediate Notebook 01 start.

`NEMOTRON_REPOSITORY_MODE=auto` uses the surrounding checkout only when Docker
can safely bind it; otherwise it clones the requested ref under the data root.
The clone is intentionally detached at the fetched commit. To update it, rerun
setup with a newer `NEMOTRON_REPOSITORY_REF`; do not run a branchless `git pull`
inside the container.

## Persistent storage

On Brev, point `NEMOTRON_DATA_ROOT` at the instance's persistent large volume.
On a standalone VM, first verify that the selected mount and Docker data root
have capacity:

```bash
df -hT /path/to/large-volume /var/lib/docker
export NEMOTRON_DATA_ROOT=/path/to/large-volume/nemotron-fine-tuning
bash launchable/setup.sh
```

The root receives separate `storage`, `huggingface`, `artifacts`, `cache`, `tmp`,
and (when bootstrapped) `repository` directories. It therefore covers the model,
converted checkpoint, LoRA adapter, merged checkpoint, the ~800 MB compressed
Mini-Dev download and extracted SQLite databases, vLLM/torch caches, and reports.
Docker image layers remain in Docker's daemon data root.

## Open Jupyter

When setup prints that Jupyter is ready, open the Brev-authenticated Secure Link
for port 8889. On a standalone VM, forward the loopback port:

```bash
ssh -N -L 8889:127.0.0.1:8889 USER@VM_HOST
```

Then open:

```text
http://127.0.0.1:8889/lab/tree/notebooks/01_cloud_api_baseline.ipynb
```

If the link is not ready despite the setup message, inspect the container rather
than starting a second Jupyter process:

```bash
docker ps --filter name=nemotron-35-ft-lab
docker logs --tail 200 nemotron-35-ft-lab
curl -v http://127.0.0.1:8889/api
```

## Notebook boundaries

- Notebook 01 is CPU/API-only. It downloads BIRD Mini-Dev and makes 50 hosted
  requests by default (25 per model).
- Notebook 02 loads local BF16 through an isolated vLLM subprocess and saves the
  100-row execution baseline; the process exits to release GPU memory.
- Notebook 03 uses all visible GPUs by default for TP1/expert parallelism,
  trains at most 64 packed steps, merges, evaluates through vLLM, and computes
  the paired execution delta.
- Notebook 04 is design-only and never launches full SFT.

The host `.venv` deliberately contains no PyTorch. GPU notebooks must use the
container kernel reached through port 8889. Installing `torch` into the host
environment does not repair a wrong-kernel problem.

## Credentials

Prefer `NVIDIA_API_KEY` and optional `HF_TOKEN` in the process environment. Do
not paste secrets into the Launchable manifest, repository, TOML config, setup
script, notebook output, or evaluation artifacts. Notebook 01 can accept an API
key through native notebook input and clears that prompt after connection.

## Driver/runtime failures

The host NVIDIA driver cannot be upgraded from inside the lifecycle container.
If setup rejects an older driver, choose a newer Brev base image/instance. R580
through R609 can use CUDA 13.x minor-version compatibility only if the actual
container smoke test succeeds. `docker login` proves registry access, not GPU
runtime readiness.
