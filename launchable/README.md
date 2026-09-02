# Create the Brev Launchable

The repository contains the versioned setup. Brev Console owns the shareable
Launchable object.

1. Confirm `brev-launchable.yaml` points to
   `https://github.com/siddBanPsu/nemotron-fine-tuning`.
2. In **Brev → Launchables → Create Launchable**, choose VM mode and paste
   `setup.sh` into the setup-script field.
3. Notebook 01 can run on any Python 3.12 CPU host. Use one H100 80 GB, at least
   128 GB host RAM, and 300 GB disk for the Notebook 02–03 Brev workshop.
   Select a current base image whose `nvidia-smi` reports driver **580.65.06 or
   newer**. Drivers below 610.43.02 use CUDA 13.x minor-version compatibility;
   setup proves CUDA with an in-container tensor operation before starting
   Jupyter. Notebook 04 is design-only and does not require a multi-GPU
   Launchable.
4. Add a Secure Link named `jupyter` on host port 8889 and use it as the CTA.
   Port 8888 is commonly occupied by Brev-managed Jupyter. Do not add either
   port to the public TCP/UDP port list.
5. Leave `NEMOTRON_PREFETCH_MODEL=0` for an immediate API-first start. Set it to
   `1` for a scheduled GPU workshop so the pinned BF16 snapshot and reusable
   Megatron checkpoint are ready before Jupyter opens. Cold provisioning can
   take longer than the lab itself, but participant time becomes predictable.
6. Notebook 01 requires an NVIDIA API key. Prefer the process environment, or
   paste it into Jupyter's native `input()` prompt and press **Enter**. The
   prompt is briefly visible and then clears; do not put the key in Launchable
   parameters. The
   public Hugging Face inputs normally work anonymously, so `HF_TOKEN` is
   optional and must likewise remain outside the manifest and repository.
7. The hosted evaluator is deliberately paced at 30 RPM under the public
   40-RPM quota. Each response is checkpointed; rerunning the evaluation cell
   resumes the same condition rather than spending requests again.

The container still runs Jupyter on port 8888 internally; `setup.sh` maps the
dedicated host port 8889 to it. If 8889 is occupied, choose another unprivileged
`NEMOTRON_JUPYTER_PORT` and update the Brev Secure Link to that same host port.
The setup preflight reports this before attempting `docker run`.

Use Brev's custom Secure Link on host port 8889 for Notebooks 02–03. The
Brev-managed Jupyter service on host port 8888 can open the checkout, but its
host `.venv` is API-only and intentionally does not install PyTorch or the
Megatron training stack.

Brev executes a pasted lifecycle script outside the repository checkout. The
setup therefore clones the public repository into a dedicated directory,
fetches `NEMOTRON_REPOSITORY_REF` (default `main`), checks out the exact fetched
commit, and prints that commit before mounting it into the container. When the
same script is run from a normal checkout, it uses that checkout directly.

The GPU preflight also stops before the large container download when the host
driver is older than 580.65.06. A host reporting driver 565.57.01 cannot run
CUDA 13.x, but an A100 on R595 is eligible for documented minor-version
compatibility. After pulling the image, setup performs a real PyTorch CUDA
tensor operation and refuses to start Jupyter if compatibility is not working.
This proves basic CUDA initialization, not the complete PEFT exercise; rehearse
the exact A100 profile before a workshop.

For private endpoint testing, copy `config/api.local.toml.example` to the
Git-ignored `config/api.local.toml` and fill in the private URL, model IDs, and
rate limit. The public Launchable must not include that local file. Set
`NEMOTRON_API_PROFILE=public` when rehearsing the exact public experience from
a checkout that also contains a private profile. Within Notebook 01, set
`API_PROFILE_OVERRIDE` to `"public"` or `"local"`, then rerun that configuration
cell and the Section 2 authentication cell; no Jupyter restart is required.

The setup launches `nvcr.io/nvidia/nemo:26.08`, persists Hugging Face downloads
and checkpoints outside the container, pins the NVIDIA Megatron-Bridge source
used to construct the notebooks, optionally performs the one-time checkpoint
conversion, and exposes only Jupyter through Brev's authenticated Secure Link.

## Running the same setup on a standalone VM

The host `.venv` is intentionally API-only. Do not run `uv pip install torch`
there: it duplicates the large CUDA/NeMo dependency stack and can exhaust the
VM home filesystem. PyTorch is already installed in the pinned NeMo image;
setup prints its version and proves CUDA before starting Jupyter. Use that
container for Notebooks 02–03.

When the VM has a large mounted data disk, route model weights, checkpoints,
notebook artifacts, caches, and temporary files to it with one setting (replace
the example path with the real absolute mount):

```bash
df -hT "$HOME" /var/lib/docker /path/to/large-volume
export NEMOTRON_DATA_ROOT=/path/to/large-volume/nemotron-fine-tuning
export NEMOTRON_PREFETCH_MODEL=0
bash launchable/setup.sh
```

For a short-lived non-Brev VM where `/tmp` is confirmed to be a disk with enough
space, the equivalent opt-in is:

```bash
findmnt -T /tmp
command -v docker
docker info --format 'Docker root: {{.DockerRootDir}}'
export NEMOTRON_DATA_ROOT=/tmp/nemotron-fine-tuning
export NEMOTRON_PREFETCH_MODEL=0
bash launchable/setup.sh
```

Start with prefetch disabled to verify the container and Jupyter path. Set
`NEMOTRON_PREFETCH_MODEL=1` on the next run when the data volume and Docker data
root have enough free space. The setup prints every selected runtime path and
its filesystem capacity before pulling the container. Docker image layers still
use the Docker daemon's data root and cannot be redirected by this setting.
For a remote VM,
keep port 8889 bound to loopback and access it through SSH forwarding:

```bash
ssh -L 8889:127.0.0.1:8889 USER@VM_HOST
```

The training code already uses the NeMo Megatron-Bridge SDK. Notebook 01 alone
uses a hosted API so local Lightning fine-tuning can be compared with hosted
Lightning and Ultra without serving those large baseline models on the
single-GPU VM.

Brev documents `/tmp` as non-persistent across a stop. Keep the public
Launchable default on its provisioned persistent disk/workspace; use `/tmp`
only for a disposable rehearsal and export any artifacts before stopping it.

## Rehearsal gate

Local structure checks do not prove API or GPU execution. Before a workshop,
run Notebook 01 against both current trial endpoints, then create a fresh target
GPU instance and run Notebooks 02–03 end to end. Record:

- cold setup and model-download time;
- hosted Lightning NVFP4, hosted Ultra NVFP4, prompt mode, and local BF16 sample counts and accuracy;
- conversion, training, merge, and tuned-evaluation wall times;
- peak GPU memory and final training loss;
- tuned-minus-local-baseline exact accuracy on the unchanged test IDs;
- tuned Lightning minus hosted Ultra accuracy on their exact shared IDs, with
  the paired interval and an explicit better/below/inconclusive conclusion.

Notebook 04 never launches full SFT. If a measured LoRA ceiling justifies an
external full-SFT experiment, rehearse `scripts/train_full.py` on the exact
multi-GPU topology. The driver gates execution below eight GPUs/600 GiB
aggregate VRAM. NVIDIA's public verification card currently validates the 4K
packed configuration on two 8-GPU H100 nodes; this repository's shorter TP1/EP8
profile is structurally derived from the official recipe but remains unverified.
