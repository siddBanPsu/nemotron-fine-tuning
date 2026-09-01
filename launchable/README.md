# Create the Brev Launchable

The repository contains the versioned setup. Brev Console owns the shareable
Launchable object.

1. Confirm `brev-launchable.yaml` points to
   `https://github.com/siddBanPsu/nemotron-fine-tuning`.
2. In **Brev → Launchables → Create Launchable**, choose VM mode and paste
   `setup.sh` into the setup-script field.
3. Notebook 01 can run on any Python 3.12 CPU host. Use one H100 80 GB, at least
   128 GB host RAM, and 300 GB disk for the Notebook 02–03 Brev workshop.
   Notebook 04 is design-only and does not require a multi-GPU Launchable.
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
