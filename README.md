# Nemotron 3.5 Lightning fine-tuning lab for Brev

This repository is a four-notebook, API-first and Brev-ready workshop for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`:

1. benchmark hosted Lightning and Nemotron Ultra NVFP4 without a GPU;
2. load the pinned local BF16 checkpoint and freeze the exact tuning baseline;
3. fine-tune with LoRA PEFT on one 80 GB GPU, measure the paired accuracy delta,
   and test whether specialized Lightning reaches or beats hosted Ultra;
4. study true full-parameter SFT as a design-only multi-GPU handoff.

Notebook 01 can run before renting a GPU. The practical Brev path is Notebooks
02–03 on one 80 GB GPU plus Notebook 04's short design exercise. Actual
full-parameter training is not part of the ordinary single-GPU Launchable.

## Why this exercise is worth running

The dataset is [PolyAI BANKING77](https://huggingface.co/datasets/PolyAI/banking77),
a CC BY 4.0 collection of 13,083 online-banking support queries across 77
fine-grained intents. It has an official 10,003-row train split and 3,080-row
test split.

The lab converts each natural label to an opaque internal code such as
`B77_61`. A fixed recorded permutation prevents the public category order from
leaking the mapping. The model is told only to return one valid route code; it
is not given the label-to-code mapping at evaluation time. That represents a useful
enterprise adaptation problem: a capable base model may understand “my card
has not arrived,” but it cannot know the organization's private routing code
until it learns the taxonomy.

The proof is deliberately strict:

- train/validation rows come only from the official train split;
- evaluation rows come only from the official test split;
- deterministic stratified sampling gives every label equal weight;
- the manifest records every selected ID;
- every model is scored on the same frozen prompts;
- exact route-code accuracy and valid-code rate are reported separately;
- tuned-minus-baseline accuracy uses a paired bootstrap confidence interval on
  identical example IDs.

The default cloud profile selects one frozen example from every label and gives
each request five lexically retrieved demonstrations drawn only from the
training split: 77 requests per model, or 154 across Lightning and Ultra. It
does not take the first 77 rows, because prepared rows are grouped by label.
Notebook 01 also implements a complete taxonomy prompt and the original opaque
zero-shot prompt. The `prompt_only` profile runs taxonomy plus retrieval (308
calls); `full` runs all three prompt conditions (462 calls). Changing from one
to three evaluation examples per label multiplies those totals by three. The
local BF16 before/after evaluation remains at 231 rows.

A lower training loss is not counted as higher accuracy. If the saved
held-out score does not improve, the notebook says that no gain was
demonstrated.

## Hardware and honest scope

Nemotron 3.5 Lightning is a hybrid Mamba-2/attention sparse-MoE model with 30B
total parameters and roughly 3B active parameters per token. NVIDIA describes
the BF16 checkpoint as the customization starting point and supports
single-H100/A100-80 inference. Sparse activation lowers compute, but all 30B
weights still exist in memory.

| Path | Suggested hardware | Workshop setting | Status |
| --- | --- | --- | --- |
| Hosted API targets | CPU only plus NVIDIA API key | 154 calls by default; paced at 30 RPM (about 5.2 minutes minimum); 308 for both prompt-only competitors; 462 for the three-condition smoke matrix | No model hosting; trial endpoint availability and limits apply |
| Exact local baseline | 1× H100 80 GB or A100 80 GB | Same 231 IDs using pinned BF16 | Required scientific baseline for the PEFT delta |
| LoRA PEFT | 1× H100 80 GB | 512-token packing, ≤40 steps, rank 16 | Derived from NVIDIA's official single-H100 Megatron-Bridge cookbook |
| Full-SFT design | Any notebook host | Memory arithmetic, topology, handoff command | Does not launch training |
| Full-SFT external target | ≥8× H100 80 GB code gate | 512 tokens, 40 steps, TP1/EP8 | Driver exists; not claimed live-validated here |
| NVIDIA verified full SFT reference | 16× H100 80 GB | 4K packed, 100 steps, TP2/EP8 | NVIDIA verification card reports ~9.9 s/step for its OpenMathInstruct-2 run |

One 80 GB GPU can hold roughly 60 GB of BF16 weights, but full AdamW training
also needs gradients, FP32 master weights and moments, activations, and
workspaces. Notebook 04 never launches training, and `scripts/train_full.py`
hard-fails below 8 GPUs/600 GiB aggregate VRAM so “full fine-tuning” cannot
silently become LoRA.

## Repository layout

```text
notebooks/
  01_cloud_api_baseline.ipynb
  02_local_bf16_baseline.ipynb
  03_peft_lora.ipynb
  04_full_finetuning_design.ipynb
launchable/
  brev-launchable.yaml
  setup.sh
  container-entrypoint.sh
scripts/
  prepare_banking77.py
  convert_checkpoint.py
  train_peft.py
  train_full.py
  export_full_checkpoint.py
  preflight.py
src/nemotron_ft_lab/
tests/
```

Large or generated files go under ignored `artifacts/`, `checkpoints/`,
`storage/`, the Hugging Face cache, or `/workspace/storage` in the container.
No credentials belong in this repository.

## Public and private API profiles

Notebook 01 uses the tracked public endpoint and model identifiers by default.
For an internal or faster OpenAI-compatible endpoint, create the ignored local
configuration:

```bash
cp config/api.local.toml.example config/api.local.toml
```

Edit `config/api.local.toml` with the private base URL, Lightning and Ultra
model identifiers, safe request rate, and timeout. Do not put an API key in the
file; continue to use `NVIDIA_API_KEY`. The notebook auto-selects the local file
when present. Inside Notebook 01, use the single configuration switch and
rerun that cell plus the authentication cell:

```python
API_PROFILE_OVERRIDE = "local"    # internal URL and IDs
# API_PROFILE_OVERRIDE = "public" # tracked public URL and IDs
# API_PROFILE_OVERRIDE = None     # auto: local file if present, otherwise public
```

This does not require a Jupyter restart. For a startup-level public selection,
use:

```bash
NEMOTRON_API_PROFILE=public jupyter lab notebooks/01_cloud_api_baseline.ipynb
```

For a configuration stored elsewhere, set `NEMOTRON_API_CONFIG` to its absolute
or repository-relative path. Individual `NEMOTRON_API_BASE_URL`, model-ID,
model-variant, request-rate, and timeout environment overrides are also
supported; see [.env.example](.env.example). Non-public profiles receive a
separate artifact prefix, so private responses cannot overwrite or resume from
the public baseline. Keep notebook outputs cleared before committing because
runtime output displays the selected endpoint and model identifiers.

## Run Notebook 01 without Brev or a GPU

Notebook 01 needs Python 3.12, network access, a small tokenizer download, and
an NVIDIA API key—but no CUDA runtime or model weights:

```bash
python3.12 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -r requirements-lab.txt --editable .
jupyter lab notebooks/01_cloud_api_baseline.ipynb
```

Prefer exporting `NVIDIA_API_KEY` before starting Jupyter. Alternatively,
Notebook 01 uses Jupyter's native `input()` prompt: paste the key and press
**Enter**. This bypasses the unreliable `ipywidgets` layer in some notebook
frontends. The key is briefly visible during entry, then the prompt is cleared.
Generated API responses and reports remain under ignored `artifacts/`.
The evaluator checkpoints every response, defaults to 30 RPM under the public
40-RPM quota, and automatically resumes after a `429` or notebook interruption.

## Brev quick start

Use [launchable/README.md](launchable/README.md) to create the Brev Console
Launchable. The checked-in manifest recommends:

- VM mode;
- one H100 80 GB, 128 GB host RAM, and 300 GB disk for Notebooks 02–03;
- `nvcr.io/nvidia/nemo:26.08`;
- NVIDIA driver 580.65.06 or newer on the Brev host for CUDA 13.x minor-version
  compatibility; 610.43.02 or newer is the container's native driver level;
- a Brev-authenticated Jupyter Secure Link on host port 8889;
- model prefetch enabled for scheduled workshops, or disabled for an immediate
  API-first start.

The setup script starts an isolated NeMo container, pins Megatron-Bridge source,
bootstraps the public repository when Brev runs the pasted script outside a Git
checkout, mounts persistent cache/checkpoint storage, verifies that both Lightning
training recipes import, and opens Notebook 01. Model prefetch now defaults to
off so the hosted API baseline opens immediately; set
`NEMOTRON_PREFETCH_MODEL=1` for a scheduled workshop to download and convert the
reusable BF16 checkpoint before Jupyter starts. Prefer the process environment
for the API key; Notebook 01 falls back to Jupyter's native `input()` channel
rather than third-party widgets. The prompt is cleared after connection and the
key is never written to evaluation artifacts. The public Hugging Face model and
dataset usually require no secret.

The preflight deliberately exits before pulling the large container if
`nvidia-smi` reports a driver older than 580.65.06. R580 through R609 use CUDA
13.x minor-version compatibility, so setup performs a real PyTorch CUDA tensor
operation inside the pinned container before starting Jupyter. This allows an
A100 on R595 to proceed while still rejecting the incompatible R565 image. An
older NeMo tag is not a drop-in workaround for this repository's pinned
Megatron-Bridge recipe and Python/PyTorch stack.

## Run without Brev

Use a compatible Linux x86_64 NVIDIA GPU host with Docker, NVIDIA Container
Toolkit, enough RAM/disk, and the same NeMo 26.08 container. From the repository
root, the Launchable's container command is the reference. A convenient local
equivalent is:

```bash
NEMOTRON_PREFETCH_MODEL=1 bash launchable/setup.sh
```

Open `http://localhost:8889` only through authenticated SSH forwarding or a
trusted local interface. The setup disables Jupyter's own token because Brev's
Secure Link supplies the access boundary; do not expose host port 8889 or
container port 8888 publicly.

## Expected workshop flow

### Notebook 01 — hosted Lightning and Ultra targets

- run without CUDA or model weights;
- prepare 1,925 training, 385 validation, and 231 held-out test rows by
  default;
- select a correct one-example-per-label smoke set by default (77 rows);
- compare opaque zero-shot, full taxonomy-in-prompt, and training-only
  retrieved-demonstration conditions without test leakage;
- default to the retrieved-demonstration condition so the normal two-model run
  remains 154 calls;
- call NVIDIA's hosted Lightning and Ultra NVFP4 endpoints with thinking disabled;
- checkpoint individual responses for safe resume;
- save model-, prompt-, and sample-profile-specific reports such as
  `baseline_api_ultra_nvfp4_retrieved_few_shot_v1_5d_1_per_label.json`;
- compare Ultra minus Lightning under identical prompt conditions and IDs.

### Notebook 02 — exact local BF16 baseline

- verify the 80 GB GPU and pinned checkpoint revision;
- rebuild or reuse the identical deterministic data bundle;
- inspect an ordinary local BF16 answer;
- evaluate the same 231 IDs;
- save `artifacts/evaluation/baseline_local_bf16.json`.

Repeating the baseline is necessary: comparing a hosted NVFP4 service with a
tuned BF16-derived checkpoint would confound quantization and serving with
fine-tuning.

### Notebook 03 — PEFT

- import the pinned Hugging Face weights once into Megatron format;
- run NVIDIA's model-specific Lightning LoRA recipe;
- train at most 40 packed steps;
- merge the adapter into a normal Hugging Face checkpoint;
- score the identical test IDs and print the absolute accuracy gain;
- save `artifacts/evaluation/peft.json`;
- align tuned Lightning to each cloud report's exact IDs and save
  `artifacts/evaluation/peft_vs_cloud_targets.json`.

The Ultra comparison is deliberately narrow. Beating a 550B/55B-active general
model on an opaque private-routing taxonomy demonstrates the value of
specialization; it does not mean the tuned 30B/3B-active Lightning model is
generally stronger than Ultra. A paired interval that crosses zero is reported
as inconclusive, not as proof of equivalence.

### Notebook 04 — full-SFT design

- calculate why full AdamW state does not fit on one GPU;
- inspect TP/EP/DP topology and all-parameter trainable scope;
- print, but never execute, the minimum multi-GPU handoff command;
- define the evidence contract an external full-SFT run must return.

## Validation

Local validation checks notebook JSON/code syntax, shell syntax, Python syntax,
imports, API resume behavior, data-split helpers, scoring, required files,
secret hygiene, and the full-SFT design-only gate:

```bash
python3 scripts/validate_repo.py
python3 -m unittest discover -s tests -v
```

These checks do not prove target-GPU memory use, runtime, accuracy gain,
checkpoint merge/export, or Brev Secure Link behavior. Record those only after
a fresh live rehearsal.

## Primary references

- [NVIDIA Nemotron 3.5 Lightning BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [NVIDIA hosted Lightning API page](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b)
- [NVIDIA hosted NVFP4 API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b)
- [NVIDIA hosted Nemotron 3 Ultra model card](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard)
- [NVIDIA hosted Nemotron 3 Ultra API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b)
- [NVIDIA Nemotron 3.5 Lightning training recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/lightning35)
- [NVIDIA Megatron-Bridge Lightning recipes and verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/model_verification_cards/nemotron-3.5-lightning)
- [NVIDIA Lightning Text2SQL LoRA cookbook](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql)
- [NVIDIA CUDA DL 26.08 release notes](https://docs.nvidia.com/deeplearning/frameworks/cuda-dl-release-notes/rel-26-08.html)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [BANKING77 dataset card](https://huggingface.co/datasets/PolyAI/banking77)
- [BANKING77 paper](https://aclanthology.org/2020.nlp4convai-1.5/)

The NVIDIA model is governed by OpenMDW 1.1. BANKING77 is CC BY 4.0; retain
dataset attribution when adapting or redistributing prepared examples.
