# Nemotron executable Text2SQL fine-tuning lab

This is an API-first, Brev-ready lab that demonstrates whether fine-tuning
improves a model on generated text that can be checked objectively. The model
must turn a database schema, natural-language question, and optional evidence
into SQLite SQL. Predicted and reference queries are executed against the same
official BIRD database; lower training loss alone never counts as success.

The repository has two deliberately separate local paths:

| Path | Model | Hardware | Training scope | Runtime status |
| --- | --- | --- | --- | --- |
| Recommended workshop | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 1×RTX PRO 6000 Blackwell Server Edition 96 GB; or A100/H100 80 GB only with driver 580.65.06+ | default: 1,024 direct rows, 2K packing, GBS 32, rank 32, ≤16 steps; extended: 2,048 rows/≤32 steps | warm-cache Notebooks 02–03 target ≤60 min; target-SKU rehearsal still required |
| Advanced exercise | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | 2×A100/H100 80 GB | 4,096 direct/reasoning rows, 2K packing, GBS 32, rank 32, ≤64 steps | measured 6 h 49 min on 2×A100 for train, merge, and evaluation |
| Full SFT handoff | Lightning 30B total parameters | 16×H100 80 GB reference gate | all parameters | design-only Notebook 05; never launches on Brev |

The default is `nano9b_workshop`. Nano and Lightning use different native
prompts, training data renderings, Megatron recipes, checkpoint directories,
raw-prediction caches, and evaluation reports. The code refuses mismatched
training or baseline artifacts rather than producing a plausible but invalid
comparison.

## The five notebooks

1. `01_cloud_api_baseline.ipynb` compares hosted Nemotron 3.5 Lightning and
   Nemotron 3 Ultra on 25 identical Mini-Dev rows each. It is CPU-only.
2. `02_local_bf16_baseline.ipynb` runs the selected pinned local BF16 model on
   all 100 frozen rows with vLLM. Nano is the default; one line selects
   Lightning.
3. `03_nano9b_workshop_lora.ipynb` fine-tunes Nano with the pinned one-GPU
   Megatron-Bridge PEFT recipe, merges it, reevaluates the same 100 IDs, and
   bootstraps the paired execution-accuracy gain.
4. `04_lightning_advanced_lora.ipynb` preserves the official-recipe Lightning
   exercise for two 80 GB GPUs and a multi-hour allocation.
5. `05_full_finetuning_design.ipynb` explains and gates true Lightning full SFT;
   it prints a handoff command but launches nothing.

Hosted scores are useful targets, not the causal fine-tuning comparison. The
claim that matters is always local BF16 base versus merged LoRA from the same
profile on the same 100 example IDs.

## Workshop quick start

Use a Brev VM with at least 80 GB VRAM, 128 GB host RAM, 200 GB disk, and host
driver 580.65.06 or newer. The least-surprising current choice is one RTX PRO
6000 Blackwell Server Edition 96 GB: an observed Brev instance with driver
595.91.07 completed this repository's NeMo 26.08 container and CUDA startup
gate. That result does not yet prove the Nano training runtime or accuracy.

Do not select by GPU name alone. An observed Brev A100 80 GB image shipped
driver 565.57.01 and was correctly rejected before the container download.
Another A100/H100 provider image is valid if `nvidia-smi` reports driver
580.65.06+. Prefetch before participants arrive:

```bash
export NEMOTRON_MODEL_PROFILE=nano9b_workshop
export NEMOTRON_PREFETCH_MODEL=1
export NEMOTRON_DATA_ROOT=/path/to/large/persistent/volume/nemotron-fine-tuning
bash launchable/setup.sh
```

Open the authenticated Brev Secure Link on host port 8889, then run Notebooks
01, 02, and 03. Notebook 01 is optional if the workshop does not have API keys.
Notebook 02 is not optional for the accuracy claim.

For the Lightning advanced path:

```bash
export NEMOTRON_MODEL_PROFILE=lightning35_advanced
export NEMOTRON_PREFETCH_MODEL=1
bash launchable/setup.sh
```

In Notebook 02 set:

```python
MODEL_PROFILE_NAME = "lightning35_advanced"
```

Then run Notebook 04. Notebook 04 itself is always pinned to Lightning even if
the container was launched with Nano; without Lightning prefetch it will perform
the large download and conversion inside the notebook.

### Upgrading an older checkout

The former generic `03_peft_lora.ipynb` is now split into Nano Notebook 03 and
Lightning Notebook 04; the full-SFT design moved to Notebook 05. Existing
unnamespaced reports and `/workspace/storage/checkpoints/bird-text2sql-lora`
are left untouched but are not reused. Run Notebook 02 again for the selected
profile, then let the new LoRA notebook create its profile-specific checkpoint.
This deliberate break prevents an old Lightning artifact from being mistaken
for a Nano result.

Notebook 03 also separates its two schedules. Leave `WORKSHOP_MODE = True` for
1,024 rows and at most 16 steps. Set it to `False` for 2,048 rows and at most
32 steps. The modes use distinct training-data, adapter, merged-model, and
evaluation paths, so changing the switch cannot resume an incompatible run.

## What “within an hour” means

The Nano path is designed around an under-one-hour **acceptance target**, not a
published result. It is complete only after a live rehearsal on the exact Brev
GPU SKU, storage tier, image, and repository commit demonstrates all of the
following from a warm cache:

- Notebook 02 baseline plus Notebook 03 training, merge, and evaluation finish
  within 60 minutes;
- the run uses exactly one GPU and needs no manual recovery;
- merged-model execution accuracy is higher than its matching BF16 baseline on
  the frozen 100 rows;
- the report includes the paired bootstrap interval, SQL validity, hardware,
  stage times, and exact model profile.

Model download, container pull, BIRD download, and Hugging Face→Megatron
conversion happen before participant time. Until that rehearsal exists, plan
session margin and do not advertise a guaranteed 60-minute completion.

The pinned Megatron-Bridge commit ships
`nemotron_nano_9b_v2_peft_1gpu_h100_bf16_config` with TP1/PP1 and model-specific
LoRA targets. Nano inference and training use `/no_think`; vLLM also uses a
float32 Mamba SSM cache as required by the model card. Training uses direct SQL
only so the prompt and target remain aligned with non-thinking evaluation.

### Important Nano maintenance caveat

Current upstream Megatron-Bridge documentation marks Nemotron Nano v2 support
deprecated and scheduled for removal, although this repository's pinned commit
still includes the working one-GPU recipe. Treat Nano as a compatibility-frozen,
bounded workshop profile. Updating Megatron-Bridge is a deliberate migration
exercise, not a routine dependency bump.

## Measured Lightning runtime: read before renting GPUs

The following is an observed rehearsal completed on 2026-09-03, not a vendor
benchmark or runtime guarantee. It used 2×NVIDIA A100 80 GB PCIe, Python 3.12,
4,096 rendered training rows, 2,048-token packing, GBS 32, and 62 resulting
steps. The model, Mini-Dev data, and converted Megatron base were already
present, so this is a warm-cache result.

| Stage | Resources | Observed time | Detail |
| --- | --- | ---: | --- |
| Matching local BF16 baseline | 1 A100 | 2.6 min | 100-row vLLM load, generation, and scoring; generation itself 66 s |
| Checkpoint check | CPU | 0.6 min | reused the converted Megatron base |
| Lightning LoRA | 2×A100, TP1/EP2 | 384.5 min | 62 steps; first step 477.5 s, later median 366.9 s |
| CPU adapter merge | CPU/RAM | 19.9 min | produced a roughly 61 GB Hugging Face checkpoint |
| Merged-model evaluation | 1 A100 | 3.7 min | load 130.5 s; generation 15 s |
| **Train + merge + evaluation** | mixed | **408.7 min / 6 h 49 min** | excludes model download and base conversion |

The baseline plus adaptation therefore occupied about 6 h 51 min. A cold run is
longer and network/storage dependent: expect a roughly 62 GB model snapshot,
the ~800 MB compressed Mini-Dev package, checkpoint conversion, dataset packing,
container pull, and compilation caches.

Training prints one `Step Time` per iteration. Estimate remaining time using the
median of several steady-state steps:

```text
remaining training minutes ≈ remaining steps × median Step Time seconds / 60
```

In the measured run, ~367 s per later step correctly signaled a multi-hour wait.
Lightning saves at its final configured step. Nano saves every eight steps and
can resume only when the recorded run contract is identical.

### Measured Lightning quality result

On the identical frozen 100-row Mini-Dev subset, that run improved execution
accuracy from 15% to 40%: +25 percentage points with a paired-bootstrap 95%
interval of +15 to +35 points. Thirty examples improved, five regressed, and
SQL validity rose from 46% to 98%. This demonstrates that this particular
Lightning run learned the task; it is not a full-BIRD score or a guarantee for
Nano or another rerun.

## Why Text2SQL is a useful fine-tuning exercise

Opaque label prediction can hide collapse behind falling loss. Text2SQL exposes
the actual generated artifact and checks its behavior:

- input: real database DDL, a natural-language question, and optional BIRD
  evidence;
- output: a single read-only SQLite `SELECT` or `WITH` statement;
- primary metric: predicted and reference SQL return the same result;
- diagnostics: syntax validity, executability, normalized SQL exact match,
  difficulty, improved examples, and regressions;
- proof: paired execution outcomes on an untouched, preselected holdout.

No short fine-tuning run is guaranteed to beat a strong base. The notebooks
print “no held-out execution gain demonstrated” when the delta is not positive.

## Data and evaluation contract

Training uses BIRD train mirrors from NVIDIA's Lightning Text2SQL cookbook:

- `xu3kev/BIRD-SQL-data-train`, pinned at
  `9122256f9d14752ed80fb9b7d158e21d9f9261aa`;
- `meowterspace45/bird-sql-train-with-reasoning`, pinned at
  `9e351e0057819f1b0917debb83c8e12f321157a4` for Lightning only.

Nano workshop mode shuffles and renders up to 1,024 direct examples with its
instruct chat template and `/no_think`; extended mode uses up to 2,048.
Lightning shuffles both sources, filters beyond 2,048 tokens, and keeps 4,096
rows. Each profile writes its own JSONL and manifest under
`artifacts/data/bird-text2sql/profiles/<profile>/`; Nano adds `workshop` or
`extended`. The manifest records
model revision, system prompt, reasoning mode, dataset revisions, row/token
counts, source distribution, and SHA-256.

Evaluation uses only official BIRD Mini-Dev. The first run verifies and extracts
the ~800 MB archive with 11 SQLite databases, then freezes 100 of 500 SELECT-only
questions before any model is scored. Selection preserves the difficulty mix
and spreads examples across databases. Question 701 is a pinned protocol
exclusion because its official gold query exceeded the 30-second repository
audit timeout; slower machines do not silently resample it.

The executor opens databases read-only, accepts one parsed `SELECT`/`WITH`,
enforces a timeout and row cap, and uses Mini-Dev set-of-result-rows semantics.
Malformed output is scored invalid instead of aborting the run. Raw predictions
are persisted before scoring and reused only when model identity, evaluation
hash, model profile, prompt, cache dtype, and generation settings match.

Notebook 01 uses a deterministic 25-row subset for each hosted model. The
normal public workload is therefore 50 calls—not 231—and requests are paced at
30 RPM below the public 40-RPM limit. Every successful response is checkpointed.

## Reproducibility pins

| Component | Pin |
| --- | --- |
| Nano 9B v2 | `6533e8de2c68e4536bf7c411d7a3ce5734111476` |
| Lightning 3.5 BF16 | `b3caaabed0263651a17dc1f2d4ce97e794f76c44` |
| Megatron-Bridge | `8bc33cd2ca1cd044e1520130f4c5e1f5e0181434` |
| NVIDIA Nemotron cookbook snapshot | `ccbea41e1ccb8a9bda9169ca83f19d18e39b9cdc` |
| Container | `nvcr.io/nvidia/nemo:26.08` |
| Python | 3.12 |

Run manifests beside LoRA checkpoints pin the profile, recipe, model, revision,
data hash, topology, schedule, and rank. A preexisting checkpoint without a
matching manifest is rejected. Use a new output directory for a changed
experiment.

## Disk, cache, and long-running cell expectations

- Nano: provision at least 200 GB for container-adjacent working data, model
  snapshot, converted base, adapter, merged model, BIRD data, and caches.
- Lightning: provision at least 300 GB. Its snapshot is roughly 62 GB and the
  merged export is another roughly 61 GB.
- Docker image layers remain under Docker's own data root, often
  `/var/lib/docker`, even when `NEMOTRON_DATA_ROOT` redirects lab files.
- `NEMOTRON_PREFETCH_MODEL=1` delays Jupyter but moves the selected model
  download and conversion before participant work.
- CPU merge intentionally shows low GPU utilization. It still needs RAM and
  fast storage.
- vLLM model loading can take longer than generation. Wait for the subprocess
  result, not a utilization snapshot.
- A stage is complete only after exit code zero and its expected checkpoint or
  report exists.

The observed Lightning run emitted non-fatal warnings about `pynvml`, missing
optional `torchao`, unavailable optional Triton kernels, and some unrecognized
MTP mapping lines during merge. Those exact warnings were harmless in that run;
do not generalize that to new tracebacks or a non-zero exit.

## Laptop/API-only setup

Notebook 01 can run without CUDA. The lightweight environment intentionally
does not install PyTorch, NeMo, Megatron, or vLLM:

```bash
python3.12 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -r requirements-lab.txt --editable .
jupyter lab notebooks/01_cloud_api_baseline.ipynb
```

Export `NVIDIA_API_KEY` before Jupyter, or use the notebook's native `input()`
fallback. The visible input is immediately cleared and never written to an
artifact.

### Public versus internal API endpoints

Public settings are tracked. For a private OpenAI-compatible endpoint:

```bash
cp config/api.local.toml.example config/api.local.toml
```

Edit the private base URL, Lightning/Ultra model IDs, served-variant labels,
rate, and timeout. Keep the key in the environment. In Notebook 01 set:

```python
API_PROFILE_OVERRIDE = "local"  # "public" or None for automatic selection
```

Endpoint and model IDs are hashed into API cache names, so internal and public
responses cannot collide. [.env.example](.env.example) documents equivalent
environment overrides and external config paths.

## Repository layout

```text
notebooks/
  01_cloud_api_baseline.ipynb
  02_local_bf16_baseline.ipynb
  03_nano9b_workshop_lora.ipynb
  04_lightning_advanced_lora.ipynb
  05_full_finetuning_design.ipynb
scripts/
  show_model_profile.py     # inspect or list the two pinned profiles
  prepare_text2sql.py       # profile-rendered BIRD train + frozen Mini-Dev
  evaluate_vllm.py          # isolated generation and executable SQL scoring
  convert_checkpoint.py
  train_peft.py             # selects the pinned Nano or Lightning recipe
  train_full.py             # Lightning-only, hardware-gated
  export_full_checkpoint.py
src/nemotron_ft_lab/
launchable/
tests/
```

Generated data, reports, API responses, databases, caches, model weights,
converted checkpoints, adapters, and merged exports are ignored by Git and are
not redistributed.

## Brev and container requirements

See [launchable/README.md](launchable/README.md) for complete setup and failure
recovery. The launchable uses VM mode, host port 8889, no public TCP ports, and
`nvcr.io/nvidia/nemo:26.08`. It requires host driver 580.65.06+ for CUDA 13.x
minor compatibility; 610.43.02+ is native. A real CUDA arithmetic smoke test
runs inside the container before Jupyter starts.

The driver is part of the selected VM image, not the GPU model. The lifecycle
script intentionally will not install or replace it: loading a new NVIDIA
kernel driver requires a reboot, which would terminate Brev's on-create job.
An observed RTX PRO 6000/driver 595.91.07 deployment passed setup using CUDA
forward compatibility; an observed A100/driver 565.57.01 deployment cannot run
this CUDA 13 stack. Choose a different provider/base image instead of lowering
the gate.

The host `.venv` is API-only. Notebooks 02–05 must use the container Python.
`ModuleNotFoundError: torch` in a GPU notebook means the wrong kernel, not that
you should install a second PyTorch stack into the host environment.

## Verify before publishing or teaching

From a Python 3.12 environment with `requirements-lab.txt` installed:

```bash
python scripts/build_notebooks.py
python scripts/validate_repo.py
python -m unittest discover -s tests
```

Laptop validation proves structure, contracts, parser/evaluator behavior,
notebook syntax, and launch-script syntax. It cannot prove GPU imports,
checkpoint conversion, memory headroom, runtime, or accuracy. Before a workshop,
run a cold setup rehearsal and a complete warm-cache Notebook 02–03 rehearsal on
the exact target instance type.

## Primary references

- [Nemotron Nano 9B v2 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [Pinned Megatron-Bridge Nano 9B v2 PEFT recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8bc33cd2ca1cd044e1520130f4c5e1f5e0181434/src/megatron/bridge/recipes/nemotronh/h100/nemotron_nano_v2.py)
- [Current Megatron-Bridge NemotronH support status](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/nemotron/nemotronh.md)
- [Nemotron 3.5 Lightning usage cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/README.md)
- [Official Lightning LoRA Text2SQL recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql/nemo-megatron-bridge)
- [Nemotron 3.5 Lightning BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [BIRD Mini-Dev benchmark](https://github.com/bird-bench/mini_dev)
