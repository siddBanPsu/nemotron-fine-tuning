# Nemotron 3.5 Lightning Text2SQL fine-tuning lab

This is a four-notebook, API-first and Brev-ready lab for adapting
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` to Text2SQL:

1. benchmark hosted Nemotron 3.5 Lightning and Nemotron 3 Ultra on executable SQL;
2. run the exact local BF16 baseline with vLLM;
3. fine-tune Lightning with LoRA using NVIDIA's official Text2SQL recipe, merge it,
   and measure the paired held-out gain;
4. inspect true full-parameter SFT as a design-only multi-GPU handoff.

The practical workshop is Notebooks 01–03. Notebook 01 needs no GPU. With the
model prefetched, Notebook 03's bounded 4,096-example/64-step profile is intended
to leave time for merge and evaluation within a one-hour GPU exercise on an
80 GB-class H100 or a sharded 2×A100/H100 allocation. Runtime still depends on
the exact GPU, storage, network, and container startup.

## Why Text2SQL is a better fine-tuning exercise

The previous opaque intent-code task could reduce training loss while collapsing
onto a few labels. This rebuild measures useful generated text instead:

- input: real database DDL, a natural-language question, and optional business evidence;
- output: executable SQLite SQL;
- primary metric: whether predicted and reference SQL return the same result;
- secondary diagnostics: parse validity, executability, normalized SQL exact match,
  and accuracy by BIRD difficulty;
- proof: local BF16 and LoRA are scored on the identical frozen IDs with a paired
  bootstrap confidence interval.

The training path follows NVIDIA's official
[Nemotron 3.5 Lightning Text2SQL LoRA cookbook](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql/nemo-megatron-bridge):
BIRD direct and reasoning examples, the model's native chat template, packed
sequences, the shipped Lightning LoRA target modules, Megatron distributed
checkpoints, adapter merge, and vLLM serving. The lab pins the reviewed cookbook,
model, datasets, and Megatron-Bridge revisions in code.

The training driver records the data hash, model revision, topology, schedule,
and LoRA settings beside the checkpoint. It refuses an incompatible resume;
choose a new output directory when changing an experiment.

## Data and evaluation contract

Training uses only the two BIRD train mirrors used by NVIDIA's cookbook:

- `xu3kev/BIRD-SQL-data-train` (9,428 direct-SQL rows before filtering);
- `meowterspace45/bird-sql-train-with-reasoning` (reasoning-augmented train rows).

The workshop shuffles the combined sources before rendering, filters examples
longer than 2,048 tokens, keeps 4,096 rows, sorts them by length, and writes the
Megatron prompt-completion JSONL plus a source/revision/hash manifest.
With the pinned inputs, this produces 2,798 direct and 1,298 reasoning rows,
4,000,344 tokens, and an estimated 1,954 packed sequences: 62 steps at GBS 32.

Evaluation uses only official
[BIRD Mini-Dev](https://github.com/bird-bench/mini_dev). The first run downloads
the official ~800 MB package containing 11 SQLite databases and freezes 100 of
the 500 SELECT-only Mini-Dev questions. The subset preserves the benchmark's
simple/moderate/challenging ratio and spreads examples across every database.
Question 701 is excluded by protocol because its official gold query exceeded
the 30-second audit timeout. The exclusion and protocol version are pinned in
the manifest; preparation fails instead of silently resampling on a slower host.
Notebook 01 takes a second deterministic 25-row subset for each hosted model,
so the normal cloud workload is 50 requests total under the public 40-RPM quota.

Execution scoring opens every SQLite database read-only, accepts only one parsed
`SELECT`/`WITH` statement, enforces a timeout and row cap, and uses the official
BIRD Mini-Dev set-of-result-rows equality rule. This is more meaningful than SQL
string match, while normalized exact match remains available for diagnosis.
Malformed generations are recorded as invalid SQL rather than aborting a run.
Local vLLM predictions are saved before scoring and reused only when the model,
evaluation hash, prompt protocol, and generation settings still match.

No benchmark guarantees that one short run will improve a strong base model.
The notebook prints “no held-out gain demonstrated” when the paired result does
not improve; loss reduction alone is never presented as success.

## Hardware and honest scope

Nemotron 3.5 Lightning has 30B total and roughly 3B active parameters per token.
Sparse activation reduces compute, but all weights still occupy memory.

| Path | Hardware | Default scope |
| --- | --- | --- |
| Hosted Lightning + Ultra | CPU and NVIDIA API key | 25 frozen rows/model; 50 calls total at 30 RPM |
| Local BF16 baseline | 1× H100/A100 80 GB | vLLM on all 100 frozen rows |
| LoRA workshop | 1× H100 80 GB or 2×A100/H100 80 GB | 4,096 rows, 2K packing, GBS 32, rank 32, ≤64 steps |
| Full-SFT notebook | Any notebook host | memory/topology design only; launches nothing |
| External full SFT | ≥16×H100 80 GB / ≥1,200 GiB aggregate VRAM | hardware-gated driver; separately rehearse |

NVIDIA reports the complete 12,544-example official LoRA epoch at about 60
minutes on one H100, 34 minutes on two, 18 on four, and 8 on eight, with roughly
79/51/35/27 GB peak memory per GPU. On one GPU, its runbook reduces the model to
the checkpoint's single MTP head; this lab does the same. These are NVIDIA's H100
measurements, not claimed A100 timings.

Full AdamW needs BF16 weights and gradients plus FP32 master weights/moments,
activations, communication buffers, and workspaces. Notebook 04 does not pretend
that a one-GPU Launchable can perform full SFT.

## Repository layout

```text
notebooks/
  01_cloud_api_baseline.ipynb
  02_local_bf16_baseline.ipynb
  03_peft_lora.ipynb
  04_full_finetuning_design.ipynb
scripts/
  prepare_text2sql.py       # BIRD train + executable Mini-Dev bundle
  evaluate_vllm.py          # isolated local generation and SQL execution
  convert_checkpoint.py
  train_peft.py
  train_full.py
  export_full_checkpoint.py
src/nemotron_ft_lab/
launchable/
tests/
```

Generated datasets, reports, model caches, converted checkpoints, adapters, and
merged exports are ignored by Git. No model weights, databases, API responses,
or credentials are redistributed by this repository.

## Notebook 01 on a laptop or CPU VM

Use Python 3.12. The lightweight environment deliberately does not install
PyTorch, CUDA, NeMo, Megatron, or vLLM:

```bash
python3.12 -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -r requirements-lab.txt --editable .
jupyter lab notebooks/01_cloud_api_baseline.ipynb
```

Export `NVIDIA_API_KEY` before starting Jupyter, or paste it into the notebook's
native `input()` prompt and press Enter. The prompt is cleared after connection.
The evaluator paces public requests at 30 RPM, honors retry headers after a 429,
and checkpoints every successful response.

### Public versus internal API endpoints

Public settings are tracked. For a private/faster OpenAI-compatible endpoint:

```bash
cp config/api.local.toml.example config/api.local.toml
```

Edit only the private base URL, Lightning/Ultra model IDs, served-variant labels,
rate, and timeout. Keep `NVIDIA_API_KEY` in the process environment. In Notebook
01, change one line and rerun configuration plus authentication:

```python
API_PROFILE_OVERRIDE = "local"   # or "public"; None selects local if present
```

Private/public caches include a short hash of the endpoint and model ID, so a
configuration change cannot resume stale responses. Environment overrides in
[.env.example](.env.example) support configs stored outside the checkout.

## Brev quick start

Follow [launchable/README.md](launchable/README.md). The Launchable uses:

- VM mode and a recommended H100 80 GB (or a rehearsed A100 80 GB alternative);
- `nvcr.io/nvidia/nemo:26.08`, which supplies the matched Python 3.12,
  PyTorch/CUDA/Transformer Engine/Megatron stack;
- NVIDIA driver 580.65.06+ for CUDA 13.x minor-version compatibility, with a real
  in-container CUDA smoke test before Jupyter starts;
- the pinned Megatron-Bridge revision and the live-validated MoE padding-mask
  compatibility hook;
- a Brev Secure Link on host port 8889;
- persistent cache, data, artifacts, checkpoints, and temporary directories.

Notebook 01 can run in a host `.venv`; Notebooks 02–03 must run in the container
Jupyter opened on port 8889. Do not install `torch` into the host environment to
repair a GPU notebook—it is the wrong interpreter boundary.

For a scheduled workshop, prefetch the ~62 GB BF16 model and convert it before
participant time:

```bash
export NEMOTRON_PREFETCH_MODEL=1
bash launchable/setup.sh
```

Use `NEMOTRON_PREFETCH_MODEL=0` for the fastest API-first startup. The Mini-Dev
archive is downloaded only when Notebook 01 or 02 prepares evaluation data.

## Large-volume storage

The model, converted checkpoint, merged export, caches, Mini-Dev databases, and
temporary files need substantial disk. NVIDIA's official LoRA cookbook asks for
about 130 GB before the merged export; provision at least 300 GB for this lab.
On a VM with a large mounted volume:

```bash
export NEMOTRON_DATA_ROOT=/path/to/large-volume/nemotron-fine-tuning
export NEMOTRON_PREFETCH_MODEL=0
bash launchable/setup.sh
```

This routes repository-controlled large writes under that root. Docker image
layers remain under Docker's own data root, commonly `/var/lib/docker`, which
also needs capacity. `/tmp` is acceptable for a disposable VM only when it is a
large disk rather than RAM-backed `tmpfs`; copy reports/checkpoints elsewhere
before shutdown.

## Verify the repository

From the lightweight Python 3.12 environment:

```bash
python scripts/build_notebooks.py
python scripts/validate_repo.py
python -m unittest discover -s tests
```

GPU correctness still requires the target container: import checks, conversion,
the training dry run, vLLM startup, and a short live train/evaluate rehearsal
cannot be proven by laptop unit tests.

## Primary references

- [Nemotron 3.5 Lightning usage cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/README.md)
- [Official Lightning LoRA Text2SQL recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql/nemo-megatron-bridge)
- [Nemotron 3.5 Lightning BF16 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- [BIRD Mini-Dev benchmark](https://github.com/bird-bench/mini_dev)
