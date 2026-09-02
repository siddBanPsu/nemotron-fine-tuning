#!/usr/bin/env python3
"""Regenerate the four checked-in notebooks from reviewable Python strings."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip() + "\n",
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "01_cloud_api_baseline.ipynb": notebook(
        [
            markdown(
                """
                # 01 — Hosted Lightning and Ultra Text2SQL baselines

                This CPU-only notebook evaluates NVIDIA-hosted Nemotron 3.5 Lightning and Nemotron 3 Ultra on the same frozen BIRD Mini-Dev SQLite questions. Each model sees database DDL, the natural-language question, and BIRD's evidence, then must produce SQL.

                The primary score is **execution accuracy**: predicted and reference SQL must return the same result on the official database. SQL validity, executability, and normalized string match are diagnostics. The default is 25 requests per model—50 total—not hundreds of calls. Responses checkpoint after every request and resume safely after a 429 or interruption.

                Hosted models are useful task targets, but they are NVFP4 services. Notebook 02 repeats the full 100-row holdout on the exact local BF16 checkpoint that Notebook 03 tunes.
                """
            ),
            code(
                """
                from pathlib import Path
                import hashlib, json, os, subprocess, sys, time

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                print('Repository:', ROOT)
                print('Artifacts:', ARTIFACTS_DIR)
                """
            ),
            markdown(
                """
                ## 1. API preflight and public/private endpoint switch

                Copy `config/api.local.toml.example` to the ignored `config/api.local.toml` for an internal OpenAI-compatible endpoint. Set `API_PROFILE_OVERRIDE` to `'public'` or `'local'` and rerun this cell plus authentication; leave it `None` for automatic selection. Keys remain in the environment or notebook input, never in TOML.
                """
            ),
            code(
                """
                subprocess.run([sys.executable, 'scripts/preflight.py', '--profile', 'api'], check=True)

                from nemotron_ft_lab.api_config import load_nvidia_api_config

                API_PROFILE_OVERRIDE = None  # None=auto, or use 'public' / 'local'
                API_CONFIG = load_nvidia_api_config(ROOT, profile=API_PROFILE_OVERRIDE)
                print('API profile:', API_CONFIG.profile_name, f'({API_CONFIG.source_label})')
                print('Endpoint:', API_CONFIG.base_url)
                print('Lightning:', API_CONFIG.lightning.model_id, '->', API_CONFIG.lightning.served_variant)
                print('Ultra:', API_CONFIG.ultra.model_id, '->', API_CONFIG.ultra.served_variant)
                print('Rate limit used by this notebook:', API_CONFIG.requests_per_minute, 'requests/minute')
                """
            ),
            markdown(
                """
                ## 2. Freeze the executable evaluation bundle

                The first run downloads the official BIRD Mini-Dev package (~800 MB compressed), extracts its SQLite databases, and selects 100 rows before any model is scored. Selection preserves the benchmark's difficulty mix and spreads rows across all 11 databases. The manifest records a pinned exclusion for question 701, whose official gold query exceeded the 30-second audit timeout; preparation fails instead of silently choosing different IDs on slower hardware. Training later uses only BIRD train mirrors; Mini-Dev is never training data.
                """
            ),
            code(
                """
                subprocess.run([
                    sys.executable, 'scripts/prepare_text2sql.py',
                    '--output-dir', str(DATA_DIR), '--evaluation-only',
                ], check=True)

                from nemotron_ft_lab.constants import DEFAULT_API_EVALUATION_SIZE, DEFAULT_SEED
                from nemotron_ft_lab.data import balanced_evaluation_subset, build_messages, read_jsonl

                all_eval_rows = read_jsonl(DATA_DIR / 'evaluation.jsonl')
                CLOUD_EVAL_SIZE = DEFAULT_API_EVALUATION_SIZE  # 25 requests/model; set <=100 before first run
                cloud_rows = balanced_evaluation_subset(
                    all_eval_rows, size=CLOUD_EVAL_SIZE, seed=DEFAULT_SEED + 1,
                )
                manifest = json.loads((DATA_DIR / 'evaluation_manifest.json').read_text())
                print('Frozen local holdout:', len(all_eval_rows))
                print('Cloud rows per model:', len(cloud_rows))
                print('Cloud requests across Lightning + Ultra:', 2 * len(cloud_rows))
                print('Difficulty mix:', manifest['difficulty_distribution'])
                print('Databases:', len(manifest['database_distribution']))
                print('Pinned gold-query exclusions:', manifest['excluded_gold_execution_question_ids'])
                print()
                print('Example user message:')
                print(build_messages(cloud_rows[0])[1]['content'][:1800])
                """
            ),
            markdown(
                """
                ## 3. Authenticate without storing the key

                Prefer exporting `NVIDIA_API_KEY` before starting Jupyter. Otherwise the built-in `input()` prompt is used because some notebook password widgets cannot accept paste. Input is briefly visible, then cleared. The key is never written to an artifact.
                """
            ),
            code(
                """
                from openai import OpenAI
                from IPython.display import clear_output

                api_key = os.environ.get('NVIDIA_API_KEY', '').strip()
                used_native_prompt = not api_key
                if used_native_prompt:
                    api_key = input('Paste NVIDIA API key (visible until Enter), then press Enter: ').strip()
                    clear_output(wait=False)
                if not api_key:
                    raise RuntimeError('An NVIDIA API key is required for hosted baselines.')
                client = OpenAI(
                    base_url=API_CONFIG.base_url, api_key=api_key,
                    max_retries=0, timeout=API_CONFIG.timeout_seconds,
                )
                del api_key
                print('API client configured; input cleared.' if used_native_prompt else 'API client configured from NVIDIA_API_KEY.')
                """
            ),
            markdown(
                """
                ## 4. Run both hosted models and execute their SQL

                Prompt protocol `v1` is identical for both models: empty system turn, then `schema`, blank line, `question`, blank line, optional `evidence`, with thinking disabled. Cache filenames include the API profile and sample size so internal/public runs cannot collide.
                """
            ),
            code(
                """
                from nemotron_ft_lab.evaluation import (
                    generate_nvidia_api_predictions, paired_execution_comparison,
                    save_report, score_predictions,
                )

                CLOUD_MODELS = {
                    'lightning': API_CONFIG.lightning,
                    'ultra': API_CONFIG.ultra,
                }
                api_reports = {}
                for name, spec in CLOUD_MODELS.items():
                    endpoint_model_fingerprint = hashlib.sha256(
                        f'{API_CONFIG.base_url}\\0{spec.model_id}\\0{manifest["evaluation_sha256"]}\\0v1'.encode()
                    ).hexdigest()[:12]
                    artifact_name = f'{API_CONFIG.artifact_prefix}{name}_{endpoint_model_fingerprint}'
                    resume_path = ARTIFACTS_DIR / f'evaluation/api_{artifact_name}_text2sql_v1_{CLOUD_EVAL_SIZE}.jsonl'
                    print()
                    print(f'Evaluating {name}: {spec.model_id}')
                    started = time.perf_counter()
                    generated = generate_nvidia_api_predictions(
                        client, cloud_rows, model=spec.model_id, resume_path=resume_path,
                        requests_per_minute=API_CONFIG.requests_per_minute,
                    )
                    report = score_predictions(generated, data_dir=DATA_DIR)
                    report.update({
                        'wall_time_seconds': time.perf_counter() - started,
                        'endpoint': API_CONFIG.base_url,
                        'api_profile': API_CONFIG.profile_name,
                        'served_variant': spec.served_variant,
                        'precision': 'NVFP4',
                        'prompt_protocol': 'bird-schema-question-evidence-v1',
                    })
                    report_path = ARTIFACTS_DIR / f'evaluation/baseline_api_{artifact_name}_text2sql_{CLOUD_EVAL_SIZE}.json'
                    save_report(report_path, report, model=spec.model_id, run_type=f'hosted-{name}-text2sql')
                    api_reports[name] = report

                metrics = ('n', 'execution_accuracy', 'sql_valid_rate', 'sql_executable_rate', 'normalized_exact_match')
                {name: {key: report[key] for key in metrics} for name, report in api_reports.items()}
                """
            ),
            code(
                """
                ultra_vs_lightning = paired_execution_comparison(
                    api_reports['lightning'], api_reports['ultra'],
                )
                ultra_vs_lightning.update({
                    'lightning_execution_accuracy': api_reports['lightning']['execution_accuracy'],
                    'ultra_execution_accuracy': api_reports['ultra']['execution_accuracy'],
                    'interpretation': 'Ultra minus Lightning on identical hosted requests',
                })
                print(json.dumps(ultra_vs_lightning, indent=2))

                for name, report in api_reports.items():
                    print()
                    print(f'{name.title()} examples:')
                    for row in report['rows'][:3]:
                        print()
                        print('Q:', row['question'])
                        print('Gold:', row['expected_sql'])
                        print('Generated:', row['generated'])
                        print('Execution correct:', row['execution_correct'])
                """
            ),
            markdown(
                """
                ## Result contract

                These hosted scores are task targets, not the causal fine-tuning comparison. Notebook 02 evaluates the pinned local BF16 base on all 100 frozen IDs. Notebook 03 evaluates the merged LoRA checkpoint on those same 100 IDs and bootstraps the paired execution-accuracy delta; it also compares tuned Lightning with hosted Lightning/Ultra only on their shared 25 IDs.
                """
            ),
        ]
    ),
    "02_local_bf16_baseline.ipynb": notebook(
        [
            markdown(
                """
                # 02 — Local BF16 Text2SQL baseline with vLLM

                This notebook loads the exact pinned BF16 customization checkpoint that Notebook 03 will tune and freezes its score on all 100 Mini-Dev rows. It uses vLLM because NVIDIA's official Nemotron 3.5 Lightning Text2SQL runbook specifies vLLM for this architecture; the model's bundled `transformers.generate()` path is not the supported baseline.

                **Required runtime:** use the NeMo container Jupyter opened by `launchable/setup.sh` through the Secure Link on host port **8889**. The host `.venv` is API-only and intentionally has no PyTorch.
                """
            ),
            code(
                """
                from pathlib import Path
                import json, os, subprocess, sys, time

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                BASELINE_PATH = ARTIFACTS_DIR / 'evaluation/baseline_local_bf16_text2sql.json'
                print('Repository:', ROOT)
                print('Artifacts:', ARTIFACTS_DIR)
                print('Python:', sys.executable)
                """
            ),
            markdown("""## 1. Verify the container, GPU, model identity, and frozen holdout"""),
            code(
                """
                subprocess.run([sys.executable, 'scripts/preflight.py', '--profile', 'inference'], check=True)
                subprocess.run([
                    sys.executable, 'scripts/prepare_text2sql.py',
                    '--output-dir', str(DATA_DIR), '--evaluation-only',
                ], check=True)

                import torch
                from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION
                from nemotron_ft_lab.data import read_jsonl

                eval_rows = read_jsonl(DATA_DIR / 'evaluation.jsonl')
                manifest = json.loads((DATA_DIR / 'evaluation_manifest.json').read_text())
                print('Model:', MODEL_ID)
                print('Pinned revision:', MODEL_REVISION)
                print('GPU(s):', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
                print('Evaluation rows:', len(eval_rows), manifest['difficulty_distribution'])
                """
            ),
            markdown(
                """
                ## 2. Run the local baseline in an isolated vLLM process

                Process isolation matters: when evaluation finishes, vLLM exits and releases GPU memory before LoRA training. One 80 GB GPU is the conservative default; set `NEMOTRON_INFERENCE_GPUS=2` only if this model/backend combination has been rehearsed on the node.
                """
            ),
            code(
                """
                INFERENCE_GPUS = int(os.environ.get('NEMOTRON_INFERENCE_GPUS', '1'))
                if not 1 <= INFERENCE_GPUS <= torch.cuda.device_count():
                    raise RuntimeError(f'NEMOTRON_INFERENCE_GPUS must be between 1 and {torch.cuda.device_count()}.')
                command = [
                    sys.executable, 'scripts/evaluate_vllm.py',
                    '--model', MODEL_ID, '--revision', MODEL_REVISION,
                    '--data-dir', str(DATA_DIR), '--output', str(BASELINE_PATH),
                    '--run-type', 'untuned-local-bf16-text2sql',
                    '--tensor-parallel-size', str(INFERENCE_GPUS),
                ]
                print('Launching:', ' '.join(command))
                started = time.perf_counter()
                subprocess.run(command, check=True)
                print(f'Baseline process finished in {(time.perf_counter() - started) / 60:.1f} min')
                """
            ),
            code(
                """
                baseline = json.loads(BASELINE_PATH.read_text())
                metrics = ('n', 'execution_accuracy', 'sql_valid_rate', 'sql_executable_rate', 'normalized_exact_match')
                print(json.dumps({key: baseline[key] for key in metrics}, indent=2))
                print()
                print('Predictions:')
                for row in baseline['rows'][:5]:
                    print()
                    print('Q:', row['question'])
                    print('Gold:', row['expected_sql'])
                    print('Generated:', row['generated'])
                    print('Execution correct:', row['execution_correct'])
                """
            ),
            markdown("""## 3. Optional context: compare available cloud reports on shared IDs"""),
            code(
                """
                from nemotron_ft_lab.evaluation import paired_execution_comparison

                baseline_by_id = {row['example_id']: row for row in baseline['rows']}
                for path in sorted((ARTIFACTS_DIR / 'evaluation').glob('baseline_api_*_text2sql_*.json')):
                    cloud = json.loads(path.read_text())
                    shared = [row['example_id'] for row in cloud['rows']]
                    if not set(shared).issubset(baseline_by_id):
                        continue
                    local_shared = {'rows': [baseline_by_id[item] for item in shared]}
                    result = paired_execution_comparison(cloud, local_shared)
                    print(path.name, json.dumps({
                        'cloud_accuracy': cloud['execution_accuracy'],
                        'local_bf16_accuracy': sum(r['execution_correct'] for r in local_shared['rows']) / len(shared),
                        'local_minus_cloud': result['absolute_execution_accuracy_gain'],
                    }, indent=2))
                """
            ),
            markdown(
                """
                ## Result contract

                Notebook 03 must reuse `baseline_local_bf16_text2sql.json` and the exact IDs in this report. A lower training loss or more parseable SQL is useful diagnostics, but the fine-tuning claim requires higher execution accuracy on the identical holdout.
                """
            ),
        ]
    ),
    "03_peft_lora.ipynb": notebook(
        [
            markdown(
                """
                # 03 — Official-recipe Text2SQL LoRA and paired evaluation

                This is the practical adaptation exercise. It follows NVIDIA's official Nemotron 3.5 Lightning Text2SQL cookbook: BIRD direct + reasoning data, native chat-template rendering, Hugging Face → Megatron conversion, packed-sequence LoRA from the shipped Lightning recipe, adapter merge, then vLLM serving.

                The workshop profile keeps 4,096 post-filter examples, 2,048-token packing, rank 32, and at most 64 steps. NVIDIA measures the full 12,544-example epoch at about 60 minutes on one H100 or 34 minutes on two H100s; this bounded profile leaves time to merge and evaluate. Two visible A100/H100 GPUs run TP1/EP2. One 80 GB GPU reduces to the checkpoint's single MTP head, as in NVIDIA's one-GPU runbook.

                **Required runtime:** use the NeMo container Jupyter through the Secure Link on host port **8889**.
                """
            ),
            code(
                """
                from pathlib import Path
                import json, os, subprocess, sys, time

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                BASELINE_PATH = ARTIFACTS_DIR / 'evaluation/baseline_local_bf16_text2sql.json'
                PEFT_REPORT_PATH = ARTIFACTS_DIR / 'evaluation/peft_text2sql.json'
                MEGATRON_BASE = Path('/workspace/storage/checkpoints/lightning35-megatron')
                LORA_ROOT = Path('/workspace/storage/checkpoints/bird-text2sql-lora')
                MERGED_MODEL = Path('/workspace/storage/checkpoints/bird-text2sql-lora-hf')
                RUN_TRAINING = True
                RUN_MERGE = True
                RUN_EVALUATION = True
                print('Repository:', ROOT)
                print('Artifacts:', ARTIFACTS_DIR)
                print('Python:', sys.executable)
                """
            ),
            markdown("""## 1. Guardrails, training data, and proof boundary"""),
            code(
                """
                subprocess.run([sys.executable, 'scripts/preflight.py', '--profile', 'peft'], check=True)
                if not BASELINE_PATH.exists():
                    raise RuntimeError('Run Notebook 02 first; the exact local BF16 baseline is required.')
                subprocess.run([
                    sys.executable, 'scripts/prepare_text2sql.py', '--output-dir', str(DATA_DIR),
                    '--training-only', '--max-train-samples', '4096', '--max-sequence-length', '2048',
                ], check=True)

                import torch
                baseline = json.loads(BASELINE_PATH.read_text())
                train_manifest = json.loads((DATA_DIR / 'training_manifest.json').read_text())
                print('Frozen baseline execution accuracy:', baseline['execution_accuracy'])
                print('Training rows:', train_manifest['training_examples'])
                print('Training mix:', train_manifest['source_distribution'])
                print('Training source:', train_manifest['official_cookbook'], '@', train_manifest['official_cookbook_revision'])
                print('Train/eval policy:', train_manifest['split_policy'])
                """
            ),
            markdown("""## 2. Convert the pinned BF16 checkpoint once"""),
            code(
                """
                from huggingface_hub import snapshot_download
                from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION

                PINNED_HF_MODEL = Path(snapshot_download(
                    repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                ))
                convert_cmd = [
                    sys.executable, 'scripts/convert_checkpoint.py',
                    '--hf-model', MODEL_ID, '--revision', MODEL_REVISION,
                    '--output', str(MEGATRON_BASE),
                ]
                if RUN_TRAINING:
                    started = time.perf_counter()
                    subprocess.run(convert_cmd, check=True)
                    print(f'Conversion stage: {(time.perf_counter() - started) / 60:.1f} min')
                else:
                    print('Would run:', ' '.join(convert_cmd))
                """
            ),
            markdown(
                """
                ## 3. Train with the shipped Lightning LoRA recipe

                The recipe—not this notebook—owns model-specific LoRA targets across Mamba, attention, routed experts, and shared experts. The wrapper changes only paths, portable all-to-all dispatch, topology, packing length, schedule, and rank. The compatibility hook preserves the live-validated Megatron padding-mask fix in the pinned stack. A run contract beside the checkpoint pins the data hash and all exposed settings; an incompatible resume is rejected instead of silently reusing stale state.
                """
            ),
            code(
                """
                VISIBLE_GPUS = torch.cuda.device_count()
                N_GPUS = int(os.environ.get('NEMOTRON_PEFT_NUM_GPUS', str(VISIBLE_GPUS)))
                if not 1 <= N_GPUS <= VISIBLE_GPUS:
                    raise RuntimeError(f'NEMOTRON_PEFT_NUM_GPUS must be between 1 and {VISIBLE_GPUS}.')
                if 128 % N_GPUS:
                    raise RuntimeError(f"{N_GPUS} ranks cannot evenly shard the model's 128 experts.")
                train_cmd = [
                    'torchrun', f'--nproc-per-node={N_GPUS}', 'scripts/train_peft.py',
                    '--megatron-checkpoint', str(MEGATRON_BASE),
                    '--data-dir', str(DATA_DIR), '--output-dir', str(LORA_ROOT),
                    '--sequence-length', '2048', '--global-batch-size', '32',
                    '--max-steps', '64', '--learning-rate', '1e-4', '--lora-rank', '32',
                ]
                print(f'Using {N_GPUS}/{VISIBLE_GPUS} GPU(s): TP=1, EP={N_GPUS}')
                print('Launch:', ' '.join(train_cmd))
                if RUN_TRAINING:
                    started = time.perf_counter()
                    subprocess.run(train_cmd, check=True)
                    print(f'LoRA stage: {(time.perf_counter() - started) / 60:.1f} min')
                """
            ),
            code(
                """
                marker = LORA_ROOT / 'latest_checkpointed_iteration.txt'
                if RUN_TRAINING:
                    if not marker.exists():
                        raise RuntimeError(f'Missing adapter marker: {marker}')
                    latest_step = int(marker.read_text().strip())
                    adapter_checkpoint = LORA_ROOT / f'iter_{latest_step:07d}'
                    print('Saved adapter:', adapter_checkpoint)
                else:
                    adapter_checkpoint = Path('/path/to/adapter')
                """
            ),
            markdown("""## 4. Merge the adapter to a standard Hugging Face checkpoint"""),
            code(
                """
                BRIDGE_DIR = Path(os.environ.get('MEGATRON_BRIDGE_DIR', '/workspace/storage/Megatron-Bridge'))
                merge_cmd = [
                    'torchrun', '--nproc-per-node=1', str(BRIDGE_DIR / 'examples/peft/merge_lora.py'),
                    '--lora-checkpoint', str(adapter_checkpoint),
                    '--hf-model-path', str(PINNED_HF_MODEL),
                    '--output', str(MERGED_MODEL), '--cpu',
                ]
                print('Merge:', ' '.join(merge_cmd))
                if RUN_MERGE:
                    started = time.perf_counter()
                    subprocess.run(merge_cmd, check=True)
                    if not (MERGED_MODEL / 'config.json').exists():
                        raise RuntimeError('Merge completed without config.json.')
                    print(f'Merge stage: {(time.perf_counter() - started) / 60:.1f} min')
                """
            ),
            markdown("""## 5. Evaluate the unchanged 100-row holdout with vLLM"""),
            code(
                """
                INFERENCE_GPUS = int(os.environ.get('NEMOTRON_INFERENCE_GPUS', '1'))
                eval_cmd = [
                    sys.executable, 'scripts/evaluate_vllm.py',
                    '--model', str(MERGED_MODEL), '--revision', '',
                    '--data-dir', str(DATA_DIR), '--output', str(PEFT_REPORT_PATH),
                    '--run-type', 'lora-peft-text2sql',
                    '--tensor-parallel-size', str(INFERENCE_GPUS),
                ]
                if RUN_EVALUATION:
                    started = time.perf_counter()
                    subprocess.run(eval_cmd, check=True)
                    print(f'Evaluation stage: {(time.perf_counter() - started) / 60:.1f} min')
                tuned = json.loads(PEFT_REPORT_PATH.read_text())
                """
            ),
            code(
                """
                from nemotron_ft_lab.evaluation import paired_execution_comparison

                comparison = paired_execution_comparison(baseline, tuned)
                comparison.update({
                    'baseline_execution_accuracy': baseline['execution_accuracy'],
                    'peft_execution_accuracy': tuned['execution_accuracy'],
                    'baseline_sql_valid_rate': baseline['sql_valid_rate'],
                    'peft_sql_valid_rate': tuned['sql_valid_rate'],
                })
                print(json.dumps(comparison, indent=2))
                if comparison['absolute_execution_accuracy_gain'] <= 0:
                    print('No held-out execution gain was demonstrated. Do not claim success from loss alone.')

                baseline_by_id = {row['example_id']: row for row in baseline['rows']}
                tuned_by_id = {row['example_id']: row for row in tuned['rows']}
                changed = [item for item in baseline_by_id if baseline_by_id[item]['execution_correct'] != tuned_by_id[item]['execution_correct']]
                for item in changed[:8]:
                    before, after = baseline_by_id[item], tuned_by_id[item]
                    print()
                    print('Q:', after['question'])
                    print('Gold:', after['expected_sql'])
                    print('Base:', before['generated'], 'correct=', before['execution_correct'])
                    print('LoRA:', after['generated'], 'correct=', after['execution_correct'])
                """
            ),
            markdown("""## 6. Compare specialized Lightning with hosted targets on shared IDs"""),
            code(
                """
                cloud_results = {}
                for path in sorted((ARTIFACTS_DIR / 'evaluation').glob('baseline_api_*_text2sql_*.json')):
                    cloud = json.loads(path.read_text())
                    ids = [row['example_id'] for row in cloud['rows']]
                    if not set(ids).issubset(tuned_by_id):
                        continue
                    tuned_shared = {'rows': [tuned_by_id[item] for item in ids]}
                    result = paired_execution_comparison(cloud, tuned_shared)
                    result.update({
                        'cloud_model': cloud['model'],
                        'cloud_execution_accuracy': cloud['execution_accuracy'],
                        'tuned_execution_accuracy_on_shared_ids': sum(r['execution_correct'] for r in tuned_shared['rows']) / len(ids),
                        'interpretation': 'tuned Lightning minus hosted target on identical Mini-Dev IDs',
                    })
                    cloud_results[path.name] = result
                    print()
                    print(path.name)
                    print(json.dumps(result, indent=2))
                target_path = ARTIFACTS_DIR / 'evaluation/peft_vs_cloud_text2sql.json'
                target_path.write_text(json.dumps(cloud_results, indent=2))
                print('Saved:', target_path)
                """
            ),
            markdown(
                """
                ## What counts as success

                A positive local BF16 → merged-LoRA execution delta shows useful task adaptation. A paired confidence interval above zero is stronger evidence; a wider interval crossing zero means the 100-row estimate is inconclusive. Reaching Ultra on the 25 shared API rows is a cost/size comparison for this task only, not a general model ranking.
                """
            ),
        ]
    ),
    "04_full_finetuning_design.ipynb": notebook(
        [
            markdown(
                """
                # 04 — Full-parameter Text2SQL SFT design (multi-GPU handoff)

                This notebook is deliberately design-only. Full SFT updates all 30B parameters; sparse activation reduces per-token compute but not the optimizer state for every trainable weight. A normal one-GPU Brev instance can run LoRA, not honest full-parameter AdamW training.

                The guarded driver uses NVIDIA's shipped `nemotron_3_5_lightning_sft_config`, BIRD packed sequences, and the same executable Mini-Dev proof contract. It is gated to NVIDIA's verified 4K full-SFT reference: 16×H100 80 GB (TP2/EP8) and at least 1,200 GiB aggregate visible VRAM. Any external run still needs target-node rehearsal.
                """
            ),
            code(
                """
                from pathlib import Path
                import json, os, sys

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                BASELINE_PATH = ARTIFACTS_DIR / 'evaluation/baseline_local_bf16_text2sql.json'
                MEGATRON_BASE = Path('/workspace/storage/checkpoints/lightning35-megatron')
                FULL_ROOT = Path('/workspace/storage/checkpoints/bird-text2sql-full')
                FULL_HF = Path('/workspace/storage/checkpoints/bird-text2sql-full-hf')
                DESIGN_ONLY = True
                """
            ),
            markdown("""## 1. Quantify the optimizer-state boundary"""),
            code(
                """
                TOTAL_PARAMETERS = 30_000_000_000
                memory = {
                    'bf16_weights_gib': TOTAL_PARAMETERS * 2 / 1024**3,
                    'bf16_gradients_gib': TOTAL_PARAMETERS * 2 / 1024**3,
                    'fp32_master_and_adam_moments_gib': TOTAL_PARAMETERS * 12 / 1024**3,
                }
                memory['subtotal_before_activations_gib'] = sum(memory.values())
                print(json.dumps(memory, indent=2))
                """
            ),
            code(
                """
                from nemotron_ft_lab.hardware import inspect_cuda, validate_full_sft_hardware

                inventory = inspect_cuda()
                print(json.dumps(inventory.as_dict(), indent=2))
                try:
                    validate_full_sft_hardware(inventory)
                    print('Full-SFT hardware gate passed; this notebook still does not launch it.')
                except RuntimeError as exc:
                    print('Design-only on this allocation:', exc)
                """
            ),
            markdown("""## 2. Produce the external handoff command"""),
            code(
                """
                TARGET_GPUS = 16
                full_cmd = [
                    'torchrun', f'--nproc-per-node={TARGET_GPUS}', 'scripts/train_full.py',
                    '--megatron-checkpoint', str(MEGATRON_BASE),
                    '--data-dir', str(DATA_DIR), '--output-dir', str(FULL_ROOT),
                    '--sequence-length', '2048', '--global-batch-size', '32',
                    '--max-steps', '64', '--learning-rate', '5e-6',
                ]
                assert DESIGN_ONLY
                print('Command for a qualifying, separately rehearsed allocation:')
                print(' '.join(full_cmd))
                print('Trainable scope: every model parameter; no PEFT config.')
                """
            ),
            markdown("""## 3. Require the same evidence after export"""),
            code(
                """
                required_evidence = {
                    'base_checkpoint': str(MEGATRON_BASE),
                    'training_data_manifest': str(DATA_DIR / 'training_manifest.json'),
                    'frozen_local_baseline': str(BASELINE_PATH),
                    'frozen_evaluation_manifest': str(DATA_DIR / 'evaluation_manifest.json'),
                    'export_command': f'python scripts/export_full_checkpoint.py --megatron-checkpoint {FULL_ROOT} --output {FULL_HF}',
                    'evaluation': 'run scripts/evaluate_vllm.py, then report paired execution-accuracy delta and 95% bootstrap CI',
                    'topology': 'record GPU type/count, TP, EP, DP, and all software revisions',
                }
                print(json.dumps(required_evidence, indent=2))
                print()
                print('No training, export, or evaluation was launched by this notebook.')
                """
            ),
            markdown(
                """
                ## Decision rule

                Prefer LoRA when Notebook 03 meets the execution target: it is faster, cheaper, easier to version, and preserves the base weights. Escalate to full SFT only after repeated LoRA experiments show a material ceiling and the additional accuracy justifies a multi-GPU training, checkpoint, and serving lifecycle.
                """
            ),
        ]
    ),
}


def main() -> None:
    target = ROOT / "notebooks"
    target.mkdir(exist_ok=True)
    for name, payload in NOTEBOOKS.items():
        path = target / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
