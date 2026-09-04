#!/usr/bin/env python3
"""Regenerate the five checked-in notebooks from reviewable Python strings."""

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


def lora_notebook(*, profile_name: str, title: str, intro: str, default_gpus: int) -> dict:
    return notebook(
        [
            markdown(f"""
                # {title}

                {intro}

                The proof boundary is unchanged: compare the merged adapter with the **matching**
                local BF16 base on the same frozen 100 BIRD Mini-Dev IDs. Training loss is a
                diagnostic; executable held-out SQL is the success metric.

                **Required runtime:** use the NeMo container Jupyter opened by
                `launchable/setup.sh` through the Secure Link on host port **8889**.
                """),
            code(f"""
                import json
                import os
                import subprocess
                import sys
                import time
                from pathlib import Path

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))

                from nemotron_ft_lab.model_profiles import get_model_profile

                PROFILE = get_model_profile('{profile_name}')
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                EVAL_DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                TRAIN_DATA_DIR = EVAL_DATA_DIR / 'profiles' / PROFILE.name
                EVALUATION_DIR = ARTIFACTS_DIR / 'evaluation' / PROFILE.artifact_slug
                EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
                BASELINE_PATH = EVALUATION_DIR / 'baseline_local_bf16_text2sql.json'
                PEFT_REPORT_PATH = EVALUATION_DIR / 'peft_text2sql.json'
                CHECKPOINT_ROOT = Path('/workspace/storage/checkpoints')
                MEGATRON_BASE = CHECKPOINT_ROOT / PROFILE.megatron_checkpoint_name
                LORA_ROOT = CHECKPOINT_ROOT / PROFILE.lora_checkpoint_name
                MERGED_MODEL = CHECKPOINT_ROOT / PROFILE.merged_checkpoint_name

                RUN_TRAINING = True
                RESUME_IF_AVAILABLE = True
                RUN_MERGE = True
                RUN_EVALUATION = True
                print(json.dumps(PROFILE.as_dict(), indent=2))
                print('Python:', sys.executable)
                print('Artifacts:', ARTIFACTS_DIR)
                """),
            markdown("""
                ## 1. Verify hardware, matching baseline, and profile-specific training data

                The training manifest binds the model revision, native system prompt, reasoning
                mode, row count, and data hash. The driver rejects a mismatched manifest.
                """),
            code("""
                subprocess.run([
                    sys.executable, 'scripts/preflight.py', '--profile', 'peft',
                    '--model-profile', PROFILE.name,
                ], check=True)
                if not BASELINE_PATH.exists():
                    raise RuntimeError(
                        f'Run Notebook 02 with MODEL_PROFILE_NAME={PROFILE.name!r} first: {BASELINE_PATH}'
                    )
                subprocess.run([
                    sys.executable, 'scripts/prepare_text2sql.py',
                    '--model-profile', PROFILE.name,
                    '--output-dir', str(TRAIN_DATA_DIR), '--training-only',
                    '--max-train-samples', str(PROFILE.default_train_samples),
                    '--max-sequence-length', '2048',
                ], check=True)

                import torch

                baseline = json.loads(BASELINE_PATH.read_text())
                if baseline.get('model_profile') != PROFILE.name:
                    raise RuntimeError('Baseline model profile does not match this LoRA run.')
                train_manifest = json.loads((TRAIN_DATA_DIR / 'training_manifest.json').read_text())
                print('Frozen baseline execution accuracy:', baseline['execution_accuracy'])
                print('Training rows:', train_manifest['training_examples'])
                print('Training mix:', train_manifest['source_distribution'])
                print('Reasoning traces enabled:', train_manifest['include_reasoning'])
                print('Train/eval policy:', train_manifest['split_policy'])
                """),
            markdown("""
                ## 2. Download and convert the pinned BF16 checkpoint once

                Both the Hugging Face snapshot and converted Megatron checkpoint are reused.
                A cold workshop should prefetch these in setup rather than spending participant
                time on network and conversion work.
                """),
            code("""
                from huggingface_hub import snapshot_download

                stage_times = {}
                started = time.perf_counter()
                PINNED_HF_MODEL = Path(snapshot_download(
                    repo_id=PROFILE.model_id, revision=PROFILE.revision,
                ))
                convert_cmd = [
                    sys.executable, 'scripts/convert_checkpoint.py',
                    '--model-profile', PROFILE.name,
                    '--hf-model', PROFILE.model_id, '--revision', PROFILE.revision,
                    '--output', str(MEGATRON_BASE),
                ]
                if RUN_TRAINING:
                    subprocess.run(convert_cmd, check=True)
                else:
                    print('Would run:', ' '.join(convert_cmd))
                stage_times['snapshot_and_conversion_minutes'] = (time.perf_counter() - started) / 60
                print(json.dumps(stage_times, indent=2))
                """),
            markdown("""
                ## 3. Train with the pinned model-family recipe

                Nano uses Megatron-Bridge's one-GPU H100 BF16 recipe and saves every eight
                steps. Lightning uses its official model-specific LoRA targets and expert
                parallelism. `--resume` is added only when a compatible checkpoint and run
                contract already exist; changing the experiment requires a new output path.
                """),
            code(f"""
                VISIBLE_GPUS = torch.cuda.device_count()
                N_GPUS = int(os.environ.get('NEMOTRON_PEFT_NUM_GPUS', '{default_gpus}'))
                if N_GPUS not in PROFILE.peft_world_sizes:
                    raise RuntimeError(f'{{PROFILE.name}} supports PEFT world sizes {{PROFILE.peft_world_sizes}}, not {{N_GPUS}}.')
                if N_GPUS > VISIBLE_GPUS:
                    raise RuntimeError(f'Requested {{N_GPUS}} GPU(s), but only {{VISIBLE_GPUS}} are visible.')
                train_cmd = [
                    'torchrun', f'--nproc-per-node={{N_GPUS}}', 'scripts/train_peft.py',
                    '--model-profile', PROFILE.name,
                    '--hf-model', PROFILE.model_id, '--revision', PROFILE.revision,
                    '--megatron-checkpoint', str(MEGATRON_BASE),
                    '--data-dir', str(TRAIN_DATA_DIR), '--output-dir', str(LORA_ROOT),
                    '--sequence-length', '2048',
                    '--global-batch-size', str(PROFILE.default_global_batch_size),
                    '--max-steps', str(PROFILE.default_max_steps),
                    '--learning-rate', '1e-4', '--lora-rank', '32',
                ]
                marker = LORA_ROOT / 'latest_checkpointed_iteration.txt'
                if RESUME_IF_AVAILABLE and marker.exists():
                    train_cmd.append('--resume')
                print(f'Using {{N_GPUS}}/{{VISIBLE_GPUS}} GPU(s).')
                print('Launch:', ' '.join(train_cmd))
                if RUN_TRAINING:
                    started = time.perf_counter()
                    subprocess.run(train_cmd, check=True)
                    stage_times['lora_training_minutes'] = (time.perf_counter() - started) / 60
                """),
            code("""
                marker = LORA_ROOT / 'latest_checkpointed_iteration.txt'
                if RUN_TRAINING:
                    if not marker.exists():
                        raise RuntimeError(f'Missing adapter marker: {marker}')
                    latest_step = int(marker.read_text().strip())
                    adapter_checkpoint = LORA_ROOT / f'iter_{latest_step:07d}'
                    if not adapter_checkpoint.is_dir():
                        raise RuntimeError(f'Missing adapter checkpoint: {adapter_checkpoint}')
                    print('Saved adapter:', adapter_checkpoint)
                else:
                    adapter_checkpoint = Path('/path/to/adapter')
                """),
            markdown("""## 4. Merge the adapter to a standard Hugging Face checkpoint"""),
            code("""
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
                    stage_times['cpu_merge_minutes'] = (time.perf_counter() - started) / 60
                """),
            markdown("""## 5. Evaluate the unchanged 100-row holdout with vLLM"""),
            code("""
                INFERENCE_GPUS = int(os.environ.get('NEMOTRON_INFERENCE_GPUS', '1'))
                eval_cmd = [
                    sys.executable, 'scripts/evaluate_vllm.py',
                    '--model-profile', PROFILE.name,
                    '--model', str(MERGED_MODEL), '--revision', '',
                    '--data-dir', str(EVAL_DATA_DIR), '--output', str(PEFT_REPORT_PATH),
                    '--run-type', f'lora-peft-text2sql-{PROFILE.name}',
                    '--tensor-parallel-size', str(INFERENCE_GPUS),
                ]
                if RUN_EVALUATION:
                    started = time.perf_counter()
                    subprocess.run(eval_cmd, check=True)
                    stage_times['merged_evaluation_minutes'] = (time.perf_counter() - started) / 60
                tuned = json.loads(PEFT_REPORT_PATH.read_text())
                print('Measured stages:', json.dumps(stage_times, indent=2))
                """),
            code("""
                from nemotron_ft_lab.evaluation import paired_execution_comparison

                comparison = paired_execution_comparison(baseline, tuned)
                comparison.update({
                    'model_profile': PROFILE.name,
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
                changed = [
                    item for item in baseline_by_id
                    if baseline_by_id[item]['execution_correct'] != tuned_by_id[item]['execution_correct']
                ]
                for item in changed[:8]:
                    before, after = baseline_by_id[item], tuned_by_id[item]
                    print()
                    print('Q:', after['question'])
                    print('Gold:', after['expected_sql'])
                    print('Base:', before['generated'], 'correct=', before['execution_correct'])
                    print('LoRA:', after['generated'], 'correct=', after['execution_correct'])
                """),
            markdown("""## 6. Optional context: compare with hosted targets on shared IDs"""),
            code("""
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
                        'tuned_execution_accuracy_on_shared_ids': (
                            sum(row['execution_correct'] for row in tuned_shared['rows']) / len(ids)
                        ),
                        'interpretation': f'tuned {PROFILE.name} minus hosted target on shared IDs',
                    })
                    cloud_results[path.name] = result
                    print(path.name, json.dumps(result, indent=2))
                target_path = EVALUATION_DIR / 'peft_vs_cloud_text2sql.json'
                target_path.write_text(json.dumps(cloud_results, indent=2) + '\\n')
                print('Saved:', target_path)
                """),
            markdown("""
                ## What counts as success

                A positive base-to-LoRA execution delta demonstrates task adaptation for this
                run. A paired 95% confidence interval above zero is stronger evidence. Matching
                a hosted larger model on only 25 shared rows is useful context, not a general
                model ranking. Always report hardware, cold/warm cache state, stage times, and
                the exact profile beside the accuracy result.
                """),
        ]
    )


NOTEBOOKS = {
    "01_cloud_api_baseline.ipynb": notebook(
        [
            markdown("""
                # 01 — Hosted Lightning and Ultra Text2SQL targets

                This CPU-only notebook evaluates NVIDIA-hosted Nemotron 3.5 Lightning and
                Nemotron 3 Ultra on the same frozen BIRD Mini-Dev SQLite questions. The default
                is 25 requests per model—50 total. Responses checkpoint after every success and
                resume safely after a 429 or interruption.

                The primary score is execution accuracy. Hosted NVFP4 services are useful task
                targets, but the causal fine-tuning comparison is always a local BF16 base versus
                its own merged LoRA checkpoint in Notebooks 02–04.
                """),
            code("""
                import hashlib
                import json
                import os
                import subprocess
                import sys
                import time
                from pathlib import Path

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                print('Repository:', ROOT)
                print('Artifacts:', ARTIFACTS_DIR)
                """),
            markdown("""
                ## 1. API preflight and public/private endpoint switch

                Copy `config/api.local.toml.example` to the ignored `config/api.local.toml` for
                an internal OpenAI-compatible endpoint. Set `API_PROFILE_OVERRIDE` to `public` or
                `local`; leave it `None` for automatic selection. Keys never belong in TOML.
                """),
            code("""
                subprocess.run([sys.executable, 'scripts/preflight.py', '--profile', 'api'], check=True)
                from nemotron_ft_lab.api_config import load_nvidia_api_config

                API_PROFILE_OVERRIDE = None
                API_CONFIG = load_nvidia_api_config(ROOT, profile=API_PROFILE_OVERRIDE)
                print('API profile:', API_CONFIG.profile_name, f'({API_CONFIG.source_label})')
                print('Endpoint:', API_CONFIG.base_url)
                print('Lightning:', API_CONFIG.lightning.model_id, '->', API_CONFIG.lightning.served_variant)
                print('Ultra:', API_CONFIG.ultra.model_id, '->', API_CONFIG.ultra.served_variant)
                print('Pacing:', API_CONFIG.requests_per_minute, 'requests/minute')
                """),
            markdown("""## 2. Freeze the official executable Mini-Dev evaluation bundle"""),
            code("""
                subprocess.run([
                    sys.executable, 'scripts/prepare_text2sql.py',
                    '--output-dir', str(DATA_DIR), '--evaluation-only',
                ], check=True)

                from nemotron_ft_lab.constants import DEFAULT_API_EVALUATION_SIZE, DEFAULT_SEED
                from nemotron_ft_lab.data import balanced_evaluation_subset, build_messages, read_jsonl

                all_eval_rows = read_jsonl(DATA_DIR / 'evaluation.jsonl')
                CLOUD_EVAL_SIZE = DEFAULT_API_EVALUATION_SIZE
                cloud_rows = balanced_evaluation_subset(all_eval_rows, size=CLOUD_EVAL_SIZE, seed=DEFAULT_SEED + 1)
                manifest = json.loads((DATA_DIR / 'evaluation_manifest.json').read_text())
                print('Frozen local holdout:', len(all_eval_rows))
                print('Cloud rows per model:', len(cloud_rows))
                print('Cloud requests across Lightning + Ultra:', 2 * len(cloud_rows))
                print('Difficulty mix:', manifest['difficulty_distribution'])
                print('Databases:', len(manifest['database_distribution']))
                print(build_messages(cloud_rows[0])[1]['content'][:1800])
                """),
            markdown("""
                ## 3. Authenticate without storing the key

                Prefer exporting `NVIDIA_API_KEY` before Jupyter starts. Native `input()` is the
                fallback because some password widgets block paste; input is briefly visible and
                then cleared.
                """),
            code("""
                from IPython.display import clear_output
                from openai import OpenAI

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
                print('API client configured; prompt cleared.' if used_native_prompt else 'API client configured from NVIDIA_API_KEY.')
                """),
            markdown("""## 4. Run both hosted models and execute their generated SQL"""),
            code("""
                from nemotron_ft_lab.evaluation import (
                    generate_nvidia_api_predictions,
                    paired_execution_comparison,
                    save_report,
                    score_predictions,
                )

                CLOUD_MODELS = {'lightning': API_CONFIG.lightning, 'ultra': API_CONFIG.ultra}
                api_reports = {}
                for name, spec in CLOUD_MODELS.items():
                    endpoint_model_fingerprint = hashlib.sha256(
                        f'{API_CONFIG.base_url}\\0{spec.model_id}\\0{manifest["evaluation_sha256"]}\\0v1'.encode()
                    ).hexdigest()[:12]
                    artifact_name = f'{API_CONFIG.artifact_prefix}{name}_{endpoint_model_fingerprint}'
                    resume_path = ARTIFACTS_DIR / f'evaluation/api_{artifact_name}_text2sql_v1_{CLOUD_EVAL_SIZE}.jsonl'
                    print(f'Evaluating {name}: {spec.model_id}')
                    started = time.perf_counter()
                    generated = generate_nvidia_api_predictions(
                        client, cloud_rows, model=spec.model_id, resume_path=resume_path,
                        requests_per_minute=API_CONFIG.requests_per_minute,
                    )
                    report = score_predictions(generated, data_dir=DATA_DIR)
                    report.update({
                        'wall_time_seconds': time.perf_counter() - started,
                        'endpoint': API_CONFIG.base_url, 'api_profile': API_CONFIG.profile_name,
                        'served_variant': spec.served_variant, 'precision': 'NVFP4',
                        'prompt_protocol': 'bird-schema-question-evidence-v1',
                    })
                    report_path = ARTIFACTS_DIR / f'evaluation/baseline_api_{artifact_name}_text2sql_{CLOUD_EVAL_SIZE}.json'
                    save_report(report_path, report, model=spec.model_id, run_type=f'hosted-{name}-text2sql')
                    api_reports[name] = report

                metrics = ('n', 'execution_accuracy', 'sql_valid_rate', 'sql_executable_rate', 'normalized_exact_match')
                print({name: {key: report[key] for key in metrics} for name, report in api_reports.items()})
                """),
            code("""
                comparison = paired_execution_comparison(api_reports['lightning'], api_reports['ultra'])
                comparison['interpretation'] = 'Ultra minus Lightning on identical hosted requests'
                print(json.dumps(comparison, indent=2))
                for name, report in api_reports.items():
                    print(f'\\n{name.title()} examples:')
                    for row in report['rows'][:3]:
                        print('\\nQ:', row['question'])
                        print('Gold:', row['expected_sql'])
                        print('Generated:', row['generated'])
                        print('Execution correct:', row['execution_correct'])
                """),
            markdown("""
                ## Result contract

                These scores are secondary targets. Run Notebook 02 once for each local profile
                you intend to tune. Notebook 03 compares Nano with Nano; Notebook 04 compares
                Lightning with Lightning. Cross-model comparisons use only shared IDs and are
                labeled as context rather than causal fine-tuning evidence.
                """),
        ]
    ),
    "02_local_bf16_baseline.ipynb": notebook(
        [
            markdown("""
                # 02 — Local BF16 baseline for either model profile

                This notebook evaluates the exact pinned local BF16 checkpoint that a later LoRA
                notebook will tune. The default is the one-GPU Nano workshop profile. Change one
                variable to create the separate Lightning advanced baseline.

                **Required runtime:** use the NeMo container Jupyter opened by
                `launchable/setup.sh` through the Secure Link on host port **8889**.
                """),
            code("""
                import json
                import os
                import subprocess
                import sys
                import time
                from pathlib import Path

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                from nemotron_ft_lab.model_profiles import get_model_profile

                MODEL_PROFILE_NAME = os.environ.get('NEMOTRON_MODEL_PROFILE', 'nano9b_workshop')
                # To prepare Notebook 04 instead, set: MODEL_PROFILE_NAME = 'lightning35_advanced'
                PROFILE = get_model_profile(MODEL_PROFILE_NAME)
                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql'
                EVALUATION_DIR = ARTIFACTS_DIR / 'evaluation' / PROFILE.artifact_slug
                EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
                BASELINE_PATH = EVALUATION_DIR / 'baseline_local_bf16_text2sql.json'
                print(json.dumps(PROFILE.as_dict(), indent=2))
                print('Python:', sys.executable)
                print('Baseline report:', BASELINE_PATH)
                """),
            markdown("""## 1. Verify the container, GPU, model identity, and frozen holdout"""),
            code("""
                subprocess.run([
                    sys.executable, 'scripts/preflight.py', '--profile', 'inference',
                    '--model-profile', PROFILE.name,
                ], check=True)
                subprocess.run([
                    sys.executable, 'scripts/prepare_text2sql.py',
                    '--output-dir', str(DATA_DIR), '--evaluation-only',
                ], check=True)

                import torch

                from nemotron_ft_lab.data import read_jsonl

                eval_rows = read_jsonl(DATA_DIR / 'evaluation.jsonl')
                manifest = json.loads((DATA_DIR / 'evaluation_manifest.json').read_text())
                print('Model:', PROFILE.model_id)
                print('Pinned revision:', PROFILE.revision)
                print('System prompt:', repr(PROFILE.system_prompt))
                print('GPU(s):', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
                print('Evaluation rows:', len(eval_rows), manifest['difficulty_distribution'])
                """),
            markdown("""
                ## 2. Run local vLLM in an isolated process

                Process exit releases GPU memory before training. Raw predictions are saved
                before SQL scoring and reused only when model, profile, prompt, Mamba cache,
                generation settings, and evaluation hash all match. Nano automatically uses the
                model-card-required float32 Mamba SSM cache and `/no_think` prompt.
                """),
            code("""
                INFERENCE_GPUS = int(os.environ.get('NEMOTRON_INFERENCE_GPUS', '1'))
                if not 1 <= INFERENCE_GPUS <= torch.cuda.device_count():
                    raise RuntimeError(f'NEMOTRON_INFERENCE_GPUS must be between 1 and {torch.cuda.device_count()}.')
                command = [
                    sys.executable, 'scripts/evaluate_vllm.py',
                    '--model-profile', PROFILE.name,
                    '--model', PROFILE.model_id, '--revision', PROFILE.revision,
                    '--data-dir', str(DATA_DIR), '--output', str(BASELINE_PATH),
                    '--run-type', f'untuned-local-bf16-text2sql-{PROFILE.name}',
                    '--tensor-parallel-size', str(INFERENCE_GPUS),
                ]
                print('Launching:', ' '.join(command))
                started = time.perf_counter()
                subprocess.run(command, check=True)
                print(f'Baseline process finished in {(time.perf_counter() - started) / 60:.1f} min')
                """),
            code("""
                baseline = json.loads(BASELINE_PATH.read_text())
                metrics = ('n', 'execution_accuracy', 'sql_valid_rate', 'sql_executable_rate', 'normalized_exact_match')
                print(json.dumps({key: baseline[key] for key in metrics}, indent=2))
                for row in baseline['rows'][:5]:
                    print('\\nQ:', row['question'])
                    print('Gold:', row['expected_sql'])
                    print('Generated:', row['generated'])
                    print('Execution correct:', row['execution_correct'])
                """),
            markdown("""
                ## Result contract

                Notebook 03 or 04 must reuse this profile-namespaced report and the exact IDs in
                it. Switching profiles here writes a different report; it never overwrites or
                masquerades as the other model's baseline.
                """),
        ]
    ),
    "03_nano9b_workshop_lora.ipynb": lora_notebook(
        profile_name="nano9b_workshop",
        title="03 — Nano 9B v2 one-GPU Text2SQL LoRA workshop",
        intro=(
            "This is the default practical path: 2,048 direct-SQL training rows, 2,048-token "
            "packing, rank 32, GBS 32, and at most 32 steps on one GPU. It targets a warm-cache "
            "end-to-end workshop run within about an hour, but that target remains unverified "
            "until the exact Brev SKU is rehearsed. The pinned recipe supports one H100; the "
            "repository conservatively recommends an 80 GB A100/H100 and treats 48 GB as experimental."
        ),
        default_gpus=1,
    ),
    "04_lightning_advanced_lora.ipynb": lora_notebook(
        profile_name="lightning35_advanced",
        title="04 — Lightning 3.5 advanced two-GPU Text2SQL LoRA",
        intro=(
            "This preserves the official-recipe Lightning exercise for users who can allocate "
            "two 80 GB GPUs and wait. It uses 4,096 direct plus reasoning rows, 2,048-token "
            "packing, rank 32, GBS 32, and at most 64 steps. A measured warm-cache 2xA100 run "
            "took 6 h 49 min for train, merge, and evaluation; this is not the one-hour workshop path."
        ),
        default_gpus=2,
    ),
    "05_full_finetuning_design.ipynb": notebook(
        [
            markdown("""
                # 05 — Lightning full-parameter Text2SQL SFT design

                This notebook is deliberately design-only. Full SFT updates all 30B Lightning
                parameters; sparse activation reduces per-token compute but not optimizer state.
                A normal one- or two-GPU Brev instance cannot run honest full-parameter AdamW.

                The guarded driver uses NVIDIA's shipped Lightning full-SFT recipe and is gated
                to the verified 16×H100 80 GB topology (TP2/EP8) with at least 1,200 GiB aggregate
                VRAM. Nano full SFT is outside this repository's workshop contract.
                """),
            code("""
                import json
                import os
                import sys
                from pathlib import Path

                ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
                os.chdir(ROOT)
                sys.path.insert(0, str(ROOT / 'src'))
                from nemotron_ft_lab.model_profiles import LIGHTNING35_ADVANCED as PROFILE

                ARTIFACTS_DIR = Path(os.environ.get('NEMOTRON_ARTIFACTS_DIR', ROOT / 'artifacts')).expanduser().resolve()
                DATA_DIR = ARTIFACTS_DIR / 'data/bird-text2sql/profiles' / PROFILE.name
                BASELINE_PATH = ARTIFACTS_DIR / 'evaluation' / PROFILE.artifact_slug / 'baseline_local_bf16_text2sql.json'
                MEGATRON_BASE = Path('/workspace/storage/checkpoints') / PROFILE.megatron_checkpoint_name
                FULL_ROOT = Path('/workspace/storage/checkpoints/bird-text2sql-lightning35-full')
                FULL_HF = Path('/workspace/storage/checkpoints/bird-text2sql-lightning35-full-hf')
                DESIGN_ONLY = True
                """),
            markdown("""## 1. Quantify the optimizer-state boundary"""),
            code("""
                TOTAL_PARAMETERS = 30_000_000_000
                memory = {
                    'bf16_weights_gib': TOTAL_PARAMETERS * 2 / 1024**3,
                    'bf16_gradients_gib': TOTAL_PARAMETERS * 2 / 1024**3,
                    'fp32_master_and_adam_moments_gib': TOTAL_PARAMETERS * 12 / 1024**3,
                }
                memory['subtotal_before_activations_gib'] = sum(memory.values())
                print(json.dumps(memory, indent=2))
                """),
            code("""
                from nemotron_ft_lab.hardware import inspect_cuda, validate_full_sft_hardware

                inventory = inspect_cuda()
                print(json.dumps(inventory.as_dict(), indent=2))
                try:
                    validate_full_sft_hardware(inventory)
                    print('Full-SFT hardware gate passed; this notebook still does not launch it.')
                except RuntimeError as exc:
                    print('Design-only on this allocation:', exc)
                """),
            markdown("""## 2. Produce the external handoff command"""),
            code("""
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
                """),
            markdown("""## 3. Require the same evidence after export"""),
            code("""
                required_evidence = {
                    'base_checkpoint': str(MEGATRON_BASE),
                    'training_data_manifest': str(DATA_DIR / 'training_manifest.json'),
                    'frozen_local_baseline': str(BASELINE_PATH),
                    'export_command': f'python scripts/export_full_checkpoint.py --megatron-checkpoint {FULL_ROOT} --output {FULL_HF}',
                    'evaluation': 'run scripts/evaluate_vllm.py, then report paired execution-accuracy delta and 95% bootstrap CI',
                    'topology': 'record GPU type/count, TP, EP, DP, software revisions, and cold/warm cache state',
                }
                print(json.dumps(required_evidence, indent=2))
                print('No training, export, or evaluation was launched by this notebook.')
                """),
            markdown("""
                ## Decision rule

                Prefer LoRA when it meets the execution target. Escalate to full SFT only after
                repeated controlled LoRA experiments show a material ceiling and the gain can
                justify a separately operated multi-GPU training and serving lifecycle.
                """),
        ]
    ),
}


OBSOLETE_NOTEBOOKS = (
    "03_peft_lora.ipynb",
    "04_full_finetuning_design.ipynb",
)


def main() -> None:
    target = ROOT / "notebooks"
    target.mkdir(exist_ok=True)
    for name in OBSOLETE_NOTEBOOKS:
        path = target / name
        if path.exists():
            path.unlink()
            print(f"Removed obsolete {path.relative_to(ROOT)}")
    for name, payload in NOTEBOOKS.items():
        path = target / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
