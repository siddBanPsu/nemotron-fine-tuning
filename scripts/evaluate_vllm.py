#!/usr/bin/env python3
"""Evaluate a local Nemotron checkpoint on frozen BIRD Mini-Dev rows with vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION
from nemotron_ft_lab.data import build_messages, read_jsonl, write_jsonl
from nemotron_ft_lab.evaluation import save_report, score_predictions


def parse_args() -> argparse.Namespace:
    artifacts = Path(os.environ.get("NEMOTRON_ARTIFACTS_DIR", "artifacts"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--data-dir", type=Path, default=artifacts / "data/bird-text2sql")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--run-type", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--force-generation", action="store_true")
    return parser.parse_args()


def _model_cache_identity(model: str, revision: str) -> str:
    path = Path(model).expanduser()
    if not path.is_dir():
        return f"hub:{model}@{revision}"
    files = sorted(
        candidate
        for pattern in ("*.safetensors", "*.bin", "*.index.json", "config.json")
        for candidate in path.glob(pattern)
        if candidate.is_file()
    )
    if not files:
        raise RuntimeError(f"Local model directory has no identifiable checkpoint files: {path}")
    inventory = [
        (candidate.name, candidate.stat().st_size, candidate.stat().st_mtime_ns) for candidate in files
    ]
    return "local-stat:" + hashlib.sha256(json.dumps(inventory).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    evaluation_path = args.evaluation or args.data_dir / "evaluation.jsonl"
    rows = read_jsonl(evaluation_path)
    evaluation_manifest = json.loads((args.data_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    predictions_path = args.predictions_output or args.output.with_name(
        f"{args.output.stem}_predictions.jsonl"
    )
    predictions_manifest_path = predictions_path.with_suffix(predictions_path.suffix + ".manifest.json")
    generation_contract = {
        "model": args.model,
        "revision": args.revision or "local-merged-checkpoint",
        "model_cache_identity": _model_cache_identity(args.model, args.revision),
        "evaluation_sha256": evaluation_manifest["evaluation_sha256"],
        "prompt_protocol": "bird-schema-question-evidence-v1",
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "bfloat16",
    }
    expected_ids = [str(row["example_id"]) for row in rows]
    can_reuse = predictions_path.is_file() and predictions_manifest_path.is_file()
    generated_rows: list[dict]
    if can_reuse and not args.force_generation:
        cached_contract = json.loads(predictions_manifest_path.read_text(encoding="utf-8"))
        generated_rows = read_jsonl(predictions_path)
        cached_ids = [str(row["example_id"]) for row in generated_rows]
        contract_matches = all(
            cached_contract.get(key) == value for key, value in generation_contract.items()
        )
        if not contract_matches or cached_ids != expected_ids:
            raise RuntimeError(
                f"Cached predictions do not match this model/evaluation: {predictions_path}. "
                "Pass --force-generation to replace them."
            )
        print(f"Reusing {len(generated_rows)} raw predictions from {predictions_path}")
        generation_seconds = float(cached_contract.get("generation_wall_time_seconds", 0.0))
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision or None,
            trust_remote_code=True,
        )
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(row),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in rows
        ]
        engine_kwargs = {
            "model": args.model,
            "tokenizer": args.model,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
        }
        if args.revision:
            engine_kwargs["revision"] = args.revision
            engine_kwargs["tokenizer_revision"] = args.revision
        generation_started = time.perf_counter()
        llm = LLM(**engine_kwargs)
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )
        generation_seconds = time.perf_counter() - generation_started
        generated_rows = [
            {**row, "generated": output.outputs[0].text.strip()}
            for row, output in zip(rows, outputs, strict=True)
        ]
        write_jsonl(predictions_path, generated_rows)
        predictions_manifest_path.write_text(
            json.dumps({**generation_contract, "generation_wall_time_seconds": generation_seconds}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved raw predictions before scoring: {predictions_path}")

    scoring_started = time.perf_counter()
    report = score_predictions(generated_rows, data_dir=args.data_dir)
    scoring_seconds = time.perf_counter() - scoring_started
    report.update(
        {
            "wall_time_seconds": generation_seconds + scoring_seconds,
            "generation_wall_time_seconds": generation_seconds,
            "scoring_wall_time_seconds": scoring_seconds,
            "backend": "vllm",
            "precision": "BF16",
            "checkpoint_revision": args.revision or "local-merged-checkpoint",
            "tensor_parallel_size": args.tensor_parallel_size,
            "raw_predictions": str(predictions_path),
            "evaluation_manifest": evaluation_manifest,
        }
    )
    save_report(args.output, report, model=args.model, run_type=args.run_type)
    summary = {
        key: report[key]
        for key in (
            "n",
            "execution_accuracy",
            "sql_valid_rate",
            "sql_executable_rate",
            "normalized_exact_match",
            "wall_time_seconds",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
