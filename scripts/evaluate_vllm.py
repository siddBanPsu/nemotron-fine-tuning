#!/usr/bin/env python3
"""Evaluate a local Nemotron checkpoint on frozen BIRD Mini-Dev rows with vLLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION
from nemotron_ft_lab.data import build_messages, read_jsonl
from nemotron_ft_lab.evaluation import save_report, score_predictions


def parse_args() -> argparse.Namespace:
    artifacts = Path(os.environ.get("NEMOTRON_ARTIFACTS_DIR", "artifacts"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--data-dir", type=Path, default=artifacts / "data/bird-text2sql")
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-type", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    evaluation_path = args.evaluation or args.data_dir / "evaluation.jsonl"
    rows = read_jsonl(evaluation_path)
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
    started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    generated_rows = [
        {**row, "generated": output.outputs[0].text.strip()}
        for row, output in zip(rows, outputs, strict=True)
    ]
    report = score_predictions(generated_rows, data_dir=args.data_dir)
    report.update(
        {
            "wall_time_seconds": time.perf_counter() - started,
            "backend": "vllm",
            "precision": "BF16",
            "checkpoint_revision": args.revision or "local-merged-checkpoint",
            "tensor_parallel_size": args.tensor_parallel_size,
            "evaluation_manifest": json.loads(
                (args.data_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
            ),
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
