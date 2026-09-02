#!/usr/bin/env python3
"""Idempotently import the pinned Hugging Face checkpoint into Megatron format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch
from megatron.bridge import AutoBridge

from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", default="/workspace/storage/checkpoints/lightning35-megatron")
    args = parser.parse_args()

    output = Path(args.output)
    marker = output / "latest_checkpointed_iteration.txt"
    if marker.exists():
        print(f"Megatron checkpoint already exists at {output}; skipping conversion.")
        return
    output.mkdir(parents=True, exist_ok=True)
    AutoBridge.import_ckpt(
        args.hf_model,
        str(output),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=args.revision or None,
    )
    if not marker.exists():
        raise RuntimeError(f"Conversion finished without expected marker: {marker}")
    print(f"Converted {args.hf_model}@{args.revision} -> {output}")


if __name__ == "__main__":
    main()
