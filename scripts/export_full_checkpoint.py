#!/usr/bin/env python3
"""Export a trained Megatron checkpoint to a standard Hugging Face directory on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from megatron.bridge import AutoBridge

from nemotron_ft_lab.constants import MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()

    source = Path(args.megatron_checkpoint)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)
    if (output / "config.json").exists():
        print(f"Hugging Face export already exists at {output}; skipping.")
        return
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model,
        revision=args.revision or None,
        trust_remote_code=True,
    )
    bridge.export_ckpt(str(source), str(output), show_progress=True, strict=False)
    if not (output / "config.json").exists():
        raise RuntimeError(f"Export finished without config.json under {output}")
    print(f"Exported {source} -> {output}")


if __name__ == "__main__":
    main()
