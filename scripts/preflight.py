#!/usr/bin/env python3
"""Report whether the current container is ready for the selected lab path."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nemotron_ft_lab.hardware import (
    inspect_cuda,
    validate_full_sft_hardware,
    validate_inference_hardware,
    validate_peft_hardware,
)
from nemotron_ft_lab.model_profiles import get_model_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("api", "inference", "peft", "full"), default="inference")
    parser.add_argument("--model-profile", default=None)
    args = parser.parse_args()

    model_profile = None if args.profile == "api" else get_model_profile(args.model_profile)

    inventory = None if args.profile == "api" else inspect_cuda()
    checks = {
        "python": sys.version.split()[0],
        "model_profile": (
            model_profile.as_dict() if model_profile is not None else "not required for API baseline"
        ),
        "docker_visible": shutil.which("nvidia-smi") is not None,
        "hardware": inventory.as_dict() if inventory is not None else "not required for API baseline",
        "packages": {},
    }
    for package in (
        "datasets",
        "huggingface_hub",
        "openai",
        "sqlglot",
        "torch",
        "transformers",
        "vllm",
    ):
        checks["packages"][package] = (
            importlib.metadata.version(package) if importlib.util.find_spec(package) else None
        )
    try:
        checks["packages"]["megatron.bridge"] = importlib.util.find_spec("megatron.bridge") is not None
    except ModuleNotFoundError:
        checks["packages"]["megatron.bridge"] = False
    print(json.dumps(checks, indent=2))

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"The launchable pins Python 3.12; detected {sys.version.split()[0]}.")
    if args.profile == "api":
        missing = [
            name for name in ("openai", "sqlglot", "huggingface_hub") if checks["packages"][name] is None
        ]
        if missing:
            raise RuntimeError(f"API baseline dependencies are missing: {missing}")
    elif args.profile == "inference":
        assert inventory is not None
        assert model_profile is not None
        validate_inference_hardware(inventory, model_profile)
        if checks["packages"]["vllm"] is None:
            raise RuntimeError("Local inference requires vLLM from the GPU container.")
    elif args.profile == "peft":
        assert inventory is not None
        assert model_profile is not None
        validate_peft_hardware(inventory, model_profile)
        if not checks["packages"]["megatron.bridge"]:
            raise RuntimeError("PEFT requires the pinned Megatron-Bridge source path.")
    elif args.profile == "full":
        assert inventory is not None
        validate_full_sft_hardware(inventory)


if __name__ == "__main__":
    main()
