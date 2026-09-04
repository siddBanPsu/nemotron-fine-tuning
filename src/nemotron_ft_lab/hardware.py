"""Hardware gates that prevent accidental full-SFT launches on undersized nodes."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any

from .model_profiles import ModelProfile, get_model_profile


@dataclass(frozen=True)
class HardwareInventory:
    gpu_count: int
    gpu_names: tuple[str, ...]
    memory_gib: tuple[float, ...]
    total_memory_gib: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_cuda() -> HardwareInventory:
    try:
        import torch
    except ModuleNotFoundError:
        raise RuntimeError(
            "PyTorch is not installed in the active interpreter "
            f"({sys.executable}). GPU profiles must run inside the pinned NeMo 26.08 "
            "container; the host/API virtual environment is only for Notebook 01. "
            "On Brev, open the custom Jupyter Secure Link on host port 8889."
        ) from None

    if not torch.cuda.is_available():
        return HardwareInventory(0, (), (), 0.0)
    names: list[str] = []
    memory: list[float] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        names.append(props.name)
        memory.append(props.total_memory / 1024**3)
    return HardwareInventory(len(names), tuple(names), tuple(memory), sum(memory))


def _profile(value: str | ModelProfile | None) -> ModelProfile:
    return value if isinstance(value, ModelProfile) else get_model_profile(value)


def validate_inference_hardware(
    inventory: HardwareInventory, model_profile: str | ModelProfile | None = None
) -> None:
    profile = _profile(model_profile)
    if inventory.gpu_count < 1:
        raise RuntimeError("Local BF16 inference requires at least one CUDA GPU.")
    if inventory.total_memory_gib < profile.min_inference_vram_gib:
        raise RuntimeError(
            f"{profile.display_name} local BF16 inference requires at least "
            f"{profile.min_inference_vram_gib:g} GiB aggregate visible VRAM; "
            f"detected {inventory.total_memory_gib:.1f} GiB."
        )


def validate_peft_hardware(
    inventory: HardwareInventory, model_profile: str | ModelProfile | None = None
) -> None:
    profile = _profile(model_profile)
    if inventory.gpu_count < 1:
        raise RuntimeError("PEFT requires at least one CUDA GPU.")
    largest_gpu = max(inventory.memory_gib, default=0.0)
    if largest_gpu < profile.min_peft_vram_gib:
        raise RuntimeError(
            f"{profile.display_name} PEFT requires a GPU with at least "
            f"{profile.min_peft_vram_gib:g} GiB visible VRAM; the largest detected GPU has "
            f"{largest_gpu:.1f} GiB. The documented workshop target remains one 80 GB A100/H100."
        )


def validate_full_sft_hardware(inventory: HardwareInventory) -> None:
    if inventory.gpu_count < 16 or inventory.total_memory_gib < 1_200:
        raise RuntimeError(
            "True full-parameter SFT is intentionally gated to NVIDIA's currently verified "
            "16x H100 80 GB reference (TP=2, EP=8), with at least 1,200 GiB aggregate "
            "visible VRAM. Use Notebook 03 (Nano) or Notebook 04 (Lightning) for PEFT."
        )
