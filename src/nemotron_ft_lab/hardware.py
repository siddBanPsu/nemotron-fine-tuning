"""Hardware gates that prevent accidental full-SFT launches on undersized nodes."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any


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
    except ModuleNotFoundError as exc:
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


def validate_inference_hardware(inventory: HardwareInventory) -> None:
    if inventory.gpu_count < 1:
        raise RuntimeError("Local BF16 inference requires at least one CUDA GPU.")
    if inventory.total_memory_gib < 75:
        raise RuntimeError(
            "The local BF16 checkpoint needs one 80 GB GPU or at least 75 GiB "
            "aggregate visible VRAM."
        )


def validate_peft_hardware(inventory: HardwareInventory) -> None:
    if inventory.gpu_count < 1:
        raise RuntimeError("PEFT requires at least one CUDA GPU.")
    if inventory.total_memory_gib < 75:
        raise RuntimeError(
            "The BF16 PEFT path needs about 79 GiB on one H100 at the official 2K profile. "
            "Use one 80 GB GPU or shard across enough GPUs to provide at least 75 GiB total."
        )


def validate_full_sft_hardware(inventory: HardwareInventory) -> None:
    if inventory.gpu_count < 8 or inventory.total_memory_gib < 600:
        raise RuntimeError(
            "True full-parameter SFT is intentionally gated: provide at least 8 GPUs and "
            "600 GiB aggregate VRAM. NVIDIA's currently verified H100 reference uses "
            "16x H100 80 GB (TP=2, EP=8). Use Notebook 03 on a single 80 GB GPU."
        )
