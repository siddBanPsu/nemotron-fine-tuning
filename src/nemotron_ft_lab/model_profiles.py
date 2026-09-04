"""Pinned model profiles for the workshop and advanced Text2SQL paths."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """Everything that must change together when the local base model changes."""

    name: str
    display_name: str
    model_id: str
    revision: str
    recipe_name: str
    artifact_slug: str
    megatron_checkpoint_name: str
    lora_checkpoint_name: str
    merged_checkpoint_name: str
    system_prompt: str
    enable_thinking: bool
    include_reasoning: bool
    default_train_samples: int
    default_max_steps: int
    default_global_batch_size: int
    peft_world_sizes: tuple[int, ...]
    min_inference_vram_gib: float
    min_peft_vram_gib: float
    mamba_ssm_cache_dtype: str | None
    full_sft_supported: bool
    support_note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


NANO9B_WORKSHOP = ModelProfile(
    name="nano9b_workshop",
    display_name="NVIDIA Nemotron Nano 9B v2 workshop",
    model_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    revision="6533e8de2c68e4536bf7c411d7a3ce5734111476",
    recipe_name="nemotron_nano_9b_v2_peft_config",
    artifact_slug="nano9b-v2",
    megatron_checkpoint_name="nano9b-v2-megatron",
    lora_checkpoint_name="bird-text2sql-nano9b-lora",
    merged_checkpoint_name="bird-text2sql-nano9b-lora-hf",
    system_prompt="/no_think",
    enable_thinking=False,
    include_reasoning=False,
    default_train_samples=2048,
    default_max_steps=32,
    default_global_batch_size=32,
    peft_world_sizes=(1,),
    min_inference_vram_gib=20.0,
    min_peft_vram_gib=45.0,
    mamba_ssm_cache_dtype="float32",
    full_sft_supported=False,
    support_note=(
        "The pinned Megatron-Bridge commit includes a one-GPU H100 BF16 PEFT recipe. "
        "Current upstream documentation marks Nemotron Nano v2 support deprecated, so "
        "this is a compatibility-frozen workshop path rather than a forward-maintained recipe."
    ),
)


LIGHTNING35_ADVANCED = ModelProfile(
    name="lightning35_advanced",
    display_name="NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16 advanced",
    model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    revision="b3caaabed0263651a17dc1f2d4ce97e794f76c44",
    recipe_name="nemotron_3_5_lightning_peft_config",
    artifact_slug="lightning35",
    megatron_checkpoint_name="lightning35-megatron",
    lora_checkpoint_name="bird-text2sql-lightning35-lora",
    merged_checkpoint_name="bird-text2sql-lightning35-lora-hf",
    system_prompt="",
    enable_thinking=False,
    include_reasoning=True,
    default_train_samples=4096,
    default_max_steps=64,
    default_global_batch_size=32,
    peft_world_sizes=(1, 2, 4, 8),
    min_inference_vram_gib=75.0,
    min_peft_vram_gib=75.0,
    mamba_ssm_cache_dtype=None,
    full_sft_supported=True,
    support_note=(
        "Advanced path based on NVIDIA's official Nemotron 3.5 Lightning Text2SQL cookbook. "
        "The repository's measured 2xA100 run is multi-hour, not a one-hour workshop path."
    ),
)


MODEL_PROFILES = {
    profile.name: profile for profile in (NANO9B_WORKSHOP, LIGHTNING35_ADVANCED)
}
DEFAULT_MODEL_PROFILE_NAME = NANO9B_WORKSHOP.name
DEFAULT_MODEL_PROFILE = NANO9B_WORKSHOP

_ALIASES = {
    "nano": NANO9B_WORKSHOP.name,
    "nano9b": NANO9B_WORKSHOP.name,
    "nano9b-v2": NANO9B_WORKSHOP.name,
    "lightning": LIGHTNING35_ADVANCED.name,
    "lightning35": LIGHTNING35_ADVANCED.name,
}


def get_model_profile(name: str | None = None) -> ModelProfile:
    """Resolve an explicit name, environment selection, or the workshop default."""

    selected = (name or os.environ.get("NEMOTRON_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE_NAME).strip()
    selected = _ALIASES.get(selected, selected)
    try:
        return MODEL_PROFILES[selected]
    except KeyError:
        choices = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown model profile {selected!r}; choose one of: {choices}") from None
