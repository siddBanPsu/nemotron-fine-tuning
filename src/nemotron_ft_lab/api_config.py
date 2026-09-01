"""Public NVIDIA API defaults with optional private, local-only overrides."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    NVIDIA_API_BASE_URL,
    NVIDIA_API_LIGHTNING_MODEL_ID,
    NVIDIA_API_LIGHTNING_MODEL_VARIANT,
    NVIDIA_API_ULTRA_MODEL_ID,
    NVIDIA_API_ULTRA_MODEL_VARIANT,
)


@dataclass(frozen=True)
class NvidiaApiModelConfig:
    model_id: str
    served_variant: str


@dataclass(frozen=True)
class NvidiaApiConfig:
    profile_name: str
    base_url: str
    requests_per_minute: float
    timeout_seconds: float
    lightning: NvidiaApiModelConfig
    ultra: NvidiaApiModelConfig
    source_label: str

    @property
    def artifact_prefix(self) -> str:
        """Keep legacy public filenames while isolating non-public results."""
        if self.profile_name == "public":
            return ""
        slug = re.sub(r"[^a-z0-9]+", "_", self.profile_name.lower()).strip("_")
        return f"{slug}_"


def _public_values() -> dict[str, Any]:
    return {
        "profile_name": "public",
        "base_url": NVIDIA_API_BASE_URL,
        "requests_per_minute": 30.0,
        "timeout_seconds": 90.0,
        "models": {
            "lightning": {
                "model_id": NVIDIA_API_LIGHTNING_MODEL_ID,
                "served_variant": NVIDIA_API_LIGHTNING_MODEL_VARIANT,
            },
            "ultra": {
                "model_id": NVIDIA_API_ULTRA_MODEL_ID,
                "served_variant": NVIDIA_API_ULTRA_MODEL_VARIANT,
            },
        },
    }


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"NVIDIA API configuration does not exist: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    unknown = set(payload).difference(
        {"profile_name", "base_url", "requests_per_minute", "timeout_seconds", "models"}
    )
    if unknown:
        raise ValueError(f"Unknown NVIDIA API configuration keys: {sorted(unknown)}")
    models = payload.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("NVIDIA API configuration 'models' must be a table.")
    unknown_models = set(models).difference({"lightning", "ultra"})
    if unknown_models:
        raise ValueError(f"Unknown NVIDIA API model sections: {sorted(unknown_models)}")
    for name, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"NVIDIA API model section {name!r} must be a table.")
        unknown_model_keys = set(model).difference({"model_id", "served_variant"})
        if unknown_model_keys:
            raise ValueError(
                f"Unknown keys in NVIDIA API model section {name!r}: {sorted(unknown_model_keys)}"
            )
    return payload


def _merge(values: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key in ("profile_name", "base_url", "requests_per_minute", "timeout_seconds"):
        if key in override:
            values[key] = override[key]
    for name, model_override in override.get("models", {}).items():
        values["models"][name].update(model_override)
        if "model_id" in model_override and "served_variant" not in model_override:
            values["models"][name]["served_variant"] = model_override["model_id"]


def load_nvidia_api_config(
    repository_root: str | Path,
    *,
    profile: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> NvidiaApiConfig:
    """Resolve public defaults, an ignored local TOML file, and environment overrides.

    The explicit ``profile`` argument is intended for a notebook-level switch
    and takes precedence over ``NEMOTRON_API_PROFILE``. A public profile forces
    tracked defaults even when a local
    configuration exists. ``NEMOTRON_API_PROFILE=local`` requires the default
    ``config/api.local.toml``. ``NEMOTRON_API_CONFIG`` selects another TOML
    path. With neither variable, a local file is auto-selected when present.
    """
    root = Path(repository_root).resolve()
    env = os.environ if environ is None else environ
    selector_value = profile if profile is not None else env.get("NEMOTRON_API_PROFILE", "auto")
    selector = str(selector_value).strip().lower()
    if selector not in {"auto", "public", "local"}:
        raise ValueError("NEMOTRON_API_PROFILE must be 'auto', 'public', or 'local'.")

    explicit_path = env.get("NEMOTRON_API_CONFIG", "").strip()
    local_path = root / "config/api.local.toml"
    selected_path: Path | None = None
    if explicit_path and selector != "public":
        candidate = Path(explicit_path).expanduser()
        selected_path = candidate if candidate.is_absolute() else root / candidate
    elif selector == "local":
        selected_path = local_path
    elif selector == "auto" and local_path.is_file():
        selected_path = local_path

    values = _public_values()
    source_label = "tracked public defaults"
    if selected_path is not None:
        file_override = _load_toml(selected_path)
        _merge(values, file_override)
        if "profile_name" not in file_override:
            values["profile_name"] = "local"
        source_label = selected_path.name

    environment_overrides = {
        "profile_name": env.get("NEMOTRON_API_PROFILE_NAME"),
        "base_url": env.get("NEMOTRON_API_BASE_URL"),
        "requests_per_minute": env.get("NEMOTRON_API_REQUESTS_PER_MINUTE"),
        "timeout_seconds": env.get("NEMOTRON_API_TIMEOUT_SECONDS"),
    }
    used_environment_override = False
    if selector != "public":
        for key, value in environment_overrides.items():
            if value is not None and value.strip():
                values[key] = value.strip()
                used_environment_override = True

        for name, prefix in (("lightning", "LIGHTNING"), ("ultra", "ULTRA")):
            model_id = env.get(f"NEMOTRON_API_{prefix}_MODEL_ID", "").strip()
            variant = env.get(f"NEMOTRON_API_{prefix}_MODEL_VARIANT", "").strip()
            if model_id:
                values["models"][name]["model_id"] = model_id
                if not variant:
                    values["models"][name]["served_variant"] = model_id
                used_environment_override = True
            if variant:
                values["models"][name]["served_variant"] = variant
                used_environment_override = True

    if used_environment_override:
        source_label += " + environment"
        if values["profile_name"] == "public":
            values["profile_name"] = "custom"

    profile_name = str(values["profile_name"]).strip().lower()
    if not profile_name or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", profile_name):
        raise ValueError("profile_name must contain only lowercase letters, digits, '_' or '-'.")
    base_url = str(values["base_url"]).strip().rstrip("/")
    if not base_url.startswith(("https://", "http://")):
        raise ValueError("NVIDIA API base_url must start with http:// or https://.")
    requests_per_minute = float(values["requests_per_minute"])
    timeout_seconds = float(values["timeout_seconds"])
    if requests_per_minute <= 0 or timeout_seconds <= 0:
        raise ValueError("NVIDIA API rate and timeout values must be positive.")

    def model_config(name: str) -> NvidiaApiModelConfig:
        model = values["models"][name]
        model_id = str(model["model_id"]).strip()
        served_variant = str(model.get("served_variant") or model_id).strip()
        if not model_id:
            raise ValueError(f"NVIDIA API {name} model_id cannot be empty.")
        return NvidiaApiModelConfig(model_id=model_id, served_variant=served_variant)

    return NvidiaApiConfig(
        profile_name=profile_name,
        base_url=base_url,
        requests_per_minute=requests_per_minute,
        timeout_seconds=timeout_seconds,
        lightning=model_config("lightning"),
        ultra=model_config("ultra"),
        source_label=source_label,
    )
