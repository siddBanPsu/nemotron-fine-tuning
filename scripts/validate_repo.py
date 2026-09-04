#!/usr/bin/env python3
"""Dependency-free structural validation for the launchable repository."""

from __future__ import annotations

import json
import py_compile
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "config/api.local.toml.example",
    "launchable/brev-launchable.yaml",
    "launchable/setup.sh",
    "launchable/container-entrypoint.sh",
    "notebooks/01_cloud_api_baseline.ipynb",
    "notebooks/02_local_bf16_baseline.ipynb",
    "notebooks/03_nano9b_workshop_lora.ipynb",
    "notebooks/04_lightning_advanced_lora.ipynb",
    "notebooks/05_full_finetuning_design.ipynb",
    "scripts/prepare_text2sql.py",
    "scripts/evaluate_vllm.py",
    "scripts/build_notebooks.py",
    "scripts/train_peft.py",
    "scripts/train_full.py",
]


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")

    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise RuntimeError(f"{path} is not nbformat 4")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
                if cell.get("execution_count") is not None or cell.get("outputs"):
                    raise RuntimeError(
                        f"Notebook outputs must be cleared before publishing: {path}:cell-{index}"
                    )

    for path in (ROOT / "launchable/setup.sh", ROOT / "launchable/container-entrypoint.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)

    full_text = (ROOT / "notebooks/05_full_finetuning_design.ipynb").read_text(encoding="utf-8")
    if "DESIGN_ONLY = True" not in full_text or "validate_full_sft_hardware" not in full_text:
        raise RuntimeError("Notebook 05 must remain design-only and hardware-aware.")
    if "subprocess.run(full_cmd" in full_text:
        raise RuntimeError("Notebook 05 must not launch the full-SFT command.")

    excluded_parts = {
        ".git",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv-api",
        "__pycache__",
        "artifacts",
        "checkpoints",
        "storage",
    }
    source_suffixes = {
        ".example",
        ".ipynb",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    source_names = {".gitignore", ".python-version"}
    source_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not excluded_parts.intersection(path.relative_to(ROOT).parts)
        and (path.suffix in source_suffixes or path.name in source_names)
    ]
    tracked_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in source_files)
    if "BANK" + "ING77" in tracked_text or "B77" + "_" in tracked_text:
        raise RuntimeError("Legacy opaque intent-routing task references remain in the repository.")
    secret_patterns = [
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"nvapi-[A-Za-z0-9_-]{16,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ]
    for path in source_files:
        source_text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.search(source_text):
                raise RuntimeError(f"Potential secret matched {pattern.pattern} in {path.relative_to(ROOT)}")

    print("Repository validation passed.")


if __name__ == "__main__":
    main()
