#!/usr/bin/env python3
"""Prepare official-recipe BIRD train data and a frozen executable Mini-Dev holdout."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nemotron_ft_lab.constants import (
    DEFAULT_EVALUATION_SIZE,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_SEED,
    EVAL_DATASET_NAME,
    EVAL_EXCLUDED_QUESTION_IDS,
    EVALUATION_PROTOCOL_VERSION,
    MINIDEV_ARCHIVE_BYTES,
    MINIDEV_ARCHIVE_FILE_ID,
    MINIDEV_ARCHIVE_SHA256,
    MINIDEV_ARCHIVE_URL,
    OFFICIAL_COOKBOOK_REPOSITORY,
    OFFICIAL_COOKBOOK_REVISION,
    REASONING_DATASET_ID,
    REASONING_DATASET_REVISION,
    TRAIN_DATASET_ID,
    TRAIN_DATASET_REVISION,
    TRAINING_PROTOCOL_VERSION,
)
from nemotron_ft_lab.data import (
    balanced_evaluation_subset,
    determine_eot_marker,
    find_minidev_root,
    render_evaluation_record,
    render_training_record,
    safe_extract_zip,
    schema_from_sqlite,
    sha256_file,
    write_jsonl,
)
from nemotron_ft_lab.model_profiles import get_model_profile


def _download_with_resume(url: str, destination: Path, expected_bytes: int, expected_sha256: str) -> str:
    if destination.exists() and destination.stat().st_size == expected_bytes:
        digest = sha256_file(destination)
        if digest != expected_sha256:
            raise RuntimeError(f"Existing Mini-Dev archive failed SHA-256 validation: {destination}")
        print(f"Official Mini-Dev archive already exists and is verified: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists() and partial.stat().st_size == expected_bytes:
        digest = sha256_file(partial)
        if digest == expected_sha256:
            partial.replace(destination)
            print(f"Completed Mini-Dev archive was recovered from {partial.name}.")
            return digest
        partial.unlink()
    elif partial.exists() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    print(f"Downloading official BIRD Mini-Dev package ({expected_bytes / 1024**2:.0f} MiB)...")
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and getattr(response, "status", None) != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        downloaded = offset
        next_report = downloaded + 128 * 1024 * 1024
        with partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  downloaded {downloaded / 1024**2:.0f} MiB", flush=True)
                    next_report += 128 * 1024 * 1024
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Mini-Dev archive size is {partial.stat().st_size}, expected {expected_bytes}. "
            "Leave the .partial file in place and rerun to resume."
        )
    partial.replace(destination)
    digest = sha256_file(destination)
    if digest != expected_sha256:
        raise RuntimeError(f"Mini-Dev archive SHA-256 is {digest}, expected {expected_sha256}.")
    return digest


def prepare_evaluation(output_dir: Path, *, size: int, seed: int, force: bool) -> None:
    evaluation_path = output_dir / "evaluation.jsonl"
    manifest_path = output_dir / "evaluation_manifest.json"
    if evaluation_path.exists() and manifest_path.exists() and not force:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        prepared_rows = [
            json.loads(line)
            for line in evaluation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        bundle_complete = len(prepared_rows) == size and all(
            (output_dir / str(row["database_path"])).is_file() for row in prepared_rows
        )
        evaluation_hash_matches = current.get("evaluation_sha256") == sha256_file(evaluation_path)
        if (
            current.get("evaluation_size") == size
            and current.get("seed") == seed
            and current.get("evaluation_protocol_version") == EVALUATION_PROTOCOL_VERSION
            and current.get("archive", {}).get("sha256") == MINIDEV_ARCHIVE_SHA256
            and bundle_complete
            and evaluation_hash_matches
        ):
            print(f"Frozen evaluation bundle already exists: {evaluation_path}")
            return

    source_dir = output_dir / "source"
    archive = source_dir / "minidev_0703.zip"
    extracted = source_dir / "minidev_0703"
    archive_digest = _download_with_resume(
        MINIDEV_ARCHIVE_URL,
        archive,
        MINIDEV_ARCHIVE_BYTES,
        MINIDEV_ARCHIVE_SHA256,
    )
    try:
        mini_dev_root = find_minidev_root(extracted)
    except RuntimeError:
        print(f"Extracting official Mini-Dev package to {extracted}...")
        safe_extract_zip(archive, extracted)
        mini_dev_root = find_minidev_root(extracted)

    rows = json.loads((mini_dev_root / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    from nemotron_ft_lab.evaluation import execute_read_only_sql

    eligible = [row for row in rows if int(row["question_id"]) not in EVAL_EXCLUDED_QUESTION_IDS]
    selected = balanced_evaluation_subset(eligible, size=size, seed=seed)
    gold_failures = []
    for row in selected:
        db_id = str(row["db_id"])
        database = mini_dev_root / "dev_databases" / db_id / f"{db_id}.sqlite"
        try:
            execute_read_only_sql(database, str(row["SQL"]), timeout_seconds=30.0)
        except Exception as exc:
            gold_failures.append((int(row["question_id"]), type(exc).__name__, str(exc)[:200]))
    if gold_failures:
        raise RuntimeError(
            f"Frozen Mini-Dev gold rehearsal failed; IDs were not resampled: {gold_failures[:5]}"
        )

    schemas: dict[str, str] = {}
    prepared = []
    for row in selected:
        db_id = str(row["db_id"])
        database = mini_dev_root / "dev_databases" / db_id / f"{db_id}.sqlite"
        if not database.is_file():
            raise RuntimeError(f"Missing Mini-Dev database for {db_id}: {database}")
        schemas.setdefault(db_id, schema_from_sqlite(database))
        prepared.append(
            render_evaluation_record(
                row,
                schema=schemas[db_id],
                database_relative_path=database.relative_to(output_dir).as_posix(),
            )
        )

    write_jsonl(evaluation_path, prepared)
    manifest = {
        "task": "BIRD Mini-Dev SQLite Text2SQL execution",
        "evaluation_dataset": EVAL_DATASET_NAME,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_size": len(prepared),
        "evaluation_sha256": sha256_file(evaluation_path),
        "seed": seed,
        "selection": "difficulty-proportional and database-spread before any model is scored",
        "gold_rehearsal": "all selected reference queries execute within 30 seconds locally",
        "excluded_gold_execution_question_ids": list(EVAL_EXCLUDED_QUESTION_IDS),
        "exclusion_reason": "question 701 exceeded 30 seconds in the repository audit; exclusion is pinned",
        "question_ids": [row["question_id"] for row in prepared],
        "difficulty_distribution": dict(sorted(Counter(row["difficulty"] for row in prepared).items())),
        "database_distribution": dict(sorted(Counter(row["db_id"] for row in prepared).items())),
        "archive": {
            "source_url": MINIDEV_ARCHIVE_URL,
            "google_drive_file_id": MINIDEV_ARCHIVE_FILE_ID,
            "bytes": archive.stat().st_size,
            "sha256": archive_digest,
        },
        "split_policy": "training uses BIRD train mirrors; evaluation uses only BIRD Mini-Dev",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(prepared)} executable evaluation rows at {evaluation_path}")


def prepare_training(
    output_dir: Path,
    *,
    model_profile: str,
    model: str,
    revision: str,
    system_prompt: str,
    max_sequence_length: int,
    max_train_samples: int,
    seed: int,
    include_reasoning: bool,
    force: bool,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    training_path = output_dir / "training.jsonl"
    manifest_path = output_dir / "training_manifest.json"
    signature = {
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "model_profile": model_profile,
        "model": model,
        "revision": revision,
        "system_prompt": system_prompt,
        "max_sequence_length": max_sequence_length,
        "max_train_samples": max_train_samples,
        "seed": seed,
        "include_reasoning": include_reasoning,
    }
    if training_path.exists() and manifest_path.exists() and not force:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(current.get(key) == value for key, value in signature.items()) and current.get(
            "training_sha256"
        ) == sha256_file(training_path):
            print(f"Prepared training data already exists: {training_path}")
            return

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
    eot_marker = determine_eot_marker(tokenizer)
    direct = load_dataset(TRAIN_DATASET_ID, split="train", revision=TRAIN_DATASET_REVISION)
    candidates: list[tuple[dict, bool]] = []
    for index, row in enumerate(direct):
        candidates.append(({**row, "_source_index": index}, False))
    if include_reasoning:
        reasoning = load_dataset(
            REASONING_DATASET_ID,
            split="train",
            revision=REASONING_DATASET_REVISION,
        )
        for index, row in enumerate(reasoning):
            candidates.append(({**row, "_source_index": index}, True))
    random.Random(seed).shuffle(candidates)

    rendered = []
    filtered_for_length = 0
    for row, is_reasoning in candidates:
        record = render_training_record(
            row,
            tokenizer,
            include_reasoning=is_reasoning,
            eot_marker=eot_marker,
            system_prompt=system_prompt,
        )
        if record["length"] > max_sequence_length:
            filtered_for_length += 1
            continue
        rendered.append(record)
        if max_train_samples and len(rendered) >= max_train_samples:
            break
    if not rendered:
        raise RuntimeError("All training examples were filtered; raise --max-sequence-length.")
    rendered.sort(key=lambda row: int(row["length"]))
    write_jsonl(training_path, rendered)
    source_counts = Counter(row["source"] for row in rendered)
    manifest = {
        **signature,
        "task": "BIRD Text2SQL supervised fine-tuning",
        "official_cookbook": OFFICIAL_COOKBOOK_REPOSITORY,
        "official_cookbook_revision": OFFICIAL_COOKBOOK_REVISION,
        "train_dataset": TRAIN_DATASET_ID,
        "train_dataset_revision": TRAIN_DATASET_REVISION,
        "reasoning_dataset": REASONING_DATASET_ID if include_reasoning else None,
        "reasoning_dataset_revision": REASONING_DATASET_REVISION if include_reasoning else None,
        "training_examples": len(rendered),
        "source_distribution": dict(sorted(source_counts.items())),
        "filtered_for_length_before_cap": filtered_for_length,
        "total_tokens": sum(int(row["length"]) for row in rendered),
        "training_sha256": sha256_file(training_path),
        "split_policy": "BIRD train only; BIRD Mini-Dev is never read by training",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rendered)} training rows ({dict(source_counts)}) at {training_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    artifacts = Path(os.environ.get("NEMOTRON_ARTIFACTS_DIR", "artifacts"))
    parser.add_argument("--output-dir", type=Path, default=artifacts / "data/bird-text2sql")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluation-only", action="store_true")
    mode.add_argument("--training-only", action="store_true")
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--evaluation-size", type=int, default=DEFAULT_EVALUATION_SIZE)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    reasoning = parser.add_mutually_exclusive_group()
    reasoning.add_argument("--with-reasoning", action="store_true")
    reasoning.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_model_profile(args.model_profile)
    model = args.model or profile.model_id
    revision = profile.revision if args.revision is None else args.revision
    max_train_samples = (
        profile.default_train_samples if args.max_train_samples is None else args.max_train_samples
    )
    include_reasoning = profile.include_reasoning
    if args.with_reasoning:
        include_reasoning = True
    elif args.no_reasoning:
        include_reasoning = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.training_only:
        prepare_evaluation(args.output_dir, size=args.evaluation_size, seed=args.seed, force=args.force)
    if not args.evaluation_only:
        prepare_training(
            args.output_dir,
            model_profile=profile.name,
            model=model,
            revision=revision,
            system_prompt=profile.system_prompt,
            max_sequence_length=args.max_sequence_length,
            max_train_samples=max_train_samples,
            seed=args.seed,
            include_reasoning=include_reasoning,
            force=args.force,
        )


if __name__ == "__main__":
    main()
