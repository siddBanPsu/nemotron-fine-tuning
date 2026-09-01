#!/usr/bin/env python3
"""Build deterministic train/validation/test artifacts from official BANKING77 splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from transformers import AutoTokenizer

from nemotron_ft_lab.constants import (
    DATASET_ID,
    DATASET_SOURCE_BASE_URL,
    DATASET_SOURCE_REPOSITORY,
    DATASET_SOURCE_REVISION,
    DATASET_SOURCE_SHA256,
    DEFAULT_SEED,
    DEFAULT_TEST_PER_LABEL,
    DEFAULT_TRAIN_PER_LABEL,
    DEFAULT_VALIDATION_PER_LABEL,
    MODEL_ID,
    MODEL_REVISION,
    ROUTE_PERMUTATION_SEED,
)
from nemotron_ft_lab.data import (
    as_prepared,
    render_evaluation_record,
    render_training_record,
    route_code,
    stratified_take,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/data/banking77")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--train-per-label", type=int, default=DEFAULT_TRAIN_PER_LABEL)
    parser.add_argument("--validation-per-label", type=int, default=DEFAULT_VALIDATION_PER_LABEL)
    parser.add_argument("--test-per-label", type=int, default=DEFAULT_TEST_PER_LABEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def with_ids(split: object, prefix: str) -> list[dict[str, object]]:
    return [{**dict(row), "example_id": f"{prefix}-{index:05d}"} for index, row in enumerate(split)]


def download_source(output_dir: Path, name: str) -> tuple[Path, str]:
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    target = source_dir / name
    if not target.exists():
        request = Request(f"{DATASET_SOURCE_BASE_URL}/{name}", headers={"User-Agent": "nemotron-ft-lab/0.1"})
        with urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    expected = DATASET_SOURCE_SHA256[name]
    if digest != expected:
        raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, received {digest}.")
    return target, digest


def load_csv(path: Path, label_to_id: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label_name = str(row["category"])
            rows.append({"text": str(row["text"]), "label": label_to_id[label_name]})
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Prepared data already exists at {output_dir}; use --force to rebuild.")
        return

    train_path, train_sha256 = download_source(output_dir, "train.csv")
    test_path, test_sha256 = download_source(output_dir, "test.csv")
    categories_path, categories_sha256 = download_source(output_dir, "categories.json")
    label_names = json.loads(categories_path.read_text(encoding="utf-8"))
    if len(label_names) != 77:
        raise RuntimeError(f"Expected 77 BANKING77 labels, found {len(label_names)}.")
    label_to_id = {name: index for index, name in enumerate(label_names)}

    train_rows = with_ids(load_csv(train_path, label_to_id), "train")
    test_rows = with_ids(load_csv(test_path, label_to_id), "test")
    if len(train_rows) != 10_003 or len(test_rows) != 3_080:
        raise RuntimeError(
            f"Official BANKING77 row counts changed: train={len(train_rows)}, test={len(test_rows)}."
        )

    validation_raw = stratified_take(
        train_rows,
        per_label=args.validation_per_label,
        seed=args.seed + 10_000,
    )
    validation_ids = {str(row["example_id"]) for row in validation_raw}
    training_raw = stratified_take(
        train_rows,
        per_label=args.train_per_label,
        seed=args.seed,
        excluded_ids=validation_ids,
    )
    test_raw = stratified_take(test_rows, per_label=args.test_per_label, seed=args.seed + 20_000)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision or None,
        trust_remote_code=True,
    )
    training = [render_training_record(as_prepared(row, label_names), tokenizer) for row in training_raw]
    validation = [render_training_record(as_prepared(row, label_names), tokenizer) for row in validation_raw]
    evaluation = [render_evaluation_record(as_prepared(row, label_names), tokenizer) for row in test_raw]

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "training": write_jsonl(output_dir / "training.jsonl", training),
        "validation": write_jsonl(output_dir / "validation.jsonl", validation),
        # Megatron-Bridge accepts a test JSONL, while the notebooks consume the richer eval file.
        "test": write_jsonl(output_dir / "test.jsonl", evaluation),
        "evaluation": write_jsonl(output_dir / "evaluation.jsonl", evaluation),
    }
    label_map = {
        route_code(index): {"label_id": index, "label_name": name}
        for index, name in enumerate(label_names)
    }
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": DATASET_ID,
        "source_repository": DATASET_SOURCE_REPOSITORY,
        "source_revision": DATASET_SOURCE_REVISION,
        "source_sha256": {
            "train.csv": train_sha256,
            "test.csv": test_sha256,
            "categories.json": categories_sha256,
        },
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "official_split_policy": "training and validation are disjoint subsets of train; evaluation is test only",
        "counts": counts,
        "train_per_label": args.train_per_label,
        "validation_per_label": args.validation_per_label,
        "test_per_label": args.test_per_label,
        "route_mapping": {
            "description": "opaque one-to-one permutation from source category index to route code",
            "seed": ROUTE_PERMUTATION_SEED,
        },
        "train_example_ids": sorted(str(row["example_id"]) for row in training_raw),
        "validation_example_ids": sorted(validation_ids),
        "test_example_ids": sorted(str(row["example_id"]) for row in test_raw),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    longest = max(row["length"] for row in training + validation)
    print(f"Prepared {counts} under {output_dir}")
    print(f"Longest rendered train/validation example: {longest} tokens")


if __name__ == "__main__":
    main()
