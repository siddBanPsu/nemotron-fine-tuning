"""BIRD Text2SQL preparation helpers shared by the five notebooks."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def text2sql_user_content(schema: str, question: str, evidence: str = "") -> str:
    """Match the user-message layout in NVIDIA's official Text2SQL cookbook."""
    sections = [schema.strip(), question.strip()]
    if evidence.strip():
        sections.append(evidence.strip())
    return "\n\n".join(sections)


def build_messages(row: dict[str, Any], *, system_prompt: str = "") -> list[dict[str, str]]:
    """Return the inference messages used by cloud and local baselines."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": text2sql_user_content(
                str(row["schema"]), str(row["question"]), str(row.get("evidence", ""))
            ),
        },
    ]


def determine_eot_marker(tokenizer: Any) -> str:
    marker = "__NEMOTRON_TEXT2SQL_MARKER__"
    rendered = tokenizer.apply_chat_template([{"role": "assistant", "content": marker}], tokenize=False)
    if marker not in rendered:
        raise RuntimeError("Tokenizer chat template did not preserve the EOT probe marker.")
    return rendered.split(marker, maxsplit=1)[-1]


def render_training_record(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    include_reasoning: bool,
    eot_marker: str | None = None,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Render one prompt-completion row using Nemotron's native chat template."""
    prompt = tokenizer.apply_chat_template(
        build_messages(row, system_prompt=system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=include_reasoning,
    )
    sql = str(row["SQL"]).strip()
    if include_reasoning:
        reasoning = str(row.get("reasoning_trace", "")).strip()
        if not reasoning:
            raise ValueError("A reasoning example is missing reasoning_trace.")
        completion = f"{reasoning}</think>{sql}"
    else:
        completion = sql
    completion += eot_marker if eot_marker is not None else determine_eot_marker(tokenizer)
    text = prompt + completion
    length = len(tokenizer(text, add_special_tokens=False).input_ids)
    return {
        "input": prompt,
        "output": completion,
        "text": text,
        "length": length,
        "source": "bird_reasoning" if include_reasoning else "bird_direct",
        "source_id": f"{row.get('db_id', 'unknown')}::{row.get('_source_index', 'unknown')}",
    }


def schema_from_sqlite(database_path: str | Path) -> str:
    """Build deterministic DDL context from an official Mini-Dev SQLite file."""
    path = Path(database_path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'view') ORDER BY type, name"
        ).fetchall()
    finally:
        connection.close()
    statements = [str(sql).strip().rstrip(";") + ";" for _, _, sql in rows]
    if not statements:
        raise RuntimeError(f"No table or view DDL found in {path}.")
    return "\n".join(statements)


def _difficulty_targets(rows: Sequence[dict[str, Any]], size: int) -> dict[str, int]:
    counts = Counter(str(row["difficulty"]) for row in rows)
    if size <= 0 or size > len(rows):
        raise ValueError(f"Evaluation size must be in [1, {len(rows)}], received {size}.")
    raw = {name: size * count / len(rows) for name, count in counts.items()}
    targets = {name: int(value) for name, value in raw.items()}
    remainder = size - sum(targets.values())
    for name in sorted(raw, key=lambda item: (-(raw[item] - targets[item]), item))[:remainder]:
        targets[name] += 1
    return targets


def balanced_evaluation_subset(
    rows: Sequence[dict[str, Any]], *, size: int, seed: int
) -> list[dict[str, Any]]:
    """Freeze a difficulty-proportional, database-spread Mini-Dev subset."""
    targets = _difficulty_targets(rows, size)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row["difficulty"]), str(row["db_id"]))].append(dict(row))
    for candidates in by_stratum.values():
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['question_id']}".encode()).hexdigest())

    selected: list[dict[str, Any]] = []
    for difficulty in sorted(targets):
        databases = sorted({db for diff, db in by_stratum if diff == difficulty})
        random.Random(f"{seed}:{difficulty}").shuffle(databases)
        queues = {db: list(by_stratum[(difficulty, db)]) for db in databases}
        cursor = 0
        for _ in range(targets[difficulty]):
            available = [db for db in databases if queues[db]]
            if not available:
                raise RuntimeError(f"Not enough {difficulty} Mini-Dev rows for the requested subset.")
            db = available[cursor % len(available)]
            selected.append(queues[db].pop(0))
            cursor += 1
    selected.sort(key=lambda row: int(row["question_id"]))
    return selected


def render_evaluation_record(
    row: dict[str, Any], *, schema: str, database_relative_path: str
) -> dict[str, Any]:
    return {
        "example_id": f"bird-mini-dev-{int(row['question_id']):04d}",
        "question_id": int(row["question_id"]),
        "db_id": str(row["db_id"]),
        "difficulty": str(row["difficulty"]),
        "question": str(row["question"]).strip(),
        "evidence": str(row.get("evidence", "")).strip(),
        "schema": schema,
        "expected_sql": str(row["SQL"]).strip(),
        "database_path": database_relative_path,
    }


def safe_extract_zip(archive: str | Path, destination: str | Path) -> None:
    """Extract a trusted archive while still rejecting path traversal entries."""
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination_path / member.filename).resolve()
            if destination_path != target and destination_path not in target.parents:
                raise RuntimeError(f"Archive member escapes destination: {member.filename}")
        handle.extractall(destination_path)


def find_minidev_root(root: str | Path) -> Path:
    matches = sorted(Path(root).rglob("mini_dev_sqlite.json"))
    candidates = [path.parent for path in matches if (path.parent / "dev_databases").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one Mini-Dev root under {root}, found {len(candidates)}.")
    return candidates[0]


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
